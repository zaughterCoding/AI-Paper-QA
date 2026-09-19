"""从 arXiv 拉取语料论文的正文，转成纯文本存到 eval/corpus/。

和其他 scripts 一样，用 conda 环境的 python 直接运行：

    E:\\conda-envs\\paperqa\\python.exe scripts\\fetch_corpus.py
    E:\\conda-envs\\paperqa\\python.exe scripts\\fetch_corpus.py --force

## 为什么要有一个脚本，而不是手工存文件

1. **可复现**：`eval/corpus/*.txt` 是 Task 13 评测集的输入。评测结果如果
   依赖"某天某人手工存的一份文本"，那结果就没有意义。
2. **可追溯**：`eval/corpus/sources.json` 记下每篇的 arXiv id 和实际抓取的 URL，
   将来能核对"这份文本到底是哪个版本"。
3. **可重建**：要换论文、要加论文，改下面的 PAPERS 重跑即可。

## 为什么抓 HTML 版而不是 PDF

arXiv 从 2023 年起为论文提供官方 HTML 版（用 LaTeXML 从作者提交的 LaTeX 源码生成），
老论文也做了回溯。相比自己下 PDF 再提取：

- **不需要新依赖**：PDF 提取要装 pypdf 之类，而且双栏论文的文本顺序经常是乱的
  （左栏半句话接右栏半句话）。HTML 版是结构化的 `<section>` / `<p>`，
  标准库 `html.parser` 就能干净地剥出来。
- **正文和噪声分得开**：参考文献、导航栏、图注在 HTML 里都有明确的标签和 class，
  可以精确剔除。PDF 里只能靠正则猜。
- **仍然是官方源**：文本来自 arXiv 自己，不是第三方转载。

## 抓下来的是什么

`eval/corpus/<slug>.txt`，**只有正文纯文本**，不带标题头——
因为导入时 title 和 content 是分开传给 `POST /documents` 的，
标题混进 content 会让每个片段都带一遍标题。

已经存在的文件默认跳过（幂等）；要重新抓用 `--force`。
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import httpx

# 仓库根目录。用 __file__ 推出来，这样从任何 cwd 运行结果都一样。
CORPUS_DIR = Path(__file__).resolve().parents[1] / "eval" / "corpus"


@dataclass(frozen=True)
class Paper:
    """一篇语料论文。

    slug 同时是文件名和导入时用的标识，所以取成人类看得懂的短名。
    """

    arxiv_id: str
    slug: str
    title: str


# 选这 5 篇的理由：它们正好覆盖这个项目所用技术的谱系，
# 而且彼此术语区分度高——Task 13 要写"expected_source 明确"的问题，
# 语料之间如果主题重叠，"这段该出自哪篇"本身就说不清。
PAPERS: list[Paper] = [
    Paper("1706.03762", "attention-is-all-you-need", "Attention Is All You Need"),
    Paper("1810.04805", "bert", "BERT: Pre-training of Deep Bidirectional Transformers"),
    Paper("2005.11401", "retrieval-augmented-generation", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"),
    Paper("2004.04906", "dense-passage-retrieval", "Dense Passage Retrieval for Open-Domain Question Answering"),
    Paper("1908.10084", "sentence-bert", "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"),
]


class ArxivHtmlTextExtractor(HTMLParser):
    """把 arXiv 的 HTML 正文剥成纯文本。

    它做的不是"去标签"，而是**挑内容**——arXiv 的 HTML 里，
    正文和噪声在标签层面就是分开的，所以可以精确地只保留正文。

    ## 边界：从哪开始收

    页面顶部有一堆和论文无关的东西：反馈表单、公告横幅、许可声明、
    版权声明。它们也在 HTML 里，纯靠"去标签"是去不掉的。

    好在 arXiv 的正文有明确的开头标记：`<h1 class="ltx_title_document">`
    （论文标题）。**在遇到它之前，什么都不收。** 标题本身留着——
    人是靠它认出这份文本是哪篇的，而且"这篇论文讲了什么"这类问题
    本来就该匹配到开头。

    找不到这个标记时直接报错（见 `extract_text`），而不是降级成
    "从头开始收"——那会静默地把许可声明当成论文内容喂进向量库，
    而且没有任何信号。（同样是 F-39 的教训：宁可响亮地失败。）

    ## 整体丢弃的内容

    - `script` / `style`：不是给人看的内容。
    - `nav` / `header` / `footer`：页面导航、"下载 PDF"之类的按钮。
    - `svg`：图片的矢量路径，展开是坐标数字。
    - `math` 的**内部**（也就是 MathML 本身）：展开后是一堆 `mi`/`mo`/`mn`
      的字形碎片（`θ`、`√`、`∑` 和大量单字母），对检索是纯噪声。
      但公式本身**不能整个丢掉**——见下面。

    ## 公式：丢内容，留 alttext

    整体丢掉 `<math>` 是最省事的，但会把句子弄断。Attention 那篇里：

        "…a neighborhood of size <math/> in the input sequence…"
        "…would increase the maximum path length to <math/>."

    丢完就成了 "of size in the input sequence"、"to ." —— 语义缺了一块，
    这种残句喂给 embedding 模型，质量是打折的。

    好在 LaTeXML 在每个 `<math>` 上都放了 `alttext` 属性，内容就是这段公式的
    LaTeX 源码（`alttext="h_{t}"`、`alttext="N=6"`、`alttext="O(n/r)"`）。
    于是策略是：**丢掉 MathML 内部，把 alttext 当作行内文本插回去**。

    插回去之前要做轻度清洗（`_latex_to_text`），把 `\\mathrm{softmax}` 变成
    `softmax` 之类。清洗的目标不是渲染出漂亮的公式，而是**让关键词能被检索到、
    让人能读懂句子**——所以只处理最碍眼的包装命令，其余原样保留。
    - `figure` / `table`：图注被单独切出来后没有图，读起来不知所云；
      表格拍平成线性文本后行列关系全丢（"28.4 41.8" 是哪一行哪一列？）。
      宁可不要，也不要给检索器喂这种半截语义。
    - `cite`：正文里的 `[1]` 引用标记。参考文献已经整段丢掉了，
      留着这些编号只会变成孤立噪声。
    - `class="ltx_authors"`：作者区。那里混着大量脚注
      （"††thanks: Equal contribution. Listing order is random. …"），
      是作者贡献声明，不是论文内容。作者名对问答也没有价值。
      注意这里是按 **class 精确匹配**（拆成 token 比），
      不能用子串——`class="ltx_document ltx_authors_1line"` 里也含 "ltx_authors" 这几个字，
      子串匹配会把整个 `<article>` 误判成作者区、丢掉全篇。

    ## 从哪结束

    `<section id="bib">`（参考文献）之后的所有内容都不要。
    它篇幅很大，而且几乎不可能是一个问题的答案——却非常容易"蹭"到高分，
    因为它包含了全篇出现过的所有术语。
    """

    # 这些标签**连同内部内容**一起丢。
    # 注意 `math` 不在这里——它的内容要丢，但它开标签上的 alttext 要留，
    # 所以走 handle_starttag 里的专门分支。
    DROP_TAGS = {"script", "style", "nav", "header", "footer", "svg", "figure", "table", "cite"}

    # 这些 class 命中的元素**连同内部内容**一起丢（精确匹配整个 class token）
    DROP_CLASSES = {"ltx_authors"}

    # 这些标签是块级的，前后要断行，段落才分得开
    BLOCK_TAGS = {"p", "div", "section", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol", "br"}

    # 正文开始的标记。见到它之前的全部内容都不要。
    START_CLASS = "ltx_title_document"

    def __init__(self) -> None:
        # convert_charrefs=True（默认）会自动把 &amp; &#x2019; 之类的实体还原成字符，
        # 否则文本里会残留 "&amp;" 这种东西，直接影响 embedding 质量。
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        # 所有还没闭合的标签名，按嵌套顺序。
        self._open_tags: list[str] = []
        # 丢弃到哪一层为止：值是 `_open_tags` 的长度阈值。
        # `None` 表示不在丢弃状态。
        #
        # 为什么记的是**深度**而不是"被丢的标签名"：
        # 最初的做法是维护一个"被丢弃标签"的栈，靠"标签名和栈顶对不对得上"
        # 判断何时停止丢弃。但只要被丢弃的元素内部还有**同名**标签
        # （作者区里嵌一个 `<div>` 就够了），内层的结束标签就会把外层的
        # 状态一起清掉——作者名于是漏进语料。不报错，只是脏了。
        # 记深度就没有这个问题：只有整个子树真的闭合了，深度才会降回去。
        self._drop_until: int | None = None
        # 走到参考文献了吗。到了就什么都不再收。
        self._in_bibliography = False
        # 遇到正文开始标记了吗。
        self._started = False

    @classmethod
    def _is_droppable(cls, tag: str, attrs: list[tuple[str, str | None]]) -> bool:
        if tag in cls.DROP_TAGS:
            return True
        classes = set((dict(attrs).get("class") or "").split())
        return bool(cls.DROP_CLASSES & classes)

    def _pop_tag(self, tag: str) -> None:
        """弹出一个结束标签。

        HTML 正常是嵌套闭合的，栈顶就该是它。但 arXiv 的 HTML 偶尔会有
        错位（比如 `<figure><svg></figure></svg>`），这时候如果只认栈顶，
        丢弃状态就永远解除不了，后面整篇文章都收不到——而输出只是"变短了"，
        没有任何别的信号。所以错位时往下找匹配的那一层，连同上面的一起弹掉。
        """
        if not self._open_tags:
            return
        if self._open_tags[-1] == tag:
            self._open_tags.pop()
            return
        if tag in self._open_tags:
            while self._open_tags and self._open_tags.pop() != tag:
                pass

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._in_bibliography:
            return
        self._open_tags.append(tag)

        if self._drop_until is not None:
            # 已经在丢弃状态里，什么都不用判断——深度记录会自动兜住整棵子树。
            return

        if tag == "section" and dict(attrs).get("id") == "bib":
            self._in_bibliography = True
            return

        if self._is_droppable(tag, attrs):
            self._drop_until = len(self._open_tags)
            return

        if self.START_CLASS in set((dict(attrs).get("class") or "").split()):
            self._started = True

        if not self._started:
            return

        if tag == "math":
            # MathML 内部照丢，但把开标签上的 alttext（LaTeX 源码）插回去，
            # 这样 "of size $r$ in" 不会变成 "of size in"。
            # 前后各留一个空格：公式两侧原本多半贴着别的词，
            # 不留空格会粘成 "sizer"；多出来的空格由 _normalize_whitespace 压掉。
            #
            # 复用 _drop_until 来跳过 MathML 内部，而不是另加一个开关：
            # "丢到这一层闭合为止"正是这里需要的行为。
            alttext = dict(attrs).get("alttext")
            if alttext:
                self._parts.append(f" {_latex_to_text(alttext)} ")
            self._drop_until = len(self._open_tags)
            return

        if tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._in_bibliography:
            return

        if self._drop_until is not None:
            self._pop_tag(tag)
            if len(self._open_tags) < self._drop_until:
                self._drop_until = None
            return

        self._pop_tag(tag)

        if not self._started:
            return

        if tag in self.BLOCK_TAGS:
            self._parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """自闭合标签，比如 `<br/>`、`<path/>`。

        必须单独处理：默认实现会依次调用 handle_starttag 和 handle_endtag，
        而这两个都会给块级标签插换行——`<br/>` 于是变成两个换行，
        段落中间凭空多出空行。
        """
        if self._in_bibliography or self._drop_until is not None or not self._started:
            return
        if tag == "br":
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_bibliography or self._drop_until is not None or not self._started:
            return
        self._parts.append(data)

    def text(self) -> str:
        if not self._started:
            raise ValueError(
                f"没找到正文开始标记（class 含 {self.START_CLASS!r} 的元素）。"
                "arXiv 的 HTML 结构可能变了——请打开页面确认，不要直接跳过这个检查，"
                "否则会把页面上的许可声明、导航文字当成论文内容写进语料。"
            )
        return _normalize_whitespace("".join(self._parts))


# LaTeX 里"只是排版包装"的命令：去掉命令本身，保留花括号里的内容。
#     \mathrm{softmax}  →  softmax
#     \textsc{BASE}     →  BASE
# 分两遍跑是为了处理嵌套的一层（\mathbf{\mathrm{x}} → x）。
_LATEX_WRAPPERS = re.compile(
    r"\\(?:mathrm|mathbf|mathit|mathsf|mathtt|mathcal|mathbb|mathfrak|mathscr"
    r"|text|textrm|textit|textbf|textsc|textsf|texttt|textnormal"
    r"|operatorname|mbox"
    # 重音符号：\hat{x} → x。丢掉的"这是个估计值"的信息，
    # 换来读者不用在句子中间碰到一个 \hat。
    r"|hat|tilde|bar|vec|dot|ddot|check|breve)\s*\{([^{}]*)\}"
)

# 常见符号 → Unicode。只收"没有争议、一看就懂"的：
# 换完之后人读得懂，分词器也不会把它切成 "\theta" 这种罕见碎片。
#
# 注意这里**只处理无参数的命令**。`\frac{1}{\sqrt{d_k}}` 这种带参数的、
# 而且参数里还套着花括号的，一律原样保留——理由见下面 _latex_to_text 的说明。
_LATEX_SYMBOLS = {
    # 运算符
    "cdot": "·", "times": "×", "div": "÷", "pm": "±", "mp": "∓",
    "geq": "≥", "leq": "≤", "neq": "≠", "approx": "≈", "equiv": "≡",
    "ll": "≪", "gg": "≫", "propto": "∝", "infty": "∞",
    # 集合与逻辑
    "in": "∈", "notin": "∉", "subset": "⊂", "supset": "⊃",
    "cup": "∪", "cap": "∩", "emptyset": "∅",
    "forall": "∀", "exists": "∃", "neg": "¬", "land": "∧", "lor": "∨",
    # 箭头
    "rightarrow": "→", "to": "→", "leftarrow": "←",
    "Rightarrow": "⇒", "Leftarrow": "⇐", "leftrightarrow": "↔",
    # 省略号与括号
    "cdots": "···", "ldots": "…", "dots": "…",
    "langle": "⟨", "rangle": "⟩", "prime": "′",
    # 大运算符
    "sum": "Σ", "prod": "Π", "int": "∫",
    # 希腊字母
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "ϑ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π",
    "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ",
    "phi": "φ", "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
    # 作者自定义的宏，LaTeXML 没有展开，直接留在源里。
    # 这几篇里 \inR 出现了 15 次（\mathbb{R} 的简写）。
    "inR": "R", "inN": "N", "inZ": "Z", "inC": "C", "inQ": "Q",
}

# 按**长度降序**拼交替式。这很重要：正则的 `|` 是从左往右取第一个能匹配的，
# 如果 `in` 排在 `inR` 前面，`\inR` 会被切成 `∈R`。长的排前面就正确了。
#
# 结尾用 `(?![a-zA-Z])`（后面不是字母）而不是 `\b`：
# `\b` 把下划线也算作单词字符，于是 `\beta_{1}` 里 `\beta` 和 `_` 之间
# **没有**边界，命令就漏掉了——而带下标正是公式里最常见的形式。
# 否定断言只看后面是不是字母，`\beta_{1}` 能匹配、`\thetaX` 仍然不会误伤。
_LATEX_SYMBOL_RE = re.compile(
    r"\\(" + "|".join(sorted(_LATEX_SYMBOLS, key=len, reverse=True)) + r")(?![a-zA-Z])"
)

# 纯空白命令：\, \; \: \! \quad \qquad \hspace{…} —— 换成一个空格
_LATEX_SPACES = re.compile(r"\\[.,;:!>]|\\(?:quad|qquad)\b|\\hspace\*?\{[^{}]*\}")

# 只影响括号大小的命令，没有语义：\left( \right) \big[ …
_LATEX_DELIMITERS = re.compile(r"\\(?:left|right|middle|big|Big|bigg|Bigg)\b")

# 转义字符：LaTeX 用反斜杠转义 % _ & # $ { }，这里还原成字符本身
_LATEX_ESCAPES = re.compile(r"\\([%$#&_{}])")


def _latex_to_text(latex: str) -> str:
    """把公式的 LaTeX 源码收拾成"人能读、检索能命中关键词"的文本。

    目标**不是**渲染出正确的公式。目标是让 `\\mathrm{softmax}(QK^{T})`
    变成 `softmax(QK^{T})` 这种程度——读者知道这里有个 softmax，
    检索 "softmax" 也能命中。

    ## 清洗的边界划在哪

    规则是：**只做单层的、非嵌套的替换**。

    `\\frac{1}{\\sqrt{d_{k}}}` 这类会原样留着。不是因为处理不了——
    写个递归下降的 LaTeX 解析器当然能处理——而是因为**半吊子清洗比不清洗更糟**：
    如果简单的 `\\frac{a}{b}` 被换成了 `a/b`，而嵌套的那种还是 `\\frac{...}{...}`，
    语料里就会同时存在两种写法，出问题时没法用一条规则去搜。
    宁可全部统一保留，也不要一部分换了一部分没换。

    真要完美渲染公式，正确做法是引入 MathML 渲染器，而不是在这里堆正则。
    这个脚本的职责是"把正文取出来"，公式的保真度不在它的承诺范围内。

    实际做的五件事：
    1. 反复去掉排版包装命令（`\\textsc{BASE}` → `BASE`），循环是为了剥掉一层嵌套
    2. 常见符号换成 Unicode（`\\theta` → `θ`、`\\cdot` → `·`）
    3. 空白命令换成空格（`\\,` → ` `）
    4. 去掉括号尺寸命令（`\\left(` → `(`）
    5. 还原转义字符（`\\%` → `%`）
    """
    text = latex

    # 限次而不是 while True：LaTeX 理论上能构造出替换不收敛的输入，
    # 而这是个抓语料的脚本，不值得为这种边角情况冒死循环的风险。
    for _ in range(5):
        replaced = _LATEX_WRAPPERS.sub(r"\1", text)
        if replaced == text:
            break
        text = replaced

    text = _LATEX_SYMBOL_RE.sub(lambda m: _LATEX_SYMBOLS[m.group(1)], text)
    text = _LATEX_DELIMITERS.sub("", text)
    text = _LATEX_SPACES.sub(" ", text)
    text = _LATEX_ESCAPES.sub(r"\1", text)

    # 最后压一次空白：`\,` 这类命令被换成空格，会和公式里原有的空格叠起来
    # （`a \, b` → `a   b`）。公式内部的连续空白本来就没有意义。
    return re.sub(r"\s+", " ", text).strip()


def _normalize_whitespace(raw: str) -> str:
    """把剥出来的文本收拾干净。三件事：

    1. **行内空白压缩**：HTML 源码为了排版，标签之间塞满了缩进和换行，
       直接拼起来会得到大量空白。`\\s` 在 Python 里也匹配 `\\xa0`（不换行空格），
       而 arXiv 的 HTML 里 `\\xa0` 很常见——不处理的话它会混进片段文本，
       既影响 token 计数也影响 embedding。
    2. **标点前不留空格**：公式和引用标记被拿掉之后会留下痕迹——
       原文 "dilated convolutions [19], increasing" 去掉 `[19]` 就成了
       "convolutions , increasing"。英文排版里标点前本来就不该有空格，
       所以这条规则不会误伤正常正文。只处理**后置**标点，
       左括号前面有空格是对的（"word ("）。
    3. **连续空行压成一个**：块级标签会在同一处插入多个换行
       （`</p></div><div><p>` 就是四个），压成空行后段落边界才是干净的 `\\n\\n`。
    """
    lines = []
    for line in raw.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        line = re.sub(r"\s+([,.;:!?%)\]}])", r"\1", line)
        lines.append(line)

    kept: list[str] = []
    previous_blank = True  # 开头不允许出现空行
    for line in lines:
        if line:
            kept.append(line)
            previous_blank = False
        elif not previous_blank:
            kept.append("")
            previous_blank = True

    return "\n".join(kept).strip()


# arXiv 在页面角上印的版本水印，形如：
#     arXiv:1706.03762v7 [cs.CL] 02 Aug 2023
# 这是**唯一**能确定"抓的是论文哪一版"的地方——`/html/1706.03762` 这个地址
# 不带版本号，arXiv 不会重定向到带版本号的 URL，直接就把最新版返回了。
# 不记下来的话，将来论文更新了，没人说得清当前这份语料是哪一版。
_WATERMARK = re.compile(r'<div id="watermark-tr">\s*([^<]+?)\s*</div>')

# 正文短于这个字符数就认为提取失败。
# 5 篇论文都在 2 万字符以上，1 万已经低得离谱——真触发了多半是
# arXiv 改了 HTML 结构，导致正文边界判断失效、只收到了一个零头。
MIN_PLAUSIBLE_CHARS = 10_000


def extract_version(html: str) -> str:
    """取出页面上的版本水印，取不到就返回空串。"""
    match = _WATERMARK.search(html)
    return match.group(1) if match else ""


def extract_text(html: str) -> str:
    """从 arXiv 的 HTML 页面里取出正文纯文本。"""
    parser = ArxivHtmlTextExtractor()
    parser.feed(html)
    parser.close()
    return parser.text()


def fetch_paper(client: httpx.Client, paper: Paper) -> tuple[str, str, str]:
    """下载一篇论文，返回 (正文文本, 版本水印, 请求的 URL)。"""
    url = f"https://arxiv.org/html/{paper.arxiv_id}"
    response = client.get(url)
    response.raise_for_status()

    text = extract_text(response.text)

    # 抽出来太短就报错，不要写出一个"看起来成功、其实几乎没内容"的文件。
    # 这种文件不会让任何东西立刻崩掉，只会让 Task 13 的评测分数莫名其妙地低，
    # 然后花几个小时去查检索逻辑——而问题其实在语料。
    if len(text) < MIN_PLAUSIBLE_CHARS:
        raise ValueError(
            f"只提取到 {len(text):,} 字符（预期 >{MIN_PLAUSIBLE_CHARS:,}）——"
            "正文边界可能失效了，请检查 arXiv 的 HTML 结构"
        )

    return text, extract_version(response.text), url


def main() -> int:
    parser = argparse.ArgumentParser(description="从 arXiv 抓取语料论文正文")
    parser.add_argument(
        "--force",
        action="store_true",
        help="重新抓取已经存在的文件（默认跳过，保证幂等）",
    )
    args = parser.parse_args()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = CORPUS_DIR / "sources.json"

    # 被跳过的文件不会重新下载，也就拿不到版本号——从上一版 manifest 里继承。
    # 不这样做的话，重跑一次脚本就会把已有的版本信息抹成空字符串。
    previous = _load_previous_entries(manifest_path)

    # 一个 client 复用连接：连着抓 5 篇不用每次重新握手。
    # 加 User-Agent 是因为 arXiv 对匿名脚本请求有限流，标明身份更礼貌也更容易被放过。
    headers = {"User-Agent": "ai-paper-qa/0.1 (learning project; corpus fetch script)"}
    sources = []
    failures = 0

    with httpx.Client(headers=headers, follow_redirects=True, timeout=60.0) as client:
        for paper in PAPERS:
            path = CORPUS_DIR / f"{paper.slug}.txt"

            if path.exists() and not args.force:
                entry = _source_entry(paper, path, previous.get(paper.slug, {}).get("version", ""))
                print(f"  跳过  {paper.slug:32} 已存在（{entry['characters']:,} 字符）")
                sources.append(entry)
                continue

            try:
                text, version, url = fetch_paper(client, paper)
            except Exception as error:  # noqa: BLE001 — 脚本要报告是哪篇失败，而不是整个崩掉
                print(f"  失败  {paper.slug:32} {type(error).__name__}: {error}")
                failures += 1
                continue

            # 写 UTF-8：正文里有大量非 ASCII 字符（希腊字母、连字符、引号变体）。
            # 用 Windows 默认编码会在写盘时直接抛 UnicodeEncodeError。
            path.write_text(text, encoding="utf-8")
            entry = _source_entry(paper, path, version)
            print(f"  完成  {paper.slug:32} {len(text):>7,} 字符  {version}")
            sources.append(entry)

    # sources.json 记录每篇的来源、版本和规模，和 .txt 一起入库。
    # 将来有人问"这个评测用的到底是哪份文本"，答案在这里。
    manifest = {
        "generated_by": "scripts/fetch_corpus.py",
        "note": (
            "正文来自 arXiv 官方 HTML 版。已剔除参考文献、作者区、页面导航、图注和表格；"
            "公式保留其 LaTeX 源码（轻度清洗过），没有渲染成数学排版。"
        ),
        "papers": sources,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n语料目录：{CORPUS_DIR}")
    if not sources:
        print("⚠️  一篇都没抓成功。")
        return 1
    if failures:
        print(f"⚠️  {failures} 篇抓取失败，语料不完整——评测结果会不可靠，请重跑。")
        return 1
    return 0


def _load_previous_entries(manifest_path: Path) -> dict[str, dict]:
    """读上一版 sources.json，按 slug 索引。文件不存在或坏掉都返回空字典。

    这里**故意不抛异常**：manifest 只是个记录，它坏掉不该阻止重新抓取语料。
    （和正文提取的逻辑相反——那边的失败必须响亮，因为产出的文本会被当数据用。）
    """
    if not manifest_path.exists():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {entry["slug"]: entry for entry in data.get("papers", [])}
    except (json.JSONDecodeError, KeyError, TypeError):
        return {}


def _source_entry(paper: Paper, path: Path, version: str) -> dict:
    return {
        "slug": paper.slug,
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "url": f"https://arxiv.org/abs/{paper.arxiv_id}",
        "version": version,
        "file": path.name,
        "characters": len(path.read_text(encoding="utf-8")),
    }


if __name__ == "__main__":
    sys.exit(main())
