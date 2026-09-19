"""语料抓取脚本的测试。

## 为什么这个脚本值得有测试

它是个 **scripts/ 下的一次性脚本**，按项目里其他脚本的惯例（db.py、
index_pending.py）本来是不测的。但它和它们有个本质区别：

**它的输出会变成数据。**

`index_pending.py` 跑错了，重跑一次就行；`fetch_corpus.py` 跑错了，
产出的是一份"看起来正常、实际缺了半篇"或"混进了一堆导航文字"的语料文件，
然后这份文件会被导入向量库，成为 Task 13 评测集的输入。
到那时再去查"为什么检索命中率这么低"，问题已经离源头很远了。

这和 F-39 是同一类：**错误不产生任何信号**，只是让下游的数字莫名其妙地差。

所以这里测的不是"脚本能跑通"，而是**几条具体的正确性契约**：
哪些内容必须被剔除、哪些必须被保留、结构变了的时候会不会响亮地失败。

## 关于直接测下划线开头的私有函数

`_latex_to_text` 和 `_normalize_whitespace` 是模块私有。直接测它们是有意的：
它们的边界情况很多（嵌套、正则边界断言），如果全部通过 `extract_text`
间接测，就得为每种情况手工构造一段 HTML，测试会变得又长又脆。
而它们本身是纯函数、契约清楚，单独测更直接。
"""

import json
import re
from pathlib import Path

import pytest

from scripts.fetch_corpus import (
    CORPUS_DIR,
    PAPERS,
    _latex_to_text,
    _normalize_whitespace,
    extract_text,
    extract_version,
    fetch_paper,
)


_DEFAULT_AUTHORS = (
    '<span class="ltx_role_author">A. Author</span>'
    '<span class="ltx_note">thanks: equal contribution, listing order is random</span>'
)


def _page(*body_parts: str, authors_html: str = _DEFAULT_AUTHORS) -> str:
    """造一个结构上和 arXiv 页面一样的 HTML。

    刻意保留了真实的 class 名和嵌套关系——测试的价值就在于
    这些标记被改动时能红，用简化过的假 HTML 就测不到东西了。
    """
    return (
        "<html><body>"
        # 正文之前的页面噪声，全都必须被挡住
        '<div id="infobox">'
        '<a href="https://info.arxiv.org/help/license">License: arXiv.org perpetual</a>'
        '<div id="watermark-tr">arXiv:1234.56789v2 [cs.CL] 01 Jan 2024</div>'
        "</div>"
        # 注意 article 的 class 里含 "ltx_authors_1line" —— 它**不是**作者区，
        # 按子串匹配"ltx_authors"的话会把整篇文章都丢掉
        '<div class="ltx_page_content">'
        '<article class="ltx_document ltx_authors_1line">'
        '<p class="ltx_p">Provided proper attribution is provided.</p>'
        '<h1 class="ltx_title ltx_title_document">A Paper</h1>'
        f'<div class="ltx_authors">{authors_html}</div>' + "".join(body_parts) + '<section id="bib" class="ltx_bibliography">'
        "<h2>References</h2>"
        "<ul><li>A cited paper that must not leak into the corpus. 2020.</li></ul>"
        "</section></article></div></body></html>"
    )


# --- 正文的边界 -------------------------------------------------------------------


def test_keeps_the_body() -> None:
    text = extract_text(_page('<p class="ltx_p">The Transformer is a model.</p>'))

    assert "The Transformer is a model." in text


def test_ignores_page_chrome_before_the_title() -> None:
    """许可声明、版本水印这些页面装饰不能进语料。

    它们混进去不会有任何报错，只会让每个片段都带上一段无关文字。
    """
    text = extract_text(_page('<p class="ltx_p">Body.</p>'))

    assert "License" not in text
    assert "infobox" not in text


def test_ignores_the_author_block() -> None:
    """作者区必须剔除——那里混着大量脚注（"thanks: equal contribution…"），
    是作者贡献声明，不是论文内容。"""
    text = extract_text(_page('<p class="ltx_p">Body.</p>'))

    assert "A. Author" not in text
    assert "equal contribution" not in text


def test_ignores_the_author_block_when_it_contains_nested_divs() -> None:
    """作者区内部有同名的 `<div>` 时，仍然要整块剔除。

    这条是变异测试逼出来的。最初的实现判断"这个 `</div>` 是不是在关闭
    被丢弃的元素"，用的是"标签名和栈顶对不对得上"——一旦被丢弃的元素内部
    还有同名的标签，内层的结束标签就会把外层的丢弃状态一起清掉，
    于是作者名、贡献声明这些噪声漏进语料。**不报错，只是语料脏了。**

    真实的 arXiv 页面上作者区用的是 `<span>`，所以没触发；
    但这属于运气，不是正确性。
    """
    nested = '<div class="ltx_affiliation">Dept.</div><span>A. Author</span>'
    html = _page('<p class="ltx_p">Body.</p>', authors_html=nested)

    text = extract_text(html)

    assert "A. Author" not in text
    assert "Dept." not in text
    assert "Body." in text


def test_does_not_confuse_ltx_authors_1line_with_the_author_block() -> None:
    """回归测试：`ltx_authors_1line` 是**布局标记**，不是作者区。

    它出现在 `<article>` 的 class 里，而 `<article>` 包着**整篇文章**。
    如果按子串匹配 "ltx_authors" 来判定作者区，就会把整篇内容全部丢掉——
    输出变成空字符串，而且不会报错。
    """
    text = extract_text(_page('<p class="ltx_p">This must survive.</p>'))

    assert "This must survive." in text


def test_ignores_the_bibliography() -> None:
    """参考文献整段剔除。

    它篇幅大、几乎不可能是答案，却因为包含全篇所有术语而非常容易"蹭"到高分。
    """
    text = extract_text(_page('<p class="ltx_p">Body.</p>'))

    assert "cited paper" not in text
    assert "References" not in text


def test_raises_when_the_body_marker_is_missing() -> None:
    """找不到正文开始标记时必须报错，不能静默降级。

    降级的后果是：从页面最顶上开始收，把许可声明、导航文字当成论文内容
    写进语料——同样没有任何信号。arXiv 改版时这个测试会红，
    那正是应该有人去看一眼的时候。
    """
    html = (
        "<html><body><p>No title marker here.</p>"
        '<section id="bib"><ul><li>ref</li></ul></section></body></html>'
    )

    with pytest.raises(ValueError, match="ltx_title_document"):
        extract_text(html)


# --- 公式 -------------------------------------------------------------------------


def test_keeps_math_alttext() -> None:
    """公式的 alttext 要保留，MathML 碎片要丢掉。

    整段丢掉会把句子弄断（"of size in the input sequence"），
    留下 MathML 又会塞进一堆 `mi`/`mo` 字形碎片。
    """
    html = _page(
        '<p class="ltx_p">a neighborhood of size '
        '<math alttext="r"><semantics><mi>r</mi><annotation>x</annotation></semantics></math>'
        " in the input</p>"
    )

    text = extract_text(html)

    assert "of size r in the input" in text
    assert "semantics" not in text
    assert "annotation" not in text


def test_drops_math_that_sits_inside_a_dropped_element() -> None:
    """被丢弃元素**内部**的公式也不该被捞回来。

    公式分支写在"是否已进入丢弃状态"的判断之后，这个测试钉住那个顺序。
    """
    html = _page(
        '<figure><math alttext="SHOULD_NOT_APPEAR"><mi>x</mi></math></figure>'
        '<p class="ltx_p">Body.</p>'
    )

    text = extract_text(html)

    assert "SHOULD_NOT_APPEAR" not in text
    assert "Body." in text


def test_drops_nested_dropped_elements() -> None:
    """丢弃是**整棵子树**，不是一层。

    `<figure><svg><path/></svg></figure>` 这种嵌套很常见。
    如果用一个布尔量而不是栈来记状态，内层闭合时就会把外层也"关掉"，
    后面整篇内容都会被当成在丢弃元素内部而消失。
    """
    html = _page(
        '<figure><svg><path d="M0 0"></path></svg>figure caption</figure>'
        '<p class="ltx_p">Body survives.</p>'
    )

    text = extract_text(html)

    assert "figure caption" not in text
    assert "Body survives." in text


def test_recovers_from_malformed_tag_nesting() -> None:
    """标签嵌套错乱时，不能把后面的正文一起丢掉。

    这是条**防御性**的测试——现实中的 arXiv 页面是机器生成的、不会错乱，
    所以它挡的不是"现在会发生的 bug"，而是"改了代码之后会发生的事"：
    `handle_endtag` 里有一段容错逻辑（栈顶对不上时，往下找匹配的那一层一起弹掉）。
    没有它的话，一个错位的 `</figure>` 会让丢弃栈永远清不干净，
    后面整篇文章都收不到——而且输出只是"变短了"，没有别的信号。
    """
    html = _page('<figure><svg></figure></svg><p class="ltx_p">Body survives.</p>')

    text = extract_text(html)

    assert "Body survives." in text


# --- 正则边界（曾经写错过的地方）--------------------------------------------------


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        # 排版包装：去掉命令，留下内容
        (r"\mathrm{softmax}", "softmax"),
        (r"\textsc{BASE}", "BASE"),
        (r"\mathbf{\mathrm{x}}", "x"),  # 嵌套一层
        # 符号表
        (r"\theta", "θ"),
        (r"\cdot", "·"),
        (r"O(n/r)", "O(n/r)"),  # 没有命令的公式原样保留
        # 命名冲突：必须靠"长的排前面"来消歧
        (r"\inR^{H}", "R^{H}"),  # 不能变成 "∈R^{H}"
        (r"x \in R", "x ∈ R"),
        # 括号与空白
        (r"\left(x\right)", "(x)"),
        (r"a \, b", "a b"),
        # 转义字符
        (r"100\%", "100%"),
    ],
)
def test_latex_to_text(latex: str, expected: str) -> None:
    assert _latex_to_text(latex) == expected


def test_symbol_matching_survives_a_subscript() -> None:
    """`\\beta_{1}` 里的 `\\beta` 必须被替换掉。

    这条是回归测试。最初的正则用 `\\b` 收尾，而下划线在正则里**也算单词字符**，
    于是 `\\beta` 和 `_` 之间没有边界、命令就漏掉了——
    带下标的公式恰恰是论文里最常见的形式，漏掉一大片却不报错。
    """
    assert _latex_to_text(r"\beta_{1}") == "β_{1}"
    assert _latex_to_text(r"\eta") == "η"


def test_symbol_matching_does_not_eat_the_start_of_a_longer_command() -> None:
    """反向的边界：`\\thetaX` 不是一个符号后跟字母 X，而是个未知命令，别动它。"""
    assert _latex_to_text(r"\thetaX") == r"\thetaX"


def test_nested_frac_is_left_alone() -> None:
    """清洗边界：嵌套的 `\\frac` 原样保留。

    不是处理不了，是**半吊子清洗比不清洗更糟**——如果一层能换、两层不能换，
    语料里就有两种写法，出问题时没法用一条规则去搜。
    这条测试把这个决定钉住，免得将来有人"顺手"加一条只处理单层的规则。
    """
    nested = r"\frac{1}{\sqrt{d_{k}}}"

    assert _latex_to_text(nested) == nested


# --- 空白归一化 -------------------------------------------------------------------


def test_normalize_collapses_whitespace() -> None:
    assert _normalize_whitespace("a  \n\n\n  b\t\tc") == "a\n\nb c"


def test_normalize_removes_space_before_punctuation() -> None:
    """公式或引用标记被拿掉后会在标点前留下空格。

    原文 "dilated convolutions [19], increasing" 去掉 `[19]` 就成了
    "convolutions , increasing"。
    """
    assert _normalize_whitespace("convolutions , increasing") == "convolutions, increasing"


def test_normalize_keeps_space_before_an_opening_bracket() -> None:
    """只清理**后置**标点。"word (" 里左括号前面有空格是对的，不能一起清掉。"""
    assert _normalize_whitespace("a function (see below)") == "a function (see below)"


def test_normalize_handles_non_breaking_space() -> None:
    """`\\xa0`（不换行空格）在 arXiv 的 HTML 里很常见。

    它看起来和普通空格一模一样，但会混进片段文本、影响 token 计数。
    """
    assert _normalize_whitespace("a\xa0b") == "a b"


# --- 版本 -------------------------------------------------------------------------


def test_extract_version() -> None:
    assert extract_version(_page("<p>x</p>")) == "arXiv:1234.56789v2 [cs.CL] 01 Jan 2024"


def test_extract_version_returns_empty_when_absent() -> None:
    """取不到版本号不算错误——manifest 只是记录，不该让抓取失败。"""
    assert extract_version("<html><body>no watermark</body></html>") == ""


# --- 抓取时的健全性检查 -----------------------------------------------------------
#
# `fetch_paper` 需要网络，所以用假的 client 来测——它在这里的价值不是"能下载"，
# 而是**下载完之后的那道检查**：正文短得离谱时宁可报错，也不要写出一个
# "看起来成功、其实几乎没内容"的文件（这种文件不会让任何东西崩，
# 只会让 Task 13 的评测分数莫名其妙地低，然后有人花几小时去查检索逻辑）。


class _FakeClient:
    """只实现 fetch_paper 用到的那两个方法。"""

    def __init__(self, html: str) -> None:
        self._html = html

    def get(self, url: str) -> "_FakeResponse":
        return _FakeResponse(self._html)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        pass


def test_fetch_paper_rejects_a_suspiciously_short_extraction() -> None:
    short_page = _page('<p class="ltx_p">正文只有这么一点点。</p>')

    with pytest.raises(ValueError, match="字符"):
        fetch_paper(_FakeClient(short_page), PAPERS[0])


def test_fetch_paper_accepts_a_full_length_extraction() -> None:
    """反向确认：正常长度的正文不该被这道检查误伤。

    只测"短的要报错"是不够的——把阈值改成无穷大也能让那条测试通过，
    但脚本就再也抓不到任何东西了。
    """
    long_page = _page(f'<p class="ltx_p">{"word " * 4000}</p>')

    text, version, url = fetch_paper(_FakeClient(long_page), PAPERS[0])

    assert len(text) > 10_000
    assert version == "arXiv:1234.56789v2 [cs.CL] 01 Jan 2024"
    assert url == "https://arxiv.org/html/1706.03762"


# --- 真实语料 ---------------------------------------------------------------------
#
# 下面这组直接读 eval/corpus/ 里已经抓好的文件（它们随仓库入库）。
# 它们**不需要网络**，跑的是"现在入库的这份数据本身干不干净"。
#
# 这一组能挡住的是：将来有人重跑脚本、或者调整了提取规则之后，
# 产出的语料里混进了 HTML 残留——那种问题看文件是看不出来的，
# 只有被导入向量库、检索结果变差之后才会有人注意到。


def _corpus_files() -> list[Path]:
    return sorted(p for p in CORPUS_DIR.glob("*.txt"))


def test_corpus_directory_is_not_empty() -> None:
    """语料为空时后面的测试会全部"通过"（因为没有文件可遍历）——
    这条先把这个假绿堵掉。"""
    assert _corpus_files(), f"{CORPUS_DIR} 里没有语料文件，先跑 scripts/fetch_corpus.py"


# 正文里出现就说明剥标签没剥干净的特征。
#
# 这里**不能**简单地找 `<` 和 `>` 就算数——论文正文里它们是正常的数学符号
# （`k<n`、`s_{i,j}>s_null`），而且"看着像标签"也没法作为判据：
# `k<n` 里的 `<n` 形状上完全就是 `<` 加一个标签名。
# 试过用 `</?[a-zA-Z]` 判断，结果在 `kernel width k<n does not` 上误报了。
#
# 所以只挑那些**在数学表达里不可能出现**的：闭合标签的 `</`、
# 带尖括号的已知标签名、属性写法，以及转义实体。
_HTML_LEFTOVERS = ("</", "<p>", "<p ", "<div", "<span", "<table", "<tr", "<td",
                   "ltx_", "href=", "xmlns", "class=", "&amp;", "&#")


@pytest.mark.parametrize("path", _corpus_files(), ids=lambda p: p.stem)
def test_corpus_has_no_html_leftovers(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    for marker in _HTML_LEFTOVERS:
        assert marker not in text, f"{path.name} 里残留了 {marker!r}"


@pytest.mark.parametrize("path", _corpus_files(), ids=lambda p: p.stem)
def test_corpus_is_plausibly_sized(path: Path) -> None:
    """每篇论文的正文都该在几万字符量级。

    太小说明正文边界失效、只收到了一个零头；太大说明把参考文献
    或者别的什么东西一起收进来了。
    """
    text = path.read_text(encoding="utf-8")

    assert 10_000 < len(text) < 200_000, f"{path.name} 有 {len(text):,} 字符，不合常理"


@pytest.mark.parametrize("path", _corpus_files(), ids=lambda p: p.stem)
def test_corpus_has_no_mathml_fragments(path: Path) -> None:
    """MathML 展开后是 `mi`/`mo`/`mn` 这类标签名。

    它们在正文里出现说明公式的 alttext 没被用上、反而把 MathML 的
    结构碎片收了进来——那正是清洗要避免的。
    """
    text = path.read_text(encoding="utf-8")

    for fragment in ("<mi", "<mo", "<mn", "semantics"):
        assert fragment not in text, f"{path.name} 里有 MathML 碎片 {fragment!r}"


def test_sources_manifest_matches_the_files() -> None:
    """manifest 必须和实际文件对得上——它是"这份语料是哪一版"的唯一记录。"""
    manifest = json.loads((CORPUS_DIR / "sources.json").read_text(encoding="utf-8"))

    recorded = {entry["file"] for entry in manifest["papers"]}
    assert recorded == {path.name for path in _corpus_files()}

    for entry in manifest["papers"]:
        assert entry["version"], f"{entry['slug']} 没记版本号"
        assert entry["characters"] > 10_000


def test_papers_declared_in_the_script_are_all_present() -> None:
    """脚本里声明的论文都要有对应的语料文件。

    少一篇的话评测集会少一块语料，而 Task 13 写的问题可能正好指向它。
    """
    on_disk = {path.stem for path in _corpus_files()}
    declared = {paper.slug for paper in PAPERS}

    assert declared <= on_disk, f"缺语料：{declared - on_disk}"
