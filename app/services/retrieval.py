"""检索服务：把一个问题变成最相关的几个片段。

这是 RAG 里 **"R"**（Retrieval）那一步，也是整个系统里唯一把两半接起来的地方：

    问题（文本）──EmbeddingClient──▶ 查询向量 ──ChunkRepository──▶ 最相似的片段

前面几步都是单向的：Task 5/8b 把**文档**灌进库，Task 8 让库能按**向量**查。
到这里才第一次出现"用户问一句话，系统去库里找"这个动作。

## 为什么这个服务这么薄

`retrieve()` 只有四行实质代码，看起来不值得单独一个类。但它薄是有意的——
它**只做编排，不做实现**：

- 文本 → 向量：`EmbeddingClient`（Task 7）
- 向量 → 片段：`ChunkRepository.search_similar`（Task 8）

真正的复杂度在这两步里面。这一层负责的是**校验**和**顺序**：
什么输入算合法、先校验还是先编码、编码调几次、查库调几次。

## 向量维度的一致性由谁保证

`EmbeddingClient` 输出 384 维（`EMBEDDING_DIM`），`chunks.embedding` 列也是
`vector(384)`。两边**必须一直是同一个数**——不一致时 pgvector 会在执行 SQL 时
直接报错（"expected 384 dimensions, not 768"），不会静默返回错误结果。
这是这个项目里最硬的一条契约（F-21），也是换模型时最容易踩的坑。

## 为什么这里**还没有**相似度阈值

一个自然的想法是：相似度低于某个值就认为"库里没有相关内容"，返回空。
现在的 `retrieve()` 不做这件事——问一个语料之外的问题，它照样返回 top_k 个片段。

不是没想到，是**数据还不够定这个值**。真模型 + 真语料的实测（2026-09-19，
5 篇论文 / 149 片段）：

| 问题 | top-1 分数 |
|---|---|
| How does multi-head attention work? | 0.6457 |
| How are negative passages sampled…? | 0.6229 |
| What is the masked language model objective? | 0.5736 |
| How does the model combine retrieved passages…? | 0.5169 |
| bi-encoder 和 cross-encoder 的区别？ | 0.3992 |
| **What is the capital of France?（语料之外）** | **0.0517** |

分离度看起来很好——语料内最低 0.3992，语料外 0.0517，差了近一个数量级，
阈值取在 0.2~0.3 之间似乎都能分开。但这是 **6 个问题**得出的结论。
Task 13/14 会有 20 个带标注的问题、能算出完整的分数分布，
那时才该决定阈值——现在定一个数，只是把"拍脑袋"写进代码而已。

在那之前，**"找不到相关内容"这件事由 Task 10 的 LLM 根据拿到的片段自己判断**：
片段和问题无关时，正确的回答是"文档里没有提到"，而不是硬编一个答案。
"""

from sqlalchemy.orm import Session

from app.rag.embeddings import EmbeddingClient
from app.repositories.chunks import ChunkRepository, RetrievedChunk

# 不传 top_k 时取几个。
#
# 5 是个经验起点，不是算出来的：太少则答案的出处可能落在第 6、7 个片段上，
# 太多则把无关文本一起塞进 Task 10 的提示词里，既费 token 又稀释重点。
# 真实取值要靠 Task 13/14 的评测集来定，这里先给一个能用的默认值。
DEFAULT_TOP_K = 5

# top_k 的上限。超过它直接报错，而不是"照做但截断到 20"。
#
# 为什么要设上限：top_k 不是一个纯粹的"查多少条"参数，它决定了
# **后面要喂给 LLM 多少上下文**。任务 10 会把这些片段拼进提示词，
# 而模型的上下文窗口和 API 费用都随 token 数增长。给一个 1000 的 top_k，
# 结果不是"检索得更全"，而是提示词超长、答案被无关内容淹没、账单变高。
#
# 上限设 20 而不是 10：给调用方留出调试空间（排查"答案到底在第几条"时
# 会临时调大），但又不足以造成真正的伤害。
MAX_TOP_K = 20


class RetrievalService:
    """把问题检索成一组片段。

    构造时要传 `session` 和 `embedding_client`，而不是在内部自己创建：

    **`embedding_client` 必须由调用方传入**，尤其是走 Web 请求时——
    调用方要传的是 `app.rag.embeddings.get_embedding_client()` 返回的那个
    **进程级单例**（F-46），不是 `EmbeddingClient(...)` 新建一个。
    新建一个意味着再加载一份几百 MB 的模型，并发几个请求就会把内存吃光。

    把这个选择留给调用方，是因为**服务层不该知道"全局只有一个模型"这件事**——
    那是应用组装（`api/routes.py` 的依赖注入）的职责。它只要一个能
    `embed_text()` 的对象就够了，测试里传假的也一样工作。
    """

    def __init__(self, session: Session, embedding_client: EmbeddingClient) -> None:
        self.chunks = ChunkRepository(session)
        self.embedding_client = embedding_client

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        """检索与 `question` 最相关的 `top_k` 个片段，按相似度从高到低。

        返回空列表是完全正常的结果（库是空的、或者所有片段都还没有向量），
        不抛异常——"没找到"和"出错了"是两回事。

        **这个方法不写数据库**，所以不做任何 commit。它只读：
        编码问题、查库、返回。（对比 `IndexingService` 是要写库的，
        所以那边由 service 决定事务边界。）
        """
        # 校验放在编码**之前**：一个空问题没有任何语义，编码它纯属浪费——
        # 而且某些模型对空字符串返回全零向量，而全零向量和任何向量的余弦距离
        # 都是未定义的（pgvector 会按"最不相似"处理），结果是一堆看似随机的片段。
        #
        # 用 strip() 而不是直接判 `not question`：只输入空格和换行的"空问题"
        # 同样没有语义，但字符串本身非空，不 strip 就漏过去了。
        if not question.strip():
            raise ValueError("question must not be empty")

        if top_k < 1 or top_k > MAX_TOP_K:
            # 报错信息里带上实际收到的值。上层（API）会把它转成 400，
            # 调用方一眼能看出是参数问题，而不是去查检索逻辑。
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}")

        # 这里传的是**原始的 question**，不是 strip() 之后的版本。
        # 保留原样是为了让"送进模型的东西"和"调用方给的东西"完全一致——
        # 出问题时可以直接拿同样的字符串复现，不用猜中间被改过什么。
        # （首尾空白对 embedding 模型没有影响，它按 token 处理。）
        query_embedding = self.embedding_client.embed_text(question)

        return self.chunks.search_similar(query_embedding=query_embedding, top_k=top_k)
