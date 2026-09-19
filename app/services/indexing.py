"""索引服务：给已经入库的片段补上向量。

## 为什么这是一个独立的服务，而不是塞进 DocumentIngestionService

设计书的数据流（`learning_docs/2026-09-14-ai-paper-qa-design.md` 第 4、5 步）
把"生成 embedding"和"保存向量"画在导入流程里。但把这两件事拆开有四个实际好处：

1. **导入保持快**：导入只写文本，不加载模型、不等编码。
   模型加载要几秒，几百个片段的编码也要时间——这些都从 `POST /documents` 里挪走了。
2. **可重试**：编码失败（模型没下载完、内存不够、进程被 kill）时，
   **文档已经在库里了**，重跑索引就行，不用把文档删了重导。
3. **换模型后可重建**：模型换了要重新编码全部语料，
   这里有一个明确的入口，而不是"把文档全删了重新导一遍"。
4. **导入服务的测试不受影响**：`DocumentIngestionService` 不需要认识
   `EmbeddingClient`，Task 5 写的测试一行都不用改。

代价是：**必须有人记得触发它**。两处触发点：

- `POST /documents` 路由在 `ingest()` 之后显式调用（正常路径）
- `scripts/index_pending.py`（手动/补漏路径）

**和 Task 5 的分工**：`ingest` 负责"文本进库"，`index` 负责"文本变成可检索的"。
两件事的事务也是分开的——这正是 `chunks.embedding` 允许 NULL 的意义（F-22）。
"""

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.rag.embeddings import EmbeddingClient
from app.repositories.chunks import ChunkRepository
from app.repositories.documents import DocumentRepository


@dataclass(frozen=True)
class IndexingResult:
    """一次索引（回填向量）的结果。

    为什么要同时给两个数字，而不是只给 `embedded_count`？

    因为**"这次嵌入了 0 个"有两种完全不同的含义**：
    - `embedded_count=0, skipped_count=0` → 这篇文档一个片段都没有（异常）
    - `embedded_count=0, skipped_count=N` → N 个片段早就有向量了（正常的重复索引）

    只返回一个数字的话，这两种情况长得一模一样。这是 F-39 那个教训的另一面：
    **让"什么都没发生"和"做完了"可区分。**
    """

    embedded_count: int
    skipped_count: int


class IndexingService:
    def __init__(self, session: Session, embedding_client: EmbeddingClient) -> None:
        self.session = session
        self.embedding_client = embedding_client
        self.chunks = ChunkRepository(session)
        self.documents = DocumentRepository(session)

    def index_document(self, document_id: uuid.UUID) -> IndexingResult:
        """给一篇文档里所有还没有向量的片段补上向量。

        重复调用是安全的（幂等）：已经有过向量的片段会被跳过，
        不会被重新编码、也不会被覆盖。

        全有或全无：所有向量写完之后才 commit。中途任何一步失败，
        整个事务回滚，不会留下"一半片段有向量、一半没有"的状态。
        """
        # 文档不存在时直接抛错，而不是"找不到 → 返回 0 → 看起来成功了"。
        # 调用方传的 id 来自某个上游，对不上说明上游出了问题；
        # 静默返回 0 会让这个错误一直藏着，直到某天发现检索结果莫名其妙地少。
        # （同样的原则见 ChunkRepository.update_embedding。）
        if self.documents.get_by_id(document_id) is None:
            raise ValueError(f"document not found: {document_id}")

        pending = self.chunks.list_without_embedding(document_id=document_id)

        if not pending:
            # 没有待办就别调模型——空列表进 encode 是浪费，某些版本还会直接抛异常。
            return IndexingResult(
                embedded_count=0,
                skipped_count=self.chunks.count_by_document(document_id),
            )

        texts = [chunk.text for chunk in pending]
        vectors = self.embedding_client.embed_texts(texts)

        # ⚠️ 这个检查不是"防御性代码"，它挡的是一个会静默发生的 bug。
        # 下面用的是 zip()，而 zip 在两边长度不等时会**悄悄地在短的那边停下**——
        # 如果客户端因为任何原因少返回了一个向量，就会有片段永远拿不到向量，
        # 而且不会有任何报错。宁可在这里炸掉。
        if len(vectors) != len(texts):
            raise ValueError(
                f"embedding client returned {len(vectors)} vectors for {len(texts)} texts"
            )

        for chunk, vector in zip(pending, vectors, strict=True):
            self.chunks.update_embedding(chunk.id, vector)

        # 事务边界在 service 层（F-18）：所有向量都写好了才提交。
        self.session.commit()

        return IndexingResult(
            embedded_count=len(pending),
            skipped_count=self.chunks.count_by_document(document_id) - len(pending),
        )

    def index_all_pending(self) -> IndexingResult:
        """给全库所有还没有向量的片段补上向量。

        这是 `scripts/index_pending.py` 的入口。和 `index_document` 的区别是
        它一次处理多篇文档，所以 commit 也只在最后做一次——
        要么全部重建，要么一篇都不动。

        不按文档循环调用 `index_document`，是因为那样每篇文档都要 commit 一次，
        中途失败会留下"前几篇好了、后面的没动"的半成品状态，
        下次还得自己判断从哪继续。
        """
        pending = self.chunks.list_without_embedding()

        if not pending:
            return IndexingResult(embedded_count=0, skipped_count=self.chunks.count_all())

        texts = [chunk.text for chunk in pending]
        vectors = self.embedding_client.embed_texts(texts)

        if len(vectors) != len(texts):
            raise ValueError(
                f"embedding client returned {len(vectors)} vectors for {len(texts)} texts"
            )

        for chunk, vector in zip(pending, vectors, strict=True):
            self.chunks.update_embedding(chunk.id, vector)

        self.session.commit()

        return IndexingResult(
            embedded_count=len(pending),
            skipped_count=self.chunks.count_all() - len(pending),
        )
