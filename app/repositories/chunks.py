"""chunks 表的读写。

这里承担两类职责，界线很清楚：
- **写**：插入片段、回填向量（导入文档时用）
- **查**：按向量相似度找出最相关的片段（提问时用）
"""

import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.tables import Chunk, Document
from app.rag.chunking import TextChunk


@dataclass(frozen=True)
class RetrievedChunk:
    """一次向量检索命中的片段。

    为什么不直接返回 Chunk 这个 ORM 对象？

    1. **装不下**：检索结果要带上所属文档的标题（title），它不在 chunks 表里，
       是 JOIN 出来的。ORM 对象没有这个字段。
    2. **不可变**：frozen dataclass 是只读的，调用方不会意外改到"数据库里的数据"
       （其实只是内存里的快照，但改它毫无意义、只会造成困惑）。
    3. **脱离 session**：上层拿到它之后不需要数据库连接还开着。
       ORM 对象一旦 session 关闭，再访问延迟加载的字段就炸（DetachedInstanceError）。

    字段含义：
        score: 余弦相似度，**越大越相似**，范围 [-1, 1]。
               因为入库的向量都做过归一化，这里的 score 等价于两个向量的点积。
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    title: str
    chunk_index: int
    text: str
    score: float


class ChunkRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_chunks(self, document_id: uuid.UUID, chunks: list[TextChunk]) -> list[Chunk]:
        """批量插入一个文档的所有片段。

        为什么用 add_all 一次插完，而不是循环里逐条 add？SQLAlchemy 会把
        add_all 的对象攒成一条批量 INSERT（executemany），比 N 次单条插入快得多。
        一个文档有几十个片段，这里的差别是几十倍。

        注意：这里**不写 embedding**。导入时先只存文本（快、不依赖模型），
        向量稍后由 Task 7 的 EmbeddingClient 批量生成再回填——
        这就是 embedding 列允许 NULL 的原因。
        """
        rows = [
            Chunk(
                document_id=document_id,
                chunk_index=chunk.index,
                text=chunk.text,
                token_count=chunk.token_count,
            )
            for chunk in chunks
        ]
        self.session.add_all(rows)
        # flush 让数据库立刻检查约束。如果切分器给出了重复的 index，
        # 唯一约束会在这里报错——错误越早暴露越好，而不是等到 commit。
        self.session.flush()
        return rows

    def count_by_document(self, document_id: uuid.UUID) -> int:
        """统计一个文档有多少片段。

        用 SQL 的 COUNT(*) 而不是 len(document.chunks)：后者会把所有片段的
        完整文本（可能几十 KB）加载进内存，只为了数个数。数据量大时这是
        典型的性能陷阱——**要计数就用 COUNT，不要把行拉回来自己数**。
        """
        count = self.session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
        )
        return count or 0

    def count_all(self) -> int:
        """统计全库片段总数。用来算"有多少片段已经有向量了"。

        和 count_by_document 一样，用 SQL 的 COUNT(*) 而不是把行拉回来自己数。
        """
        count = self.session.scalar(select(func.count()).select_from(Chunk))
        return count or 0

    def list_without_embedding(self, document_id: uuid.UUID | None = None) -> list[Chunk]:
        """找出还没有向量的片段——回填流程的输入。

        `document_id` 传了就只找这一篇的，不传就找全库的。

        为什么按 `(document_id, chunk_index)` 排序？
        **为了让结果确定。** 不写 ORDER BY 时，PostgreSQL 返回行的顺序是
        未定义的（取决于物理存储、并行扫描、甚至缓存命中情况），同一个查询
        两次跑可能给出不同顺序。对回填来说顺序不影响正确性，但对**测试**影响很大：
        断言"第一个被写入向量的是哪条"会随机失败。
        （同一个教训见 F-11：`ORDER BY created_at DESC` 不稳定会导致分页丢记录。）
        """
        statement = select(Chunk).where(Chunk.embedding.is_(None))
        if document_id is not None:
            statement = statement.where(Chunk.document_id == document_id)

        return list(self.session.scalars(statement.order_by(Chunk.document_id, Chunk.chunk_index)))

    def update_embedding(self, chunk_id: uuid.UUID, embedding: list[float]) -> None:
        """把一个片段的向量写回数据库。

        找不到 chunk 时抛 ValueError 而不是静默返回：这个方法的调用方是
        回填流程，如果 id 对不上说明上游出了问题，静默跳过会让
        "有些片段永远没有向量"这种 bug 一直藏着，直到检索结果莫名其妙地差。
        """
        chunk = self.session.get(Chunk, chunk_id)
        if chunk is None:
            raise ValueError(f"chunk not found: {chunk_id}")

        chunk.embedding = embedding
        # flush 而不是 commit——事务边界属于 service 层，repository 不决定
        # "这次写入要不要和别的写入一起成功"。
        self.session.flush()

    def search_similar(
        self, query_embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        """找出与查询向量最相似的 top_k 个片段。

        生成的 SQL 大致是：

            SELECT chunks.id, chunks.document_id, documents.title, ...,
                   1 - (chunks.embedding <=> :query) AS score
            FROM chunks JOIN documents ON documents.id = chunks.document_id
            WHERE chunks.embedding IS NOT NULL
            ORDER BY chunks.embedding <=> :query
            LIMIT :top_k

        三个要点：

        1. **`<=>` 是 pgvector 的余弦距离算子**，越小越相似。所以 ORDER BY 用
           升序（默认），而对外返回的 score 用 `1 - 距离` 换算成"越大越相似"——
           距离是给数据库排序用的，相似度是给人看的。
        2. **`WHERE embedding IS NOT NULL` 不是可有可无的**。导入文档时向量还没生成，
           库里合法地存在大量 embedding 为 NULL 的片段。不排除它们，
           这些片段会以 NULL 距离参与排序（PostgreSQL 里 NULL 排最后，
           但语义上它们根本不该出现在结果里）。
        3. **必须要 JOIN documents** 才能拿到标题——回答问题时需要告诉用户
           "这段话出自哪篇文档"，光有 chunk 文本没法给出处。
        """
        if top_k < 1:
            # 数据库对 LIMIT -1 是直接报错的。在这里拦下来，报错信息才有意义。
            raise ValueError("top_k must be >= 1")

        distance = Chunk.embedding.cosine_distance(query_embedding)

        statement = (
            select(
                Chunk.id.label("chunk_id"),
                Chunk.document_id.label("document_id"),
                Document.title.label("title"),
                Chunk.chunk_index.label("chunk_index"),
                Chunk.text.label("text"),
                # 距离 → 相似度。两个向量都归一化过，所以这里就是余弦相似度。
                (1 - distance).label("score"),
            )
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.embedding.is_not(None))
            .order_by(distance)
            .limit(top_k)
        )

        return [
            RetrievedChunk(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                title=row.title,
                chunk_index=row.chunk_index,
                text=row.text,
                score=row.score,
            )
            for row in self.session.execute(statement)
        ]
