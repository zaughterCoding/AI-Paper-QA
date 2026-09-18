"""chunks 表的读写。"""

import uuid

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.tables import Chunk
from app.rag.chunking import TextChunk


class ChunkRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_chunks(self, document_id: uuid.UUID, chunks: list[TextChunk]) -> list[Chunk]:
        """批量插入一个文档的所有片段。

        为什么用 add_all 一次插完，而不是循环里逐条 add？SQLAlchemy 会把
        add_all 的对象攒成一条批量 INSERT（executemany），比 N 次单条插入快得多。
        一个文档有几十个片段，这里的差别是几十倍。
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
