"""数据库表定义。

三张表的关系：
    documents 1 ──── n chunks        一篇文档切成很多片段
    qa_logs                          独立表，记录每次问答（Task 12 才写入）
"""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import EMBEDDING_DIM
from app.core.database import Base


def utc_now() -> datetime:
    """统一用带时区的 UTC 时间。

    不要用 datetime.now()：它返回不带时区的本地时间，服务器换个时区、
    或者和数据库比对时就会出现"差 8 小时"这种经典事故。
    """
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    source: Mapped[str] = mapped_column(String(500), nullable=False)

    # 内容的 SHA-256。唯一约束保证同一篇文档不会被重复导入，
    # 同时它也是一个索引，Task 5 的"查重"能直接命中索引而不是全表扫描。
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # cascade 是 ORM 层的行为：删除 Document 对象时，SQLAlchemy 会自动
    # 连带删除它关联的 chunk 对象。它和下面外键上的 ondelete="CASCADE"
    # （数据库层的行为）是两道保险，双保险在删除操作上很常见。
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        # ondelete="CASCADE"：数据库层面，删文档时自动删掉它的所有 chunk。
        # 没有它，删除会被数据库拒绝（外键约束报错）。
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        # 外键列一定要建索引：否则"取出某文档的所有 chunk"会全表扫描
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # 向量列，维度由 EMBEDDING_DIM 决定（384 = all-MiniLM-L6-v2 的输出维度）。
    # nullable=True 是有意的：Task 5 导入时先存文本（快），
    # Task 7/8 再批量生成向量并回填，中间存在一个"还没有向量"的合法状态。
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )

    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        # 同一篇文档里，chunk 序号不能重复。把它交给数据库而不是靠代码自觉：
        # 代码可能有 bug，约束不会。
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_chunk_index"),
    )


class QALog(Base):
    """每次提问的审计记录。Task 12 开始写入。"""

    __tablename__ = "qa_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    # 检索命中的 chunk id 列表。用 PG 的数组类型直接存，省一张关联表。
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
