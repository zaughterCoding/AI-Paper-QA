"""Database tables: ``documents``, ``chunks`` and ``qa_logs``."""

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import EMBEDDING_DIM
from app.core.database import Base


def utc_now() -> datetime:
    """Timezone-aware UTC.

    Never ``datetime.now()``: its naive local time shifts with the server's timezone
    and silently disagrees with timestamps already in the database.
    """
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    source: Mapped[str] = mapped_column(String(500), nullable=False)

    # SHA-256 of the content. Unique, so the same document cannot be imported twice,
    # and indexed, so the duplicate check is an index lookup rather than a table scan.
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # ORM-level cascade: deleting a Document through the session also deletes its
    # chunks. The ondelete="CASCADE" on the foreign key below does the same inside the
    # database. Both are kept so neither path can leave orphaned rows.
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        # Deleting a document removes its chunks at the database level; without this
        # the delete would be rejected by the foreign key.
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        # Indexed, otherwise "fetch every chunk of this document" scans the table.
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # nullable=True on purpose: ingestion stores text first (fast, no model) and
    # vectors are backfilled in a later pass, so "no vector yet" is a valid state.
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(EMBEDDING_DIM), nullable=True
    )

    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        # Enforced by the database rather than by application code, which can have bugs.
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_chunk_index"),
    )


class QALog(Base):
    """One audit row per question answered."""

    __tablename__ = "qa_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    # A Postgres array rather than a join table.
    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
