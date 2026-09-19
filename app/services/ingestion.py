"""Ingestion service: turn raw text into a document plus its chunks."""

import hashlib
import uuid
from dataclasses import dataclass

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.rag.chunking import TextChunk, TextChunker
from app.repositories.chunks import ChunkRepository
from app.repositories.documents import DocumentRepository


@dataclass(frozen=True)
class IngestionResult:
    """A database-independent summary, so the API layer never handles ORM objects."""

    document_id: uuid.UUID
    chunk_count: int
    created: bool  # True when newly created, False when existing content was reused


class DocumentIngestionService:
    def __init__(self, session: Session, chunker: TextChunker | None = None) -> None:
        self.session = session
        # The chunker is injectable so a test can supply one returning fixed chunks and
        # exercise the orchestration without depending on the real splitting algorithm.
        self.chunker = chunker or TextChunker()
        self.documents = DocumentRepository(session)
        self.chunks = ChunkRepository(session)

    def ingest(self, title: str, source: str, content: str) -> IngestionResult:
        """Import one document. Re-importing identical content adds nothing."""
        self._validate(title=title, content=content)

        # A fingerprint of the content itself: the same text always hashes to the same
        # value, which makes it usable as the "already imported?" test.
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        existing = self.documents.get_by_content_hash(content_hash)
        if existing is not None:
            # Idempotent: a repeat import returns the original document rather than
            # failing or storing a copy. The chunk count is read back from the database so
            # the caller receives the same information as on the first import.
            return IngestionResult(
                document_id=existing.id,
                chunk_count=self.chunks.count_by_document(existing.id),
                created=False,
            )

        try:
            document = self.documents.create_document(
                title=title, source=source, content_hash=content_hash
            )
            # The document is created first so its chunks have a document_id to attach to.
            # Chunking is pure computation and touches no database state.
            text_chunks: list[TextChunk] = self.chunker.chunk(content)
            saved = self.chunks.create_chunks(document_id=document.id, chunks=text_chunks)

            # Committed only once every chunk is prepared, so a failure cannot leave a
            # document behind with no chunks.
            self.session.commit()
        except IntegrityError:
            # Race fallback: the check-then-insert above has a window in which two
            # concurrent requests both see "absent" and both insert. The unique constraint
            # rejects the second; re-reading after the rollback turns it into the same
            # idempotent result the caller would otherwise have seen.
            self.session.rollback()
            existing = self.documents.get_by_content_hash(content_hash)
            if existing is None:
                raise  # not caused by a duplicate, so report it rather than swallow it
            return IngestionResult(
                document_id=existing.id,
                chunk_count=self.chunks.count_by_document(existing.id),
                created=False,
            )

        return IngestionResult(
            document_id=document.id,
            chunk_count=len(saved),
            created=True,
        )

    @staticmethod
    def _validate(title: str, content: str) -> None:
        """Enforce the business rules.

        The API layer already validates the request shape with pydantic, but the service
        is also called by scripts, tests and evaluation code. Validating next to the data
        means no entry point can bypass it.
        """
        if not title.strip():
            raise ValueError("title must not be empty")
        if not content.strip():
            raise ValueError("content must not be empty")
