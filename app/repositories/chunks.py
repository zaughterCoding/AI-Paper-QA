"""Reads and writes for the ``chunks`` table."""

import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.tables import Chunk, Document
from app.rag.chunking import TextChunk


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned by a vector search.

    A plain dataclass rather than the ORM object, for three reasons. It carries the
    parent document's title, which lives in another table and is not a Chunk attribute.
    Being frozen, a caller cannot accidentally mutate what looks like stored data. And it
    survives the session closing, whereas touching a detached ORM instance raises
    DetachedInstanceError.

    ``score`` is cosine similarity in [-1, 1], higher meaning more similar. Stored vectors
    are normalised, so it equals the dot product of the two vectors.
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
        """Insert every chunk of one document.

        add_all accumulates a single batched INSERT (executemany), far faster than one
        INSERT per chunk.

        No embedding is written here: ingestion stores text only, and vectors are
        backfilled in a later pass, which is why the column is nullable.
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
        # flush surfaces constraint violations immediately: a duplicate chunk_index from
        # the splitter fails here rather than at commit, which is easier to trace.
        self.session.flush()
        return rows

    def count_by_document(self, document_id: uuid.UUID) -> int:
        """Number of chunks belonging to one document.

        COUNT(*) in the database rather than len(document.chunks), which would load every
        chunk's full text into memory just to count rows.
        """
        count = self.session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.document_id == document_id)
        )
        return count or 0

    def count_all(self) -> int:
        """Total number of chunks in the database."""
        count = self.session.scalar(select(func.count()).select_from(Chunk))
        return count or 0

    def list_without_embedding(self, document_id: uuid.UUID | None = None) -> list[Chunk]:
        """Chunks that have no vector yet -- the input to a backfill run.

        Scoped to one document when ``document_id`` is given, otherwise to the whole
        database.

        The ordering is for determinism, not correctness: without ORDER BY, Postgres may
        return the same rows in a different order on each run, which makes a test that
        depends on which chunk is processed first flaky.
        """
        statement = select(Chunk).where(Chunk.embedding.is_(None))
        if document_id is not None:
            statement = statement.where(Chunk.document_id == document_id)

        return list(self.session.scalars(statement.order_by(Chunk.document_id, Chunk.chunk_index)))

    def update_embedding(self, chunk_id: uuid.UUID, embedding: list[float]) -> None:
        """Store the vector for one chunk.

        Raises when the chunk is missing instead of returning quietly. The caller is a
        backfill run, so an unknown id means something upstream is wrong, and skipping it
        would leave chunks permanently unsearchable with nothing to show why.
        """
        chunk = self.session.get(Chunk, chunk_id)
        if chunk is None:
            raise ValueError(f"chunk not found: {chunk_id}")

        chunk.embedding = embedding
        # flush, not commit: the repository does not decide whether this write succeeds
        # together with others.
        self.session.flush()

    def search_similar(
        self, query_embedding: list[float], top_k: int
    ) -> list[RetrievedChunk]:
        """The ``top_k`` chunks closest to ``query_embedding``.

        The generated SQL is roughly:

            SELECT chunks.id, chunks.document_id, documents.title, ...,
                   1 - (chunks.embedding <=> :query) AS score
            FROM chunks JOIN documents ON documents.id = chunks.document_id
            WHERE chunks.embedding IS NOT NULL
            ORDER BY chunks.embedding <=> :query
            LIMIT :top_k

        Three points. ``<=>`` is pgvector's cosine distance operator, so it sorts ascending
        while the returned score is converted to "higher is more similar" for callers. The
        ``WHERE embedding IS NOT NULL`` is load-bearing: chunks without a vector are a
        normal state and must not take part in the ordering. And the JOIN supplies the
        title, without which an answer could not cite its source.
        """
        if top_k < 1:
            # Postgres rejects LIMIT -1 outright; failing here gives a clearer message.
            raise ValueError("top_k must be >= 1")

        distance = Chunk.embedding.cosine_distance(query_embedding)

        statement = (
            select(
                Chunk.id.label("chunk_id"),
                Chunk.document_id.label("document_id"),
                Document.title.label("title"),
                Chunk.chunk_index.label("chunk_index"),
                Chunk.text.label("text"),
                # distance -> similarity; both vectors are normalised, so this is cosine.
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
