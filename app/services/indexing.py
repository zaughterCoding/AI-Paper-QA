"""Indexing service: give already-stored chunks their vectors.

Embedding is kept out of ingestion for four reasons. Ingestion stays fast, writing text
without loading a model or waiting on encoding. A failed encoding run is retryable, since
the document is already stored and indexing can simply be repeated rather than the
document re-imported. Swapping the model has an explicit rebuild entry point instead of
"delete everything and import it again". And the ingestion service never needs to know
that ``EmbeddingClient`` exists.

The cost is that something must trigger it: the ``POST /documents`` route calls it after
``ingest()``, and ``scripts/index_pending.py`` covers the manual path.
"""

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.rag.embeddings import EmbeddingClient
from app.repositories.chunks import ChunkRepository
from app.repositories.documents import DocumentRepository


@dataclass(frozen=True)
class IndexingResult:
    """Outcome of one indexing run.

    Both counts are reported because "embedded 0" has two very different meanings:
    ``embedded_count=0, skipped_count=0`` means the document has no chunks at all, while
    ``embedded_count=0, skipped_count=N`` means all N already had vectors. A single number
    could not tell those apart.
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
        """Embed every chunk of one document that has no vector yet.

        Idempotent: chunks that already have a vector are skipped, never re-encoded or
        overwritten.

        All or nothing: the commit happens once every vector is written, so a failure
        part-way through rolls the whole transaction back rather than leaving half the
        chunks indexed.
        """
        # An unknown document raises instead of returning zero. Returning zero would look
        # like a successful no-op and hide the upstream bug that produced the bad id.
        if self.documents.get_by_id(document_id) is None:
            raise ValueError(f"document not found: {document_id}")

        pending = self.chunks.list_without_embedding(document_id=document_id)

        if not pending:
            # Nothing to do, so the model is never called.
            return IndexingResult(
                embedded_count=0,
                skipped_count=self.chunks.count_by_document(document_id),
            )

        texts = [chunk.text for chunk in pending]
        vectors = self.embedding_client.embed_texts(texts)

        # This guards a silent failure rather than an expected one. zip() below stops at
        # the shorter side, so a client returning too few vectors would leave chunks
        # permanently unembedded with no error raised anywhere.
        if len(vectors) != len(texts):
            raise ValueError(
                f"embedding client returned {len(vectors)} vectors for {len(texts)} texts"
            )

        for chunk, vector in zip(pending, vectors, strict=True):
            self.chunks.update_embedding(chunk.id, vector)

        # Transaction boundary: everything is written before anything is committed.
        self.session.commit()

        return IndexingResult(
            embedded_count=len(pending),
            skipped_count=self.chunks.count_by_document(document_id) - len(pending),
        )

    def index_all_pending(self) -> IndexingResult:
        """Embed every chunk in the database that has no vector yet.

        Unlike index_document this spans documents, so it commits once at the end: either
        the whole rebuild lands or none of it does. Looping over index_document would
        commit per document and could stop half-way, leaving the caller to work out where
        to resume.
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
