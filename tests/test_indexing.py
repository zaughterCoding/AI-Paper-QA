"""Indexing service (vector backfill) tests.

This covers the write path, text becoming vectors that are stored correctly, while
test_retrieval.py covers the read path, finding the right chunk for a given vector. They
meet in `test_index_document_makes_chunks_searchable`, since making chunks retrievable is
the only point of the backfill.
"""

import uuid

import pytest
from sqlalchemy.orm import Session

from app.models.tables import Chunk, Document
from app.repositories.chunks import ChunkRepository
from app.services.indexing import IndexingResult, IndexingService
from tests.fakes import FakeEmbeddingClient


@pytest.fixture
def embedder() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture
def repo(db_session: Session) -> ChunkRepository:
    return ChunkRepository(db_session)


@pytest.fixture
def service(db_session: Session, embedder: FakeEmbeddingClient) -> IndexingService:
    return IndexingService(db_session, embedder)


def make_document(
    session: Session, title: str = "Attention Is All You Need", content_hash: str = "a" * 64
) -> Document:
    document = Document(title=title, source="test", content_hash=content_hash)
    session.add(document)
    session.flush()
    return document


def add_chunk(
    session: Session,
    document: Document,
    index: int,
    text: str,
    embedding: list[float] | None = None,
) -> Chunk:
    chunk = Chunk(
        document_id=document.id,
        chunk_index=index,
        text=text,
        token_count=len(text.split()),
        embedding=embedding,
    )
    session.add(chunk)
    session.flush()
    return chunk


def embedding_of(session: Session, chunk: Chunk) -> list[float] | None:
    """Read a chunk's vector back from the database rather than from memory.

    `expire()` marks the object stale so the next attribute access queries again.
    Without it the value read could be the one just written in memory, which would only
    prove the test's own write rather than that the data reached the database.
    """
    session.expire(chunk)
    stored = session.get(Chunk, chunk.id)
    assert stored is not None
    return list(stored.embedding) if stored.embedding is not None else None


def count_pending(session: Session) -> int:
    """How many chunks in the whole database still have no vector."""
    return len(ChunkRepository(session).list_without_embedding())


# --- repository: finding chunks to backfill -----------------------------------


def test_list_without_embedding_skips_chunks_that_already_have_vectors(db_session, repo) -> None:
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "done", embedding=[0.5] * 384)
    pending = add_chunk(db_session, document, 1, "todo")

    assert [chunk.id for chunk in repo.list_without_embedding()] == [pending.id]


def test_list_without_embedding_can_be_limited_to_one_document(db_session, repo) -> None:
    first = make_document(db_session, content_hash="a" * 64)
    second = make_document(db_session, content_hash="b" * 64)
    chunk_of_first = add_chunk(db_session, first, 0, "first")
    add_chunk(db_session, second, 0, "second")

    assert [chunk.id for chunk in repo.list_without_embedding(document_id=first.id)] == [
        chunk_of_first.id
    ]
    assert len(repo.list_without_embedding()) == 2


def test_list_without_embedding_returns_a_deterministic_order(db_session, repo) -> None:
    """The order must be deterministic and independent of insertion order.

    Without ORDER BY, PostgreSQL row order is undefined, so which chunk gets written first
    would be a flaky assertion. Rows are inserted out of order (3, 1, 2, 0) and must come
    back as 0, 1, 2, 3; without the ORDER BY this fails intermittently.
    """
    document = make_document(db_session)
    for index in (3, 1, 2, 0):
        add_chunk(db_session, document, index, f"text {index}")

    order = [chunk.chunk_index for chunk in repo.list_without_embedding()]

    assert order == [0, 1, 2, 3]


def test_count_all_counts_every_chunk_regardless_of_embedding(db_session, repo) -> None:
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "done", embedding=[0.5] * 384)
    add_chunk(db_session, document, 1, "todo")

    assert repo.count_all() == 2


# --- basic behaviour ----------------------------------------------------------


def test_index_document_writes_vectors_for_all_chunks(db_session, service, embedder) -> None:
    document = make_document(db_session)
    chunks = [add_chunk(db_session, document, i, f"chunk text {i}") for i in range(3)]

    result = service.index_document(document.id)

    assert result == IndexingResult(embedded_count=3, skipped_count=0)
    for chunk in chunks:
        assert embedding_of(db_session, chunk) == pytest.approx(
            embedder.embed_text(chunk.text), abs=1e-6
        )


def test_index_document_returns_result_with_both_counts(db_session, service) -> None:
    """The result has to tell "nothing pending" apart from "document missing"; a result
    carrying only the embedded count reports both as 0."""
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "already done", embedding=[0.1] * 384)
    add_chunk(db_session, document, 1, "needs work")

    result = service.index_document(document.id)

    assert result.embedded_count == 1
    assert result.skipped_count == 1


def test_index_document_only_embeds_the_given_document(db_session, service) -> None:
    """Encoding other documents along the way is as much a bug as encoding too few."""
    first = make_document(db_session, content_hash="a" * 64)
    second = make_document(db_session, content_hash="b" * 64)
    chunk_of_first = add_chunk(db_session, first, 0, "first document")
    chunk_of_second = add_chunk(db_session, second, 0, "second document")

    service.index_document(first.id)

    assert embedding_of(db_session, chunk_of_first) is not None
    assert embedding_of(db_session, chunk_of_second) is None


def test_index_document_raises_for_unknown_document(service) -> None:
    """An unknown id raises rather than returning 0, which would look like a run that
    finished successfully with nothing to do. Same principle as
    ChunkRepository.update_embedding.
    """
    with pytest.raises(ValueError):
        service.index_document(uuid.uuid4())


# --- what gets encoded --------------------------------------------------------


def test_index_document_embeds_the_chunk_text(db_session, service, embedder) -> None:
    """The chunk's own text must be encoded, not something else such as the title."""
    document = make_document(db_session, title="A Very Different Title")
    chunk = add_chunk(db_session, document, 0, "self attention mechanism")

    service.index_document(document.id)

    assert embedding_of(db_session, chunk) == pytest.approx(
        embedder.embed_text("self attention mechanism"), abs=1e-6
    )


def test_index_document_makes_chunks_searchable(db_session, service, repo, embedder) -> None:
    """The only point of the backfill is that the chunk becomes retrievable.

    Testing that a vector was written is not enough: it can be written and still not found
    (a missing normalization, a wrong column dimension), so the write and read paths are
    exercised together here.
    """
    document = make_document(db_session)
    chunk = add_chunk(db_session, document, 0, "self attention mechanism")
    query = embedder.embed_text("self attention mechanism")

    assert repo.search_similar(query, top_k=5) == []  # before the backfill

    service.index_document(document.id)

    results = repo.search_similar(query, top_k=5)  # after the backfill
    assert len(results) == 1
    assert results[0].chunk_id == chunk.id
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


# --- idempotence --------------------------------------------------------------


def test_index_document_is_idempotent(db_session, service) -> None:
    """A second call does not re-encode chunks that already have a vector."""
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "text")
    add_chunk(db_session, document, 1, "more text")

    first = service.index_document(document.id)
    second = service.index_document(document.id)

    assert first == IndexingResult(embedded_count=2, skipped_count=0)
    # The second call must skip everything, not recompute the same values
    assert second == IndexingResult(embedded_count=0, skipped_count=2)


def test_index_document_does_not_overwrite_existing_vector(db_session, service, embedder) -> None:
    """Chunks that already have a vector keep it.

    This is where "skip" and "recompute" differ: an implementation that re-encodes the
    whole document would replace the old vector. That is both a wasted encode and
    questionable, since the stored vector may come from another model.
    """
    document = make_document(db_session)
    old_vector = embedder.embed_text("banana bread recipe")
    chunk = add_chunk(db_session, document, 0, "self attention", embedding=old_vector)

    service.index_document(document.id)

    assert embedding_of(db_session, chunk) == pytest.approx(old_vector, abs=1e-6)


def test_index_document_with_nothing_pending_does_not_touch_the_client(db_session, service) -> None:
    """With nothing pending the model must not be called; an empty encode is wasted work
    and some model versions raise on it."""

    class ExplodingClient(FakeEmbeddingClient):
        def embed_texts(self, texts):  # type: ignore[override]
            raise AssertionError("embedding client must not be called when there is nothing pending")

    document = make_document(db_session)
    add_chunk(db_session, document, 0, "done", embedding=[0.1] * 384)

    result = IndexingService(db_session, ExplodingClient()).index_document(document.id)

    assert result == IndexingResult(embedded_count=0, skipped_count=1)


# --- whole-database backfill --------------------------------------------------


def test_index_all_pending_covers_every_document(db_session, service) -> None:
    first = make_document(db_session, content_hash="a" * 64)
    second = make_document(db_session, content_hash="b" * 64)
    add_chunk(db_session, first, 0, "alpha")
    add_chunk(db_session, first, 1, "beta")
    add_chunk(db_session, second, 0, "gamma")

    result = service.index_all_pending()

    assert result == IndexingResult(embedded_count=3, skipped_count=0)
    assert count_pending(db_session) == 0


def test_index_all_pending_skips_already_indexed_chunks(db_session, service, embedder) -> None:
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "old", embedding=embedder.embed_text("old"))
    add_chunk(db_session, document, 1, "new")

    result = service.index_all_pending()

    assert result == IndexingResult(embedded_count=1, skipped_count=1)


def test_index_all_pending_on_empty_database_is_a_noop(db_session, service) -> None:
    assert service.index_all_pending() == IndexingResult(embedded_count=0, skipped_count=0)


# --- a client returning the wrong number of vectors ---------------------------


def test_index_document_rejects_wrong_number_of_vectors(db_session) -> None:
    """A client returning too few vectors must raise rather than write only some.

    This guards a real silent failure: `zip()` stops at the shorter side, so the missing
    chunks would never get a vector and nothing would report it. The implementation
    therefore compares the lengths explicitly.
    """

    class ShortClient(FakeEmbeddingClient):
        def embed_texts(self, texts):  # type: ignore[override]
            return super().embed_texts(texts)[:-1]  # deliberately one short

    document = make_document(db_session)
    add_chunk(db_session, document, 0, "one")
    add_chunk(db_session, document, 1, "two")

    with pytest.raises(ValueError):
        IndexingService(db_session, ShortClient()).index_document(document.id)


def test_index_all_pending_rejects_wrong_number_of_vectors(db_session) -> None:
    class ShortClient(FakeEmbeddingClient):
        def embed_texts(self, texts):  # type: ignore[override]
            return []

    document = make_document(db_session)
    add_chunk(db_session, document, 0, "one")

    with pytest.raises(ValueError):
        IndexingService(db_session, ShortClient()).index_all_pending()
