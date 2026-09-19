"""Vector retrieval tests, covering the repository layer and the service layer.

Retrieval splits into two layers that fail differently:

- repository: is the SQL right? sorting direction, NULL filtering, JOIN fan-out. A mistake
  returns the wrong chunks, which is usually obvious.
- service: is the orchestration right? validation before encoding, and how many times each
  step runs. A mistake here produces results that look completely normal, so only pinned
  call counts catch it.

Test data uses the real 384 dimensions: `chunks.embedding` is declared vector(384), so
3-dimension fake vectors cannot be inserted, and the dimension under test should be the
real one anyway. Vectors come from tests/fakes.py, where shared words mean more similar
vectors, which is what makes the ranking assertions meaningful.
"""

import uuid

import pytest
from sqlalchemy.orm import Session

from app.models.tables import Chunk, Document
from app.repositories.chunks import ChunkRepository, RetrievedChunk
from app.services.retrieval import DEFAULT_TOP_K, MAX_TOP_K, RetrievalService
from tests.fakes import FakeEmbeddingClient


@pytest.fixture
def embedder() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture
def repo(db_session: Session) -> ChunkRepository:
    return ChunkRepository(db_session)


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


# --- empty database and edge cases --------------------------------------------


def test_search_on_empty_database_returns_empty_list(repo, embedder) -> None:
    assert repo.search_similar(embedder.embed_text("anything"), top_k=5) == []


def test_search_rejects_top_k_below_one(repo, embedder) -> None:
    """The database rejects LIMIT -1 outright; catching it here yields a usable error."""
    with pytest.raises(ValueError):
        repo.search_similar(embedder.embed_text("anything"), top_k=0)


def test_search_skips_chunks_without_embedding(db_session, repo, embedder) -> None:
    """Chunks with no vector must not appear in the results.

    This is not hypothetical: import stores text only and vectors are backfilled later, so
    chunks with a NULL embedding legitimately exist in the meantime.
    """
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "self attention", embedding=None)
    add_chunk(
        db_session, document, 1, "self attention", embedding=embedder.embed_text("self attention")
    )

    results = repo.search_similar(embedder.embed_text("self attention"), top_k=10)

    assert len(results) == 1
    assert results[0].chunk_index == 1


# --- ordering: the core of retrieval ------------------------------------------


def test_search_returns_most_similar_chunk_first(db_session, repo, embedder) -> None:
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "banana bread recipe", embedder.embed_text("banana bread recipe"))
    add_chunk(db_session, document, 1, "self attention mechanism", embedder.embed_text("self attention mechanism"))
    add_chunk(db_session, document, 2, "gradient descent optimizer", embedder.embed_text("gradient descent optimizer"))

    results = repo.search_similar(embedder.embed_text("self attention layer"), top_k=3)

    assert results[0].text == "self attention mechanism"


def test_search_orders_by_decreasing_similarity(db_session, repo, embedder) -> None:
    """Results come back by decreasing similarity.

    The chunks share 4, 3, 2 and 1 of the query's words, giving four distinct cosine
    scores, so the whole order can be asserted rather than just the first row. Tied scores
    would make the order among them undefined and the assertion flaky.
    """
    document = make_document(db_session)
    texts = ["alpha beta gamma delta", "alpha beta gamma", "alpha beta", "alpha"]
    for index, text in enumerate(texts):
        add_chunk(db_session, document, index, text, embedder.embed_text(text))

    results = repo.search_similar(embedder.embed_text("alpha beta gamma delta"), top_k=4)

    assert [result.text for result in results] == texts
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


def test_search_score_of_identical_text_is_close_to_one(db_session, repo, embedder) -> None:
    """score is a cosine similarity, so identical text scores 1.0.

    The assertion is about meaning, not just a number: writing `distance` instead of
    `1 - distance`, or dropping the normalization, fails here immediately.
    """
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "self attention", embedder.embed_text("self attention"))

    results = repo.search_similar(embedder.embed_text("self attention"), top_k=1)

    assert results[0].score == pytest.approx(1.0, abs=1e-6)


# --- top_k and the shape of the results ---------------------------------------


def test_search_respects_top_k(db_session, repo, embedder) -> None:
    document = make_document(db_session)
    for index in range(5):
        text = f"chunk number {index}"
        add_chunk(db_session, document, index, text, embedder.embed_text(text))

    assert len(repo.search_similar(embedder.embed_text("chunk number 0"), top_k=2)) == 2
    assert len(repo.search_similar(embedder.embed_text("chunk number 0"), top_k=5)) == 5


def test_search_returns_each_chunk_at_most_once(db_session, repo, embedder) -> None:
    """The JOIN with documents must not multiply the rows.

    Classic fan-out: a chunk joined to several document rows would appear several times and
    quietly hand back fewer than top_k distinct chunks.
    """
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "self attention", embedder.embed_text("self attention"))

    results = repo.search_similar(embedder.embed_text("self attention"), top_k=10)

    assert len(results) == 1
    assert len({result.chunk_id for result in results}) == len(results)


def test_search_returns_metadata_from_both_tables(db_session, repo, embedder) -> None:
    """Results carry fields from both tables.

    The title is what lets an answer cite its source; without it a result is an isolated
    paragraph the user cannot verify.
    """
    document = make_document(db_session, title="Attention Is All You Need")
    chunk = add_chunk(
        db_session, document, 3, "self attention", embedder.embed_text("self attention")
    )

    result = repo.search_similar(embedder.embed_text("self attention"), top_k=1)[0]

    assert isinstance(result, RetrievedChunk)
    assert result.chunk_id == chunk.id
    assert result.document_id == document.id
    assert result.title == "Attention Is All You Need"
    assert result.chunk_index == 3
    assert result.text == "self attention"


def test_search_spans_multiple_documents(db_session, repo, embedder) -> None:
    """Retrieval is global, not limited to a single document.

    This is a key difference from pasting a whole paper into the model: a question can be
    answered from any document.
    """
    first = make_document(db_session, title="Paper A", content_hash="a" * 64)
    second = make_document(db_session, title="Paper B", content_hash="b" * 64)
    add_chunk(db_session, first, 0, "banana bread recipe", embedder.embed_text("banana bread recipe"))
    add_chunk(db_session, second, 0, "self attention", embedder.embed_text("self attention"))

    results = repo.search_similar(embedder.embed_text("self attention"), top_k=5)

    assert results[0].title == "Paper B"


# --- backfilling vectors ------------------------------------------------------


def test_update_embedding_writes_the_vector(db_session, repo, embedder) -> None:
    document = make_document(db_session)
    chunk = add_chunk(db_session, document, 0, "self attention", embedding=None)
    vector = embedder.embed_text("self attention")

    repo.update_embedding(chunk.id, vector)
    db_session.flush()

    # Re-read from the database instead of trusting the in-memory object
    stored = db_session.get(Chunk, chunk.id)
    assert stored is not None
    assert stored.embedding is not None
    assert list(stored.embedding) == pytest.approx(vector, abs=1e-6)


def test_update_embedding_makes_chunk_searchable(db_session, repo, embedder) -> None:
    """The point of the backfill is that the chunk becomes retrievable, which is what
    gives this method its meaning."""
    document = make_document(db_session)
    chunk = add_chunk(db_session, document, 0, "self attention", embedding=None)
    query = embedder.embed_text("self attention")

    assert repo.search_similar(query, top_k=5) == []

    repo.update_embedding(chunk.id, embedder.embed_text("self attention"))

    results = repo.search_similar(query, top_k=5)
    assert len(results) == 1
    assert results[0].chunk_id == chunk.id


def test_update_embedding_overwrites_existing_vector(db_session, repo, embedder) -> None:
    """A repeated backfill overwrites the old vector instead of raising or duplicating."""
    document = make_document(db_session)
    chunk = add_chunk(db_session, document, 0, "text", embedder.embed_text("banana bread recipe"))
    new_vector = embedder.embed_text("self attention")

    repo.update_embedding(chunk.id, new_vector)

    stored = db_session.get(Chunk, chunk.id)
    assert stored is not None
    assert list(stored.embedding) == pytest.approx(new_vector, abs=1e-6)

    results = repo.search_similar(embedder.embed_text("self attention"), top_k=1)
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


def test_update_embedding_raises_for_unknown_chunk(repo, embedder) -> None:
    """An unknown id raises instead of being skipped.

    Skipping quietly hides chunks that never get a vector, until someone notices retrieval
    is worse than expected with no idea where to look.
    """
    with pytest.raises(ValueError):
        repo.update_embedding(uuid.uuid4(), embedder.embed_text("self attention"))


# --- the retrieval service ----------------------------------------------------
#
# These do not re-test which chunks come back; they test the orchestration: whether
# validation runs before encoding, how many times each step runs, and whether arguments are
# forwarded unchanged.
#
# Such mistakes have no symptom. Encoding the question twice returns identical results,
# only slower and more expensive; a missing validation lets an empty question return a pile
# of chunks. Neither shows up in manual checking, so the counts are pinned instead.


class _CountingEmbeddingClient:
    """Wraps the fake client and records the text passed to each encode.

    A small hand-written class rather than Mock(): it needs one behaviour, and it keeps the
    real behaviour, so the returned vectors still match what is in the database and the
    same double serves as both a counter and a working client.
    """

    def __init__(self) -> None:
        self._inner = FakeEmbeddingClient()
        self.encoded_texts: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.encoded_texts.append(text)
        return self._inner.embed_text(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.encoded_texts.extend(texts)
        return self._inner.embed_texts(texts)


class _SearchSpy:
    """Wraps `search_similar` and records each call's arguments.

    Wrapping rather than monkeypatching the class: `RetrievalService.__init__` creates its
    own `ChunkRepository`, so the instance is only reachable as `service.chunks`. Python
    looks up instance attributes before class attributes, so an attribute of the same name
    set there intercepts the call while `_original` still runs the real query.
    """

    def __init__(self, repository: ChunkRepository) -> None:
        self._original = repository.search_similar
        self.calls: list[dict] = []

    def __call__(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        self.calls.append({"query_embedding": query_embedding, "top_k": top_k})
        return self._original(query_embedding=query_embedding, top_k=top_k)


@pytest.fixture
def counter() -> _CountingEmbeddingClient:
    return _CountingEmbeddingClient()


@pytest.fixture
def service(db_session: Session, counter: _CountingEmbeddingClient) -> RetrievalService:
    return RetrievalService(db_session, counter)


@pytest.fixture
def search_spy(service: RetrievalService) -> _SearchSpy:
    spy = _SearchSpy(service.chunks)
    service.chunks.search_similar = spy  # type: ignore[method-assign]
    return spy


def seed_chunks(db_session: Session, embedder: FakeEmbeddingClient) -> None:
    """One document with three chunks on different topics."""
    document = make_document(db_session)
    for index, text in enumerate(
        ["banana bread recipe", "self attention mechanism", "gradient descent optimizer"]
    ):
        add_chunk(db_session, document, index, text, embedder.embed_text(text))


# --- validation ---------------------------------------------------------------


def test_retrieve_rejects_an_empty_question(service) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        service.retrieve("")


def test_retrieve_rejects_a_whitespace_only_question(service) -> None:
    """A whitespace-only question carries no meaning either.

    `not question` misses it, since the string is non-empty, while the resulting vector is
    either all zeros or nothing but punctuation.
    """
    with pytest.raises(ValueError, match="must not be empty"):
        service.retrieve("   \n\t  ")


@pytest.mark.parametrize("top_k", [0, -1, -100, MAX_TOP_K + 1, 1000])
def test_retrieve_rejects_top_k_out_of_range(service, top_k: int) -> None:
    with pytest.raises(ValueError, match="top_k must be between"):
        service.retrieve("self attention", top_k=top_k)


@pytest.mark.parametrize("top_k", [1, DEFAULT_TOP_K, MAX_TOP_K])
def test_retrieve_accepts_top_k_within_range(db_session, embedder, counter, top_k: int) -> None:
    """The boundary values themselves must be valid.

    Testing only the out-of-range cases would also pass with the upper bound set to 0, after
    which the service could not retrieve anything at all. Every bound check needs an
    in-range check beside it, or a wrong threshold goes unnoticed.
    """
    seed_chunks(db_session, embedder)

    assert len(RetrievalService(db_session, counter).retrieve("self attention", top_k=top_k)) <= top_k


# --- orchestration: call counts and order -------------------------------------


def test_retrieve_embeds_the_question_exactly_once(db_session, embedder, counter) -> None:
    """The question is encoded once.

    A second encode is invisible in the output, it only adds a model forward pass and its
    cost to every question. Nothing but a pinned count catches that.
    """
    seed_chunks(db_session, embedder)

    RetrievalService(db_session, counter).retrieve("self attention")

    assert counter.encoded_texts == ["self attention"]


def test_retrieve_searches_the_database_exactly_once(db_session, embedder, service, search_spy) -> None:
    """The database is queried once as well, not once to look and once to fetch.

    The `service` and `search_spy` fixtures must yield the same instance: a second
    RetrievalService built in the test body would leave this assertion watching an object
    nobody calls. That version passes by accident, and is then "fixed" by weakening the
    count the next time someone touches the test.
    """
    seed_chunks(db_session, embedder)

    service.retrieve("self attention")

    assert len(search_spy.calls) == 1


def test_retrieve_passes_the_question_to_the_embedder_unchanged(db_session, embedder, counter) -> None:
    """The raw string reaches the embedder, not a stripped or truncated copy.

    Keeping it unchanged makes failures reproducible: the same string can be replayed
    without guessing what an intermediate layer did to it.
    """
    seed_chunks(db_session, embedder)

    RetrievalService(db_session, counter).retrieve("  self attention  ")

    assert counter.encoded_texts == ["  self attention  "]


def test_retrieve_forwards_top_k_to_the_repository(db_session, embedder, counter) -> None:
    """top_k is forwarded as given, neither dropped nor replaced by the default."""
    seed_chunks(db_session, embedder)
    service = RetrievalService(db_session, counter)
    spy = _SearchSpy(service.chunks)
    service.chunks.search_similar = spy  # type: ignore[method-assign]

    service.retrieve("self attention", top_k=2)

    assert spy.calls[0]["top_k"] == 2


def test_retrieve_uses_the_default_top_k_when_omitted(db_session, embedder, counter) -> None:
    """Without top_k the service uses DEFAULT_TOP_K rather than returning everything."""
    seed_chunks(db_session, embedder)
    document = make_document(db_session, content_hash="b" * 64)
    for index in range(DEFAULT_TOP_K + 3):
        text = f"self attention variant {index}"
        add_chunk(db_session, document, index, text, embedder.embed_text(text))

    results = RetrievalService(db_session, counter).retrieve("self attention")

    assert len(results) == DEFAULT_TOP_K


def test_retrieve_validates_the_question_before_encoding(service, counter) -> None:
    """The empty-question check must run before encoding.

    In the other order a meaningless question still costs a model forward pass, and some
    models return an all-zero vector for it, so retrieval returns arbitrarily ranked chunks
    and dresses up invalid input as poor retrieval quality.
    """
    with pytest.raises(ValueError):
        service.retrieve("")

    assert counter.encoded_texts == []


def test_retrieve_validates_top_k_before_encoding(db_session, embedder, counter) -> None:
    """An out-of-range top_k is caught before encoding for the same reason."""
    seed_chunks(db_session, embedder)

    with pytest.raises(ValueError):
        RetrievalService(db_session, counter).retrieve("self attention", top_k=0)

    assert counter.encoded_texts == []


def test_retrieve_does_not_commit(db_session, embedder, counter, monkeypatch) -> None:
    """Retrieval is read-only and must not commit.

    Not pedantry: a request may share this session between retrieval and other work, so a
    commit inside the retrieval path would take the transaction boundary away from the
    caller.
    """

    def fail() -> None:
        raise AssertionError("retrieve() must not commit")

    seed_chunks(db_session, embedder)
    monkeypatch.setattr(db_session, "commit", fail)

    assert RetrievalService(db_session, counter).retrieve("self attention")


# --- end to end: the results themselves ---------------------------------------


def test_retrieve_returns_chunks_ordered_by_similarity(db_session, embedder, counter) -> None:
    """Both layers together: the question is encoded, the database is queried, and the
    most relevant chunks come back."""
    seed_chunks(db_session, embedder)

    results = RetrievalService(db_session, counter).retrieve("self attention layer", top_k=3)

    assert [result.text for result in results][0] == "self attention mechanism"
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_spans_multiple_documents(db_session, embedder, counter) -> None:
    """Retrieval is global: the answer can live in any document.

    That is the point of this layer. A question is not bound to one document, and the user
    does not need to know which one holds the answer.
    """
    first = make_document(db_session, title="Paper A", content_hash="a" * 64)
    second = make_document(db_session, title="Paper B", content_hash="b" * 64)
    add_chunk(db_session, first, 0, "banana bread recipe", embedder.embed_text("banana bread recipe"))
    add_chunk(db_session, second, 0, "self attention", embedder.embed_text("self attention"))

    results = RetrievalService(db_session, counter).retrieve("self attention", top_k=5)

    assert results[0].title == "Paper B"


def test_retrieve_returns_empty_list_on_an_empty_database(service) -> None:
    """An empty database returns an empty list rather than raising.

    "Not found" and "failed" are different things: the first is a normal business result
    (no corpus imported yet, or a question outside the corpus), the second is what the
    caller has to handle. Merging them leaves the caller unable to tell "tell the user
    nothing was found" from "raise an alarm".
    """
    assert service.retrieve("self attention") == []


def test_retrieve_ignores_chunks_without_vectors(db_session, embedder, counter) -> None:
    """Chunks with no vector are not retrievable; this layer cannot bypass the repository's
    filter."""
    document = make_document(db_session)
    add_chunk(db_session, document, 0, "self attention", embedding=None)

    assert RetrievalService(db_session, counter).retrieve("self attention") == []
