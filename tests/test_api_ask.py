"""Ask API tests.

These run the full HTTP stack -- routing, dependency injection, retrieval, PostgreSQL --
with only the two model calls replaced. What they cover is the contract a client sees:
status codes, JSON field names, and which failures a caller can do something about.

The RAG behaviour underneath is not retested. Whether the prompt is assembled correctly
belongs to test_answering.py, and whether retrieval ranks correctly belongs to
test_retrieval.py. Repeating them here would add a third place to update and no coverage.

The one thing that *is* pinned here and nowhere else is the seam: that a question and a
top_k arriving as JSON reach the retrieval service unchanged, and that the answer and its
sources come back out in a shape a client can use.
"""

import re
from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.api import routes
from app.api.schemas import AskRequest
from app.core.database import get_db_session
from app.main import create_app
from app.models.tables import Chunk, Document, QALog
from app.rag.embeddings import get_embedding_client
from app.rag.llm import LLMClient, LLMError, get_llm_client
from app.services.answering import NO_SOURCES_ANSWER
from app.services.retrieval import DEFAULT_TOP_K, MAX_TOP_K
from tests.fakes import FakeEmbeddingClient, FakeLLMClient

# 400 words yields 3 chunks at the default chunk_size; 800 yields 6, which is more than
# DEFAULT_TOP_K and so makes "the default was applied" observable.
THREE_CHUNKS = 400
SIX_CHUNKS = 800

# A key-shaped string that must never appear in a response.
SECRET = "sk-secret-value-must-not-leak"


def _count(session: Session, model: type) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


def count_logs(session: Session) -> int:
    """Audit rows in the table. Zero is what most of these tests assert, since the fixture
    rolls every test back and the table therefore starts empty."""
    return _count(session, QALog)


@contextmanager
def app_client(db_session: Session, embedder: FakeEmbeddingClient, llm) -> Iterator[TestClient]:
    """An app wired to the test database, with no real model loaded and no network.

    Three dependencies are overridden:

    - `get_db_session` would open its own connection, bypassing conftest's transaction, so
      the data would really be written and tests would pollute each other.
    - `get_embedding_client` left in place would download and load the real
      sentence-transformers model on the first request, and the singleton would stay
      loaded for the rest of the session.
    - `get_llm_client` likewise, and it would call a real API.

    A context manager rather than a fixture so a test can supply a different client --
    a failing one, or a real LLMClient over a mock transport -- without a second fixture.
    """
    app = create_app()
    app.dependency_overrides[get_db_session] = lambda: db_session
    app.dependency_overrides[get_embedding_client] = lambda: embedder
    app.dependency_overrides[get_llm_client] = lambda: llm

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def embedder() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture
def llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def client(
    db_session: Session, embedder: FakeEmbeddingClient, llm: FakeLLMClient
) -> Iterator[TestClient]:
    with app_client(db_session, embedder, llm) as test_client:
        yield test_client


def import_document(
    client: TestClient, words: int = THREE_CHUNKS, title: str = "Attention Is All You Need"
) -> dict:
    """Import a document through the real endpoint and return its response body.

    Going through POST /documents rather than inserting rows keeps the test honest about
    what retrieval will actually see: chunks are stored the way ingestion stores them,
    vectors included.
    """
    content = " ".join(f"token{index}" for index in range(words))
    response = client.post(
        "/documents", json={"title": title, "source": "arxiv:1706.03762", "content": content}
    )
    assert response.status_code == 201
    return response.json()


# --- the happy path -----------------------------------------------------------


def test_ask_returns_the_answer_and_its_sources(client: TestClient) -> None:
    import_document(client)

    response = client.post("/ask", json={"question": "what is self attention?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "A canned answer [1]."
    assert len(body["sources"]) == 3


def test_source_items_expose_exactly_the_documented_fields(client: TestClient) -> None:
    """The field set is part of the contract, in both directions.

    A missing field breaks clients; an extra one leaks internals. `chunk_id` is the
    specific leak being pinned here -- it is on the retrieved chunk and deliberately not
    published, because a caller identifies a source by document and position.
    """
    import_document(client)

    body = client.post("/ask", json={"question": "q"}).json()

    assert body["sources"]
    assert all(
        set(source) == {"document_id", "title", "chunk_index", "text", "score"}
        for source in body["sources"]
    )


def test_sources_come_back_in_the_order_the_prompt_numbered_them(
    client: TestClient, llm: FakeLLMClient
) -> None:
    """`[n]` in the answer must resolve to `sources[n - 1]`.

    This is the contract that makes citations usable, and it is the one thing that can
    only be checked across the boundary: the prompt is built inside the service and the
    response is built in the route, so a mismatch between them is invisible to both.
    """
    import_document(client, title="Alpha")
    import_document(client, title="Beta")

    body = client.post("/ask", json={"question": "q", "top_k": 4}).json()
    labelled = re.findall(r"^\[(\d+)\] (.+)$", llm.prompts[0], flags=re.MULTILINE)

    assert [title for _, title in labelled] == [source["title"] for source in body["sources"]]
    assert [int(number) for number, _ in labelled] == list(range(1, len(body["sources"]) + 1))


def test_ask_sends_the_question_to_the_model(client: TestClient, llm: FakeLLMClient) -> None:
    import_document(client)

    client.post("/ask", json={"question": "how many GPUs were used?"})

    assert "how many GPUs were used?" in llm.prompts[0]


def test_sources_are_ordered_best_first(client: TestClient) -> None:
    """Ordering is not cosmetic: it decides what `[1]` points at."""
    import_document(client, words=SIX_CHUNKS)

    body = client.post("/ask", json={"question": "q"}).json()

    scores = [source["score"] for source in body["sources"]]
    assert scores == sorted(scores, reverse=True)


def test_ask_does_not_change_the_corpus(client: TestClient, db_session: Session) -> None:
    """Asking reads and never writes the corpus, however the request turns out.

    Only the corpus is asserted here. ``/ask`` does write one audit row, and that is a
    different claim with its own tests further down; keeping the two apart means a change
    to logging cannot make this test pass or fail for the wrong reason.
    """
    import_document(client)
    before = (_count(db_session, Document), _count(db_session, Chunk))

    client.post("/ask", json={"question": "q"})

    assert (_count(db_session, Document), _count(db_session, Chunk)) == before


# --- top_k: the default, an explicit value, and the bounds --------------------


def test_ask_uses_the_default_top_k_when_the_caller_does_not_specify(
    client: TestClient,
) -> None:
    document = import_document(client, words=SIX_CHUNKS)
    # The precondition, asserted rather than assumed: if chunking ever produces no more
    # chunks than the default, this test would pass without testing anything.
    assert document["chunk_count"] > DEFAULT_TOP_K

    body = client.post("/ask", json={"question": "q"}).json()

    assert len(body["sources"]) == DEFAULT_TOP_K


def test_ask_honours_an_explicit_top_k(client: TestClient) -> None:
    import_document(client, words=SIX_CHUNKS)

    body = client.post("/ask", json={"question": "q", "top_k": 2}).json()

    assert len(body["sources"]) == 2


def test_ask_accepts_top_k_of_one(client: TestClient) -> None:
    import_document(client)

    body = client.post("/ask", json={"question": "q", "top_k": 1}).json()

    assert len(body["sources"]) == 1


def test_ask_accepts_top_k_at_the_upper_boundary(client: TestClient) -> None:
    """The bounds test needs an inside counterpart.

    "Rejects everything above the limit" is also what an off-by-one in the other direction
    looks like, and a suite that only checks the rejections cannot tell the two apart.
    """
    import_document(client)

    response = client.post("/ask", json={"question": "q", "top_k": MAX_TOP_K})

    assert response.status_code == 200


def test_the_request_bounds_come_from_the_retrieval_service() -> None:
    """The schema's limits and the service's limits must be the same numbers.

    Both layers validate deliberately -- the schema answers without a database round trip,
    the service is the authority for callers that are not HTTP. What must not happen is
    the two disagreeing: a literal 20 left in the schema while MAX_TOP_K moved to 30 would
    make the API reject 21 to 30 before the service ever saw it, and no other test would
    fail. Reading the published JSON schema is what catches that.
    """
    top_k = AskRequest.model_json_schema()["properties"]["top_k"]

    assert top_k["maximum"] == MAX_TOP_K
    assert top_k["minimum"] == 1
    assert top_k["default"] == DEFAULT_TOP_K


# --- validation: two layers, both ending in 422 -------------------------------

# The first four are stopped by pydantic before any code runs; the fifth passes pydantic --
# "   " has non-zero length -- and is stopped by the retrieval service's blank check. Both
# paths must produce 422, because a client has no use for the distinction.
#
# Note what is *not* here: a top_k the schema accepts but the service rejects. Since the
# two share one constant, that input cannot be expressed over HTTP, and the service's own
# bound is covered by test_retrieval.py instead.
INVALID_PAYLOADS = [
    pytest.param({}, id="question-missing"),
    pytest.param({"question": ""}, id="question-empty"),
    pytest.param({"question": 42}, id="question-not-a-string"),
    pytest.param({"question": "q", "top_k": 0}, id="top_k-zero"),
    pytest.param({"question": "q", "top_k": -1}, id="top_k-negative"),
    pytest.param({"question": "q", "top_k": MAX_TOP_K + 1}, id="top_k-above-maximum"),
    pytest.param({"question": "q", "top_k": "five"}, id="top_k-not-a-number"),
    pytest.param({"question": "   "}, id="question-blank"),
]


@pytest.mark.parametrize("payload", INVALID_PAYLOADS)
def test_ask_rejects_an_invalid_payload(
    client: TestClient, llm: FakeLLMClient, payload: dict
) -> None:
    response = client.post("/ask", json=payload)

    assert response.status_code == 422
    # A rejected request must not cost a completion. Asserting only the status would pass
    # just as well if the model had been called first and the answer thrown away.
    assert llm.prompts == []


# --- an empty corpus is a normal outcome, not an error -------------------------


def test_ask_on_an_empty_corpus_says_the_sources_are_insufficient(
    client: TestClient, llm: FakeLLMClient
) -> None:
    """200, not 404 or 422: nothing went wrong, the corpus simply has no answer."""
    response = client.post("/ask", json={"question": "what is self attention?"})

    assert response.status_code == 200
    assert response.json() == {"answer": NO_SOURCES_ANSWER, "sources": []}
    # With nothing to condition on the answer is already known, so the completion is
    # skipped: it would add latency and give the model a chance to answer from its own
    # memory instead of from sources.
    assert llm.prompts == []


# --- the model endpoint failing is a 502, never a 500 or a fake answer ---------


class FailingLLMClient:
    """Stands in for an LLMClient whose endpoint refused or was unreachable."""

    def generate(self, prompt: str) -> str:
        raise LLMError("the model endpoint failed: 503 Service Unavailable")


def test_ask_returns_502_when_the_model_endpoint_fails(
    db_session: Session, embedder: FakeEmbeddingClient
) -> None:
    """The request was fine and so was this code; the dependency failed.

    Not a 500: nothing here is broken. Not a 200 with the sources and no answer either --
    a silent downgrade would hide an outage behind a response that looks like a normal
    "the sources are insufficient".
    """
    with app_client(db_session, embedder, FailingLLMClient()) as client:
        import_document(client)
        response = client.post("/ask", json={"question": "q"})

    assert response.status_code == 502


# --- the audit log: one row per answered question -----------------------------
#
# The repository and the decision about failed writes are tested elsewhere; what is pinned
# here is the acceptance criterion a client can see, along with the two ways a request can
# finish without producing a row.


def test_ask_records_one_audit_row(client: TestClient, db_session: Session) -> None:
    """Exactly one row per successful request, holding the question, the answer and the
    chunk ids the model was shown."""
    import_document(client)

    response = client.post("/ask", json={"question": "how many GPUs were used?"})
    body = response.json()

    assert count_logs(db_session) == 1
    log = db_session.scalar(select(QALog))
    assert log.question == "how many GPUs were used?"
    assert log.answer == body["answer"]
    assert log.latency_ms >= 0
    # The ids are the stored chunk rows, identified through the response's own sources --
    # which carry document_id and chunk_index and deliberately no chunk id. Comparing in
    # that order also pins that the log keeps retrieval's ordering, so the ids line up with
    # the [n] markers in the answer.
    chunk_ids_by_index = {
        chunk.chunk_index: str(chunk.id) for chunk in db_session.scalars(select(Chunk))
    }
    assert log.retrieved_chunk_ids == [
        chunk_ids_by_index[source["chunk_index"]] for source in body["sources"]
    ]
    assert len(log.retrieved_chunk_ids) == len(body["sources"])


def test_ask_records_a_row_for_an_unanswerable_question(
    client: TestClient, db_session: Session
) -> None:
    """An empty corpus is a 200 with no sources, and it is still an answered request.

    These rows are the interesting ones: counting them is how the retrieval miss rate
    becomes visible, so an implementation that only logged when the model was called would
    hide exactly the cases worth looking at.
    """
    response = client.post("/ask", json={"question": "what is self attention?"})

    assert response.status_code == 200
    log = db_session.scalar(select(QALog))
    assert log.retrieved_chunk_ids == []
    assert log.answer == NO_SOURCES_ANSWER


def test_ask_records_nothing_when_the_model_endpoint_fails(
    db_session: Session, embedder: FakeEmbeddingClient
) -> None:
    """A 502 has no answer. Recording the failure would put a row in the table that means
    "we answered nothing", which is not the same as "we could not ask"."""
    with app_client(db_session, embedder, FailingLLMClient()) as client:
        import_document(client)
        response = client.post("/ask", json={"question": "q"})

    assert response.status_code == 502
    assert count_logs(db_session) == 0


def test_ask_records_nothing_when_the_request_is_rejected(
    client: TestClient, db_session: Session
) -> None:
    import_document(client)

    assert client.post("/ask", json={"question": ""}).status_code == 422

    assert count_logs(db_session) == 0


class BrokenQALogRepository:
    """A repository whose writes fail the way an unavailable database fails.

    Mirrors the real signature rather than taking ``**kwargs``: a signature mismatch would
    raise TypeError, which is deliberately *not* swallowed, and this test would then fail
    for a reason that has nothing to do with what it is checking.
    """

    def __init__(self, session: Session) -> None:
        pass

    def create_log(
        self, question: str, answer: str, retrieved_chunk_ids: list[str], latency_ms: int
    ) -> None:
        raise OperationalError("INSERT INTO qa_logs ...", {}, Exception("connection lost"))


def test_ask_still_answers_when_the_audit_write_fails(
    db_session: Session, embedder: FakeEmbeddingClient, llm: FakeLLMClient, monkeypatch
) -> None:
    """The answer survives an unwritable audit table, and the client never learns of it.

    This is the client-visible half of the decision made in AnswerService. A 500 here would
    discard a generated answer over a bookkeeping failure, and the caller has no way to tell
    that apart from the model having failed.
    """
    monkeypatch.setattr(routes, "QALogRepository", BrokenQALogRepository)

    with app_client(db_session, embedder, llm) as client:
        import_document(client)
        response = client.post("/ask", json={"question": "q"})

    assert response.status_code == 200
    assert response.json()["answer"] == "A canned answer [1]."
    assert count_logs(db_session) == 0


def test_ask_does_not_leak_the_api_key_when_the_provider_rejects_it(
    db_session: Session, embedder: FakeEmbeddingClient
) -> None:
    """A 502 must carry the diagnostic and nothing else.

    Built from the real LLMClient over a mock transport rather than the fake above: the
    question is what httpx's own exception text contains, and a hand-written fake would
    answer that with whatever message I chose to put in it. Here the 401 comes from a real
    response object, through the real wrapping code, and the key really is on the request
    that produced it.
    """

    def reject(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {SECRET}"
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    llm = LLMClient(
        api_key=SECRET,
        model="test-model",
        base_url="https://example.invalid",
        http_client=httpx.Client(transport=httpx.MockTransport(reject)),
    )

    with app_client(db_session, embedder, llm) as client:
        import_document(client)
        response = client.post("/ask", json={"question": "q"})

    assert response.status_code == 502
    assert SECRET not in response.text
    # The provider's status is preserved through the LLMError wrapping: without it, an
    # expired key and a provider outage would look identical to whoever debugs this.
    assert "401" in response.json()["detail"]
