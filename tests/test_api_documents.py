"""Document API tests.

Unlike test_ingestion.py these go through the full HTTP stack (routing, dependency
injection, service, repository, PostgreSQL), so they cover status codes, JSON field names
and request validation. Service internals remain test_ingestion.py's job.
"""

import uuid
from collections.abc import Iterator
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.core.database import get_db_session
from app.main import create_app
from app.models.tables import Chunk, Document
from app.rag.embeddings import get_embedding_client
from tests.fakes import FakeEmbeddingClient

# 400 words; with the default chunk_size=180 / overlap=30 this yields 3 chunks
CONTENT = " ".join(f"token{i}" for i in range(400))


def _count(session: Session, model: type) -> int:
    return session.scalar(select(func.count()).select_from(model)) or 0


@pytest.fixture
def embedder() -> FakeEmbeddingClient:
    return FakeEmbeddingClient()


@pytest.fixture
def client(db_session: Session, embedder: FakeEmbeddingClient) -> Iterator[TestClient]:
    """An app wired to the test database, with no real model loaded.

    Two dependencies are overridden:

    - `get_db_session` would open its own connection, which bypasses the transactional
      isolation from conftest: the data would really be written and tests would pollute
      each other.
    - `get_embedding_client` is a process-wide singleton. Left in place, the first POST
      would download and load the real sentence-transformers model, and once built the
      singleton would stay loaded for the rest of the session.

    `dependency_overrides` is FastAPI's mechanism for this: `Depends(...)` resolves to the
    object provided here.
    """
    app = create_app()
    app.dependency_overrides[get_db_session] = lambda: db_session
    app.dependency_overrides[get_embedding_client] = lambda: embedder

    # `with` rather than a bare TestClient(app) so the startup/shutdown lifespan runs and
    # tests follow the same path as production.
    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_create_document_returns_id_and_chunk_count(client: TestClient, db_session: Session) -> None:
    response = client.post(
        "/documents",
        json={"title": "Attention", "source": "arxiv:1706.03762", "content": CONTENT},
    )

    assert response.status_code == 201

    body = response.json()
    uuid.UUID(body["document_id"])  # raises if the id is not a valid UUID
    assert body["chunk_count"] == 3

    # The response claims success, so the rows must actually be there
    assert _count(db_session, Document) == 1
    assert _count(db_session, Chunk) == 3


def test_create_document_reports_whether_it_was_new(client: TestClient) -> None:
    payload = {"title": "Attention", "source": "arxiv", "content": CONTENT}

    first = client.post("/documents", json=payload).json()
    second = client.post("/documents", json=payload).json()

    assert first["created"] is True
    assert second["created"] is False
    assert second["document_id"] == first["document_id"]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="all-fields-missing"),
        pytest.param({"title": "", "source": "arxiv", "content": CONTENT}, id="empty-title"),
        pytest.param({"title": "t", "source": "", "content": CONTENT}, id="empty-source"),
        pytest.param({"title": "t", "source": "arxiv", "content": ""}, id="empty-content"),
        pytest.param({"title": "t" * 301, "source": "arxiv", "content": CONTENT}, id="title-too-long"),
        # The last two pass pydantic (non-zero length) but fail the service's blank check,
        # so they cover the path where a ValueError becomes a 422.
        pytest.param({"title": "   ", "source": "arxiv", "content": CONTENT}, id="blank-title"),
        pytest.param({"title": "t", "source": "arxiv", "content": "  \n\t "}, id="blank-content"),
    ],
)
def test_create_document_rejects_invalid_payload(
    client: TestClient, db_session: Session, payload: dict
) -> None:
    response = client.post("/documents", json=payload)

    assert response.status_code == 422
    # A failed validation must write nothing
    assert _count(db_session, Document) == 0
    assert _count(db_session, Chunk) == 0


# --- are vectors really generated on import -----------------------------------------
#
# Regression tests for a step that once fell between tasks: the design called for
# generating embeddings, but nothing implemented it, so chunks were stored with a NULL
# vector and retrieval could never see them. If that step goes missing again, these fail.


def _count_without_embedding(session: Session) -> int:
    return session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.embedding.is_(None))
    ) or 0


def _payload(content: str = CONTENT) -> dict:
    return {"title": "Attention", "source": "arxiv:1706.03762", "content": content}


def test_create_document_embeds_every_chunk(
    client: TestClient, db_session: Session, embedder: FakeEmbeddingClient
) -> None:
    """Every chunk must have a vector after POST /documents.

    Asserting that the document was stored is not enough: in the bug this pins, documents
    and chunks were stored perfectly well, only without vectors, and retrieval could never
    find them.
    """
    body = client.post("/documents", json=_payload()).json()

    assert body["chunk_count"] == 3
    assert body["embedded_chunk_count"] == 3
    assert _count_without_embedding(db_session) == 0


def test_create_document_stores_vectors_that_match_the_chunk_text(
    client: TestClient, db_session: Session, embedder: FakeEmbeddingClient
) -> None:
    """Each vector must be computed from that chunk's own text."""
    client.post("/documents", json=_payload())

    chunks = list(db_session.scalars(select(Chunk).order_by(Chunk.chunk_index)))
    for chunk in chunks:
        assert list(chunk.embedding) == pytest.approx(
            embedder.embed_text(chunk.text), abs=1e-6
        )


def test_create_document_does_not_reembed_on_duplicate(
    client: TestClient, db_session: Session
) -> None:
    """A duplicate import must not encode everything again.

    The assertion is on `embedded_chunk_count == 0` (nothing was written this time), not on
    the vectors still being present: the latter would not prove no work was repeated.
    """
    client.post("/documents", json=_payload())

    second = client.post("/documents", json=_payload()).json()

    assert second["created"] is False
    assert second["chunk_count"] == 3
    assert second["embedded_chunk_count"] == 0


def test_create_document_backfills_vectors_lost_earlier(
    client: TestClient, db_session: Session
) -> None:
    """Missing vectors are restored by re-importing the same content.

    Import and indexing are two separate transactions, so a failed indexing run (killed
    process, out of memory) leaves the document stored and the response a 500. Retrying the
    same content takes the duplicate branch, which runs indexing again and fills in the
    missing vectors: a failed request does not mean lost data.
    """
    client.post("/documents", json=_payload())

    # Force the state a failed indexing run leaves behind
    db_session.execute(update(Chunk).values(embedding=None), execution_options={"synchronize_session": False})
    db_session.flush()
    db_session.expire_all()
    assert _count_without_embedding(db_session) == 3

    body = client.post("/documents", json=_payload()).json()

    assert body["created"] is False
    assert body["embedded_chunk_count"] == 3
    assert _count_without_embedding(db_session) == 0


def test_create_document_fails_loudly_when_indexing_fails(db_session: Session) -> None:
    """When indexing fails the request must fail too, never return a 201 that looks fine.

    Catching the indexing error and still returning 201 with embedded_chunk_count=0 leaves
    every earlier test green while every vector is missing: the document lists, but
    retrieval can never find it, and nothing signals it.

    The document and chunks stay committed (import and indexing are separate
    transactions), so a retry can fill in the vectors -- see the test above.
    """

    class BrokenClient(FakeEmbeddingClient):
        def embed_texts(self, texts):  # type: ignore[override]
            raise RuntimeError("model exploded")

    app = create_app()
    app.dependency_overrides[get_db_session] = lambda: db_session
    app.dependency_overrides[get_embedding_client] = lambda: BrokenClient()

    # raise_server_exceptions=False: by default TestClient re-raises the server error and
    # the status code cannot be asserted, which is exactly what is under test here.
    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.post("/documents", json=_payload())

    assert response.status_code == 500
    # The document survived (separate transactions), but holds no vectors
    assert _count(db_session, Document) == 1
    assert _count(db_session, Chunk) == 3
    assert _count_without_embedding(db_session) == 3


def test_list_documents_starts_empty(client: TestClient) -> None:
    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == []


def test_list_documents_returns_imported_documents(client: TestClient) -> None:
    client.post("/documents", json={"title": "A", "source": "src-a", "content": CONTENT})
    client.post("/documents", json={"title": "B", "source": "src-b", "content": CONTENT + " extra"})

    body = client.get("/documents").json()

    assert {item["title"] for item in body} == {"A", "B"}
    # The field set is part of the response contract: extra fields leak internals, missing
    # ones break clients
    assert all(set(item) == {"id", "title", "source", "created_at"} for item in body)


def test_list_documents_returns_newest_first(client: TestClient, db_session: Session) -> None:
    """Ordering is tested with fixed timestamps, not with "inserted first comes first".

    The latter can fail on Windows: two inserts landing in the same clock tick produce
    identical created_at values and an undefined order. Tests must not depend on that luck.
    """
    db_session.add_all(
        [
            Document(
                title="older",
                source="s",
                content_hash="a" * 64,
                created_at=datetime(2020, 1, 1, tzinfo=timezone.utc),
            ),
            Document(
                title="newer",
                source="s",
                content_hash="b" * 64,
                created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
        ]
    )
    db_session.flush()

    body = client.get("/documents").json()

    assert [item["title"] for item in body] == ["newer", "older"]
