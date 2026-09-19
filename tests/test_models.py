"""Schema tests: defaults, constraints, ORM relationships and the pgvector column, checked
against real PostgreSQL."""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.core.config import EMBEDDING_DIM
from app.models.tables import Chunk, Document


def _make_document(**overrides) -> Document:
    defaults = {
        "title": "Attention Is All You Need",
        "source": "https://arxiv.org/abs/1706.03762",
        "content_hash": uuid.uuid4().hex,  # different each time, so no unique clash
    }
    return Document(**{**defaults, **overrides})


def _vector_literal(values: list[float]) -> str:
    """pgvector's text form '[0.1,0.1,...]', for comparing in raw SQL."""
    return "[" + ",".join(str(v) for v in values) + "]"


def test_document_gets_uuid_and_timestamp(db_session):
    document = _make_document()
    db_session.add(document)
    db_session.commit()

    assert isinstance(document.id, uuid.UUID)
    assert document.created_at is not None
    # Timezone-aware UTC, not a naive local time
    assert document.created_at.tzinfo is not None


def test_document_content_hash_must_be_unique(db_session):
    db_session.add(_make_document(content_hash="same-hash"))
    db_session.commit()

    db_session.add(_make_document(content_hash="same-hash"))
    # The unique constraint must reject the duplicate rather than store two rows
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_chunk_links_back_to_document(db_session):
    document = _make_document()
    db_session.add(document)
    db_session.flush()  # document.id is available, without committing yet

    chunk = Chunk(
        document_id=document.id,
        chunk_index=0,
        text="Self-attention relates different positions of a single sequence.",
        token_count=9,
    )
    db_session.add(chunk)
    db_session.commit()

    # Back-reference from a chunk to its document
    assert chunk.document.id == document.id
    assert chunk.embedding is None  # no vector on a fresh import; a legal state


def test_chunk_accepts_vector_embedding(db_session):
    document = _make_document()
    db_session.add(document)
    db_session.flush()

    vector = [0.1] * EMBEDDING_DIM
    chunk = Chunk(
        document_id=document.id,
        chunk_index=0,
        text="dummy",
        token_count=1,
        embedding=vector,
    )
    db_session.add(chunk)
    db_session.commit()

    # Let the database compute the distance, which proves the vector was stored at 384
    # dimensions. `<=>` is pgvector's cosine distance; against itself it is 0.
    distance = db_session.execute(
        text("SELECT embedding <=> CAST(:query AS vector) FROM chunks WHERE id = :id"),
        {"query": _vector_literal(vector), "id": chunk.id},
    ).scalar()

    assert distance == pytest.approx(0.0)


def test_chunk_index_must_be_unique_per_document(db_session):
    document = _make_document()
    db_session.add(document)
    db_session.flush()

    db_session.add(Chunk(document_id=document.id, chunk_index=0, text="a", token_count=1))
    db_session.add(Chunk(document_id=document.id, chunk_index=0, text="b", token_count=1))
    with pytest.raises(IntegrityError):
        db_session.commit()
