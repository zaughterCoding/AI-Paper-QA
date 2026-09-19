"""Document ingestion service tests.

These run against real PostgreSQL (the `db_session` fixture from conftest), so they verify
that data really reaches the database rather than that the mocking lines up.
"""

import hashlib

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.tables import Chunk, Document
from app.rag.chunking import TextChunk, TextChunker
from app.repositories.documents import DocumentRepository
from app.services.ingestion import DocumentIngestionService

PARAGRAPH = (
    "The dominant sequence transduction models are based on complex recurrent "
    "or convolutional neural networks that include an encoder and a decoder."
)

# 400 words; with the default chunk_size=180 / overlap=30 this yields 3 chunks:
# [0:180] [150:330] [300:400], the last one stopping at the end of the text.
LONG_CONTENT = " ".join(f"token{i}" for i in range(400))


def _count(session: Session, model: type) -> int:
    """Row count of a table, via COUNT(*) rather than len(session.query(...).all())."""
    return session.scalar(select(func.count()).select_from(model)) or 0


def test_ingest_saves_document_and_chunks(db_session: Session) -> None:
    service = DocumentIngestionService(db_session)

    result = service.ingest(
        title="Attention Is All You Need",
        source="arxiv:1706.03762",
        content=LONG_CONTENT,
    )

    assert result.created is True
    assert result.chunk_count == 3

    document = db_session.get(Document, result.document_id)
    assert document is not None
    assert document.title == "Attention Is All You Need"
    assert document.source == "arxiv:1706.03762"

    # The chunks are stored and all belong to that document
    stored = db_session.scalars(
        select(Chunk).where(Chunk.document_id == result.document_id)
    ).all()
    assert len(stored) == 3


def test_ingest_stores_chunks_in_order_with_content(db_session: Session) -> None:
    # A small chunker makes the chunk boundaries easy to predict
    service = DocumentIngestionService(db_session, chunker=TextChunker(chunk_size=5, overlap=1))

    result = service.ingest(title="Counting", source="test", content="0 1 2 3 4 5 6 7 8 9")

    rows = db_session.scalars(
        select(Chunk)
        .where(Chunk.document_id == result.document_id)
        .order_by(Chunk.chunk_index)
    ).all()

    assert result.chunk_count == 3
    assert [row.chunk_index for row in rows] == [0, 1, 2]
    assert [row.text for row in rows] == ["0 1 2 3 4", "4 5 6 7 8", "8 9"]
    assert [row.token_count for row in rows] == [5, 5, 2]


def test_ingest_stores_sha256_of_content(db_session: Session) -> None:
    service = DocumentIngestionService(db_session)

    result = service.ingest(title="Attention", source="arxiv", content=PARAGRAPH)

    document = db_session.get(Document, result.document_id)
    assert document.content_hash == hashlib.sha256(PARAGRAPH.encode("utf-8")).hexdigest()


def test_ingest_is_idempotent_for_same_content(db_session: Session) -> None:
    """Importing the same content twice produces no second document.

    That is what the content_hash unique constraint is for, and the service has to handle
    it: uploading the same paper again is normal, and it should neither fail nor store a
    duplicate.
    """
    service = DocumentIngestionService(db_session)

    first = service.ingest(title="First title", source="src-a", content=PARAGRAPH)
    second = service.ingest(title="Second title", source="src-b", content=PARAGRAPH)

    assert second.document_id == first.document_id
    assert second.created is False
    # The second import reports the same chunk count, not 0
    assert second.chunk_count == first.chunk_count

    assert _count(db_session, Document) == 1
    assert _count(db_session, Chunk) == first.chunk_count

    # The existing row wins; the later title does not overwrite it
    document = db_session.get(Document, first.document_id)
    assert document.title == "First title"
    assert document.source == "src-a"


@pytest.mark.parametrize(
    ("title", "content"),
    [
        ("", PARAGRAPH),
        ("   ", PARAGRAPH),
        ("\n\t", PARAGRAPH),
        ("Attention", ""),
        ("Attention", "   \n\t  "),
    ],
)
def test_ingest_rejects_empty_title_or_content(
    db_session: Session, title: str, content: str
) -> None:
    service = DocumentIngestionService(db_session)

    with pytest.raises(ValueError):
        service.ingest(title=title, source="arxiv", content=content)

    # A rejected import must write nothing at all, not half a document
    assert _count(db_session, Document) == 0
    assert _count(db_session, Chunk) == 0


def test_ingest_commits_so_data_survives_rollback(db_session: Session) -> None:
    """The service commits its own transaction.

    The data is still there after an explicit rollback, which means ingest committed. Had
    it forgotten, the rows would still be sitting in an uncommitted transaction and would
    disappear here.
    """
    service = DocumentIngestionService(db_session)

    result = service.ingest(title="Attention", source="arxiv", content=PARAGRAPH)

    db_session.rollback()
    db_session.expunge_all()  # drop the identity map cache to force a real query

    assert db_session.get(Document, result.document_id) is not None
    assert _count(db_session, Chunk) == result.chunk_count


def test_ingest_recovers_from_concurrent_insert(
    db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The race fallback: nothing found on lookup, but the row exists by insert time.

    Two requests uploading the same paper at the same moment produce this, rarely and never
    reproducibly, so the first lookup is faked as a miss: the service then tries to insert,
    the unique constraint rejects it, and the except branch recovers. Without that fallback
    the user gets a 500 instead of "already imported".
    """
    service = DocumentIngestionService(db_session)
    first = service.ingest(title="Attention", source="arxiv", content=PARAGRAPH)

    real_lookup = DocumentRepository.get_by_content_hash
    calls = {"count": 0}

    def lookup_missing_first(self: DocumentRepository, content_hash: str):
        calls["count"] += 1
        if calls["count"] == 1:
            return None  # simulate: the other request has not committed yet
        return real_lookup(self, content_hash)

    monkeypatch.setattr(DocumentRepository, "get_by_content_hash", lookup_missing_first)

    second = service.ingest(title="Attention", source="arxiv", content=PARAGRAPH)

    assert calls["count"] == 2  # confirms the "insert failed -> look up again" path
    assert second.document_id == first.document_id
    assert second.created is False
    assert _count(db_session, Document) == 1


class _RecordingChunker:
    """Fake chunker: records what it was given and returns fixed chunks.

    Its purpose is to make "the service hands the raw content to the chunker" assertable.
    """

    def __init__(self, chunks: list[TextChunk]) -> None:
        self._chunks = chunks
        self.received: list[str] = []

    def chunk(self, text: str) -> list[TextChunk]:
        self.received.append(text)
        return self._chunks


def test_ingest_uses_injected_chunker_and_persists_its_output(db_session: Session) -> None:
    chunker = _RecordingChunker(
        [
            TextChunk(index=0, text="alpha beta", token_count=2),
            TextChunk(index=1, text="gamma delta", token_count=2),
        ]
    )
    service = DocumentIngestionService(db_session, chunker=chunker)

    result = service.ingest(title="Fixed", source="test", content=PARAGRAPH)

    # The chunker receives the raw content, not something polluted by the title
    assert chunker.received == [PARAGRAPH]

    rows = db_session.scalars(
        select(Chunk)
        .where(Chunk.document_id == result.document_id)
        .order_by(Chunk.chunk_index)
    ).all()
    assert [(row.text, row.token_count) for row in rows] == [
        ("alpha beta", 2),
        ("gamma delta", 2),
    ]
