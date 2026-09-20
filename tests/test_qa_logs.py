"""QA log repository tests, against the real database.

The column types are the point of this file. Everything else about the audit row is
Python; ``retrieved_chunk_ids`` is a Postgres array and ``created_at`` is a timestamptz, and
neither of those can be checked against a fake. An empty array in particular has to be
stored and read back as ``[]`` rather than as NULL, because "retrieval found nothing" is
one of the outcomes the table exists to count.

What a failing audit write costs the caller is decided in ``AnswerService`` and tested in
``test_answering.py``; what is tested here is the other half of that decision -- that a
failure leaves the session usable rather than poisoned, which is what makes catching it
and carrying on actually safe.
"""

import uuid
from datetime import timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.tables import QALog
from app.repositories.qa_logs import QALogRepository


@pytest.fixture
def qa_logs(db_session: Session) -> QALogRepository:
    return QALogRepository(db_session)


def count_logs(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(QALog)) or 0


def test_create_log_returns_a_row_with_its_generated_fields(qa_logs: QALogRepository) -> None:
    """The id and the timestamp are filled in by the database, so they are only readable
    if the insert actually went out."""
    log = qa_logs.create_log(
        question="what is self attention?", answer="It relates positions [1].",
        retrieved_chunk_ids=[], latency_ms=42,
    )

    assert isinstance(log.id, uuid.UUID)
    assert log.created_at is not None


def test_every_field_round_trips(qa_logs: QALogRepository, db_session: Session) -> None:
    chunk_ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    created = qa_logs.create_log(
        question="how many GPUs?", answer="Eight [1].", retrieved_chunk_ids=chunk_ids,
        latency_ms=1234,
    )
    # Re-read from the database rather than asserting on the object that was passed in:
    # the object proves the values went in, a fresh SELECT proves what came back out, and
    # a column type that mangles its input would pass the first check and fail this one.
    stored = db_session.scalar(select(QALog).where(QALog.id == created.id))

    assert stored.question == "how many GPUs?"
    assert stored.answer == "Eight [1]."
    assert stored.retrieved_chunk_ids == chunk_ids
    assert stored.latency_ms == 1234


def test_created_at_is_timezone_aware(qa_logs: QALogRepository) -> None:
    """A naive timestamp would mean a different instant depending on the server's locale,
    and rows written by two machines would not be comparable."""
    log = qa_logs.create_log(question="q", answer="a", retrieved_chunk_ids=[], latency_ms=1)

    assert log.created_at.tzinfo is not None
    assert log.created_at.utcoffset() == timezone.utc.utcoffset(None)


def test_an_empty_chunk_id_list_is_stored_as_an_empty_array(
    qa_logs: QALogRepository, db_session: Session
) -> None:
    """The column is NOT NULL, so NULL is not an option -- but ``[]`` and NULL would still
    be easy to confuse on the way out. "Retrieval returned nothing" must survive as a value
    that can be counted, because that count is the retrieval miss rate."""
    log = qa_logs.create_log(
        question="unanswerable?", answer="Not enough information.", retrieved_chunk_ids=[],
        latency_ms=3,
    )

    stored = db_session.scalar(select(QALog).where(QALog.id == log.id))

    assert stored.retrieved_chunk_ids == []
    assert stored.retrieved_chunk_ids is not None


def test_chunk_ids_are_readable_back_as_text(qa_logs: QALogRepository, db_session: Session) -> None:
    """They are stored as text so they can later be matched against ``chunks.id``, which a
    query would do by casting one side; this pins the format that cast depends on."""
    chunk_id = uuid.uuid4()

    log = qa_logs.create_log(
        question="q", answer="a", retrieved_chunk_ids=[str(chunk_id)], latency_ms=5
    )

    assert db_session.scalar(select(QALog.retrieved_chunk_ids).where(QALog.id == log.id)) == [
        str(chunk_id)
    ]


def test_each_call_writes_its_own_row(qa_logs: QALogRepository, db_session: Session) -> None:
    """Two questions are two rows. A shared or reused row would make the table useless for
    counting anything."""
    qa_logs.create_log(question="first", answer="a", retrieved_chunk_ids=[], latency_ms=1)
    qa_logs.create_log(question="second", answer="b", retrieved_chunk_ids=[], latency_ms=2)

    assert count_logs(db_session) == 2


def test_a_failed_insert_leaves_the_session_usable(
    qa_logs: QALogRepository, db_session: Session
) -> None:
    """The repository rolls back before re-raising, and this is what that buys.

    A statement that fails leaves the session unable to run another one until it has been
    rolled back; every later statement on it raises PendingRollbackError. The caller that
    catches the failure -- the answer service -- expects to carry on serving the request,
    so a poisoned session would turn one unwritable log into a broken response.

    The failure is provoked through the database rather than by stubbing ``commit``: a
    pending row that violates NOT NULL makes the real INSERT fail, so the exception is the
    real IntegrityError and the session really is in the state this is about.
    """
    db_session.add(QALog(question="q", answer=None, retrieved_chunk_ids=[], latency_ms=1))

    with pytest.raises(IntegrityError):
        qa_logs.create_log(question="q", answer="a", retrieved_chunk_ids=[], latency_ms=1)

    # No "and now it works" caveat: if the rollback were removed, this line would raise
    # PendingRollbackError rather than returning a number.
    assert count_logs(db_session) == 0
