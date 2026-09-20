"""Writes for the ``qa_logs`` table.

One row per question the service answered: what was asked, what was answered, which chunks
the answer was allowed to read, and how long it took.

This is an audit trail rather than application state. Nothing reads it back to decide
anything, which is why the only method here writes -- there is no query that the
application itself needs. A debugging or evaluation pass reads the table directly.
"""

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models.tables import QALog


class QALogRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_log(
        self, question: str, answer: str, retrieved_chunk_ids: list[str], latency_ms: int
    ) -> QALog:
        """Insert one audit row and commit it.

        The other repositories hold the session and leave committing to the service, on the
        grounds that only the service knows which writes have to land together. That
        reasoning is what makes this method different rather than inconsistent: there is
        exactly one write here, so there is no boundary for a caller to place. Inserting and
        committing in one call is also what lets a caller treat "record this exchange" as a
        single operation it can catch and ignore -- which is what the answer service needs,
        since it must not lose an already generated answer to a bookkeeping failure.

        ``retrieved_chunk_ids`` is a list of strings because the column is ``ARRAY(String)``,
        and because what is being recorded is history: these are the chunk rows the model
        was shown on this request, and they stop resolving the moment the document is
        re-imported and gets new ones. The API omits ``chunk_id`` for the opposite reason --
        it is of no use to a client -- while here it is the whole point.

        Raises whatever the database raises; the caller decides what a failed audit write
        should cost.
        """
        log = QALog(
            question=question,
            answer=answer,
            retrieved_chunk_ids=retrieved_chunk_ids,
            latency_ms=latency_ms,
        )
        self.session.add(log)

        try:
            # commit() flushes first, so no separate flush is needed: the INSERT goes out
            # here, and the id and created_at defaults are readable on the returned object
            # once it succeeds.
            self.session.commit()
        except SQLAlchemyError:
            # A failed statement leaves the session unable to run another one until it has
            # been rolled back; every subsequent statement raises PendingRollbackError.
            # Rolling back here is what keeps one unwritable log from breaking the rest of
            # a request that caught the failure and carried on.
            self.session.rollback()
            raise

        return log
