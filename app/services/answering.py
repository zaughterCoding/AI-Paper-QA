"""Answer service: a question in, a cited answer out.

This is the "G" in RAG and the only place where retrieval and generation meet:

    question --RetrievalService--> chunks --prompt--> LLMClient --> answer
                                                                       |
                                                         QALogRepository (audit row)

What this layer owns is the prompt: which chunks reach the model, in what order, and how
they are labelled so the answer can cite them.

There is no separate context-truncation step. ``top_k`` is the context budget and
``MAX_TOP_K`` bounds it, so the prompt has a known upper size rather than growing with
whatever the corpus happens to contain.

Recording the exchange is the one write here, and it is deliberately the last thing that
happens and deliberately unable to fail the request. See ``_record``.
"""

import logging
import time
from dataclasses import dataclass

from sqlalchemy.exc import SQLAlchemyError

from app.rag.llm import LLMClient
from app.repositories.chunks import RetrievedChunk
from app.repositories.qa_logs import QALogRepository
from app.services.retrieval import DEFAULT_TOP_K, RetrievalService

logger = logging.getLogger(__name__)

# Returned when retrieval finds nothing at all.
#
# Produced without calling the model. With no sources there is nothing for the answer to
# be conditioned on, so the outcome is known before the request; sending it anyway would
# add latency and give the model a chance to answer from its own memory instead.
NO_SOURCES_ANSWER = (
    "The sources do not contain enough information to answer this question."
)

PROMPT_TEMPLATE = """You are an assistant answering questions using only the provided sources.
If the sources do not contain enough information, say that the sources are insufficient.

Question:
{question}

Sources:
{sources}

Answer with a concise explanation and cite source numbers like [1], [2].
"""


@dataclass(frozen=True)
class AnswerResult:
    """An answer together with the chunks it was allowed to use.

    ``sources`` is carried out of the service so a caller can resolve the ``[n]`` markers
    in the text without querying again, and so a client can show what the answer rests on.
    The order is the order the model was given, so ``[1]`` is ``sources[0]``.
    """

    answer: str
    sources: list[RetrievedChunk]


def format_sources(chunks: list[RetrievedChunk]) -> str:
    """Number ``chunks`` as ``[1]``.., in the order the model will read them.

    The number rather than the title is what gets cited: one paper contributes several
    chunks, so titles repeat, while an index is unambiguous and short enough for the model
    to copy reliably.
    """
    return "\n\n".join(
        f"[{position}] {chunk.title}\n{chunk.text}"
        for position, chunk in enumerate(chunks, start=1)
    )


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """The full prompt for one question and the chunks retrieved for it."""
    return PROMPT_TEMPLATE.format(question=question, sources=format_sources(chunks))


class AnswerService:
    """Answer a question from the corpus, and record that it was answered.

    All three collaborators are injected instead of built here: the retrieval service so
    the application's wiring is decided by the caller, the LLM client so the same
    process-wide instance is reused rather than one connection pool per request, and the
    audit repository so this class never has to hold a database session of its own. That
    last one is the reason the service can be tested without a database at all: everything
    it does except the final write is pure orchestration over whatever it was handed.
    """

    def __init__(
        self, retrieval: RetrievalService, llm_client: LLMClient, qa_logs: QALogRepository
    ) -> None:
        self.retrieval = retrieval
        self.llm_client = llm_client
        self.qa_logs = qa_logs

    def answer(self, question: str, top_k: int = DEFAULT_TOP_K) -> AnswerResult:
        """Retrieve, generate, then record.

        Validation lives entirely in ``RetrievalService.retrieve`` and happens first, so an
        empty question or an out-of-range ``top_k`` raises before the model is asked
        anything: a bad request never costs a completion. Nothing is recorded for a request
        that raised -- there is no answer to record, and a log row is about an answer.

        The timer starts before retrieval and stops before the audit write, so the latency
        it reports is the wait the caller experienced, not that plus the bookkeeping.
        """
        started = time.perf_counter()
        chunks = self.retrieval.retrieve(question, top_k)

        if not chunks:
            result = AnswerResult(answer=NO_SOURCES_ANSWER, sources=[])
        else:
            # strip(): surrounding newlines are noise for a caller rendering the answer, and
            # an all-whitespace answer should read as empty rather than as content.
            answer = self.llm_client.generate(build_prompt(question, chunks)).strip()
            result = AnswerResult(answer=answer, sources=chunks)

        self._record(question, result, started)
        return result

    def _record(self, question: str, result: AnswerResult, started: float) -> None:
        """Write the audit row for one answered question.

        A failing audit write does not fail the answer. By the time this runs the answer
        exists, and it is what the caller asked for; discarding it because a bookkeeping
        insert did not land would turn a working response into a 500 and, since nothing is
        cached, buy a second model call to get the same text back. Observability is
        supposed to be passive.

        The failure is not hidden either. It is reported at ERROR, and an ERROR record
        reaches stderr even in a process that never configured logging -- Python's handler
        of last resort -- so a table that cannot be written shows up in the log rather than
        nowhere. There is no separate metric or alert to add here.

        Only database failures are treated this way. Anything that is not a
        ``SQLAlchemyError`` means this code is wrong -- the wrong argument, the wrong type
        -- and that belongs in a 500, where it is loud and fixable, rather than being filed
        under "the database was busy". A bare ``except Exception`` would do exactly that
        misfiling.
        """
        latency_ms = round((time.perf_counter() - started) * 1000)

        try:
            self.qa_logs.create_log(
                question=question,
                answer=result.answer,
                retrieved_chunk_ids=[str(chunk.chunk_id) for chunk in result.sources],
                latency_ms=latency_ms,
            )
        except SQLAlchemyError:
            logger.exception("could not record the QA log for question %r", question)
