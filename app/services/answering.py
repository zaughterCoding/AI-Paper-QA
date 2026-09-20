"""Answer service: a question in, a cited answer out.

This is the "G" in RAG and the only place where retrieval and generation meet:

    question --RetrievalService--> chunks --prompt--> LLMClient --> answer

What this layer owns is the prompt: which chunks reach the model, in what order, and how
they are labelled so the answer can cite them.

There is no separate context-truncation step. ``top_k`` is the context budget and
``MAX_TOP_K`` bounds it, so the prompt has a known upper size rather than growing with
whatever the corpus happens to contain.
"""

from dataclasses import dataclass

from app.rag.llm import LLMClient
from app.repositories.chunks import RetrievedChunk
from app.services.retrieval import DEFAULT_TOP_K, RetrievalService

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
    """Answer a question from the corpus.

    Both collaborators are injected instead of built here: the retrieval service so the
    application's wiring is decided by the caller, and the LLM client so the same
    process-wide instance is reused rather than one connection pool per request.
    """

    def __init__(self, retrieval: RetrievalService, llm_client: LLMClient) -> None:
        self.retrieval = retrieval
        self.llm_client = llm_client

    def answer(self, question: str, top_k: int = DEFAULT_TOP_K) -> AnswerResult:
        """Retrieve, then generate. Reads only, and never commits.

        Validation lives entirely in ``RetrievalService.retrieve`` and happens first, so an
        empty question or an out-of-range ``top_k`` raises before the model is asked
        anything: a bad request never costs a completion.
        """
        chunks = self.retrieval.retrieve(question, top_k)

        if not chunks:
            return AnswerResult(answer=NO_SOURCES_ANSWER, sources=[])

        # strip(): surrounding newlines are noise for a caller rendering the answer, and an
        # all-whitespace answer should read as empty rather than as content.
        answer = self.llm_client.generate(build_prompt(question, chunks)).strip()
        return AnswerResult(answer=answer, sources=chunks)
