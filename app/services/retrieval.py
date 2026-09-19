"""Retrieval service: turn a question into the most relevant chunks.

This is the "R" in RAG and the only place where the two halves meet:

    question (text) --EmbeddingClient--> query vector --ChunkRepository--> chunks

``retrieve()`` is deliberately thin: it orchestrates and does not implement. Text to
vector is ``EmbeddingClient``, vector to chunks is
``ChunkRepository.search_similar``. What this layer owns is validation and ordering --
which inputs are legal, whether validation happens before encoding, and how many times
each collaborator is called.

There is deliberately no similarity threshold here. A question outside the corpus still
returns its top_k chunks, and judging whether those chunks actually answer it is left to
the LLM that reads them.
"""

from sqlalchemy.orm import Session

from app.rag.embeddings import EmbeddingClient
from app.repositories.chunks import ChunkRepository, RetrievedChunk

# How many chunks to return when the caller does not say.
#
# Five is a starting point rather than a derived value: too few and the passage backing
# an answer may sit at rank six, too many and irrelevant text is pushed into the prompt,
# costing tokens and diluting the signal. The evaluation set is what should settle it.
DEFAULT_TOP_K = 5

# Upper bound on top_k. Anything above it is rejected rather than quietly clamped.
#
# top_k is not merely "how many rows to read": it decides how much context reaches the
# LLM, and a caller asking for 1000 would get a prompt that is long, expensive and mostly
# noise rather than a more thorough search. Twenty leaves room to raise it while
# investigating which rank holds an answer, without allowing real damage.
MAX_TOP_K = 20


class RetrievalService:
    """Retrieve a set of chunks for a question.

    ``embedding_client`` is injected rather than built here. Under a web request the
    caller must pass the process-wide instance from
    ``app.rag.embeddings.get_embedding_client()``; constructing a new one would load
    another copy of the model and exhaust memory once a few requests overlap. Leaving the
    choice to the caller keeps this service unaware that only one model exists per
    process, which is a matter of application wiring. All it needs is something with
    ``embed_text()``, which is also what lets tests pass a fake.
    """

    def __init__(self, session: Session, embedding_client: EmbeddingClient) -> None:
        self.chunks = ChunkRepository(session)
        self.embedding_client = embedding_client

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        """The ``top_k`` chunks most similar to ``question``, best first.

        An empty list is a normal result -- the database is empty, or no chunk has a
        vector yet -- and not an error. "Nothing found" and "something failed" are
        different outcomes.

        This method reads only and never commits.
        """
        # Validated before encoding, because encoding an empty question is pure waste and
        # some models return an all-zero vector for it. A zero vector has no defined
        # cosine distance, and pgvector would then rank the results arbitrarily.
        #
        # strip() rather than a bare truth test: a whitespace-only question is just as
        # meaningless, but the string itself is non-empty.
        if not question.strip():
            raise ValueError("question must not be empty")

        if top_k < 1 or top_k > MAX_TOP_K:
            # The message carries the received value so the API layer can turn this into a
            # 400 that points at the argument rather than at the retrieval logic.
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}")

        # The original question is encoded, not the stripped version, so what reaches the
        # model is exactly what the caller supplied and any problem can be reproduced with
        # the same string.
        query_embedding = self.embedding_client.embed_text(question)

        return self.chunks.search_similar(query_embedding=query_embedding, top_k=top_k)
