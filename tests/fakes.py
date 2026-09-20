"""Test doubles: no model downloads, no LLM calls, no network, no unpredictable slowness.

Kept in one module so the doubles cannot be copied around and drift apart into versions
that no longer represent the real components.
"""

import math
import zlib

from app.core.config import EMBEDDING_DIM
from app.repositories.chunks import RetrievedChunk
from app.services.retrieval import DEFAULT_TOP_K


class FakeEmbeddingClient:
    """Deterministic fake embeddings, matching the real client's shape and normalization.

    A bag-of-words hash: each word is hashed onto one of 384 dimensions and accumulated,
    so texts sharing vocabulary end up more similar and retrieval tests can verify the
    ranking logic instead of comparing unrelated random numbers.

    What it cannot imitate is meaning: it does not know cat and kitten are related, only
    that they share no words. It can therefore validate the retrieval pipeline, not
    retrieval quality.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim

        for word in text.lower().split():
            # crc32, not the builtin hash(): the builtin is salted per process
            # (PYTHONHASHSEED), so the same word would land on a different dimension
            # in each run and the tests would stop being deterministic.
            bucket = zlib.crc32(word.encode("utf-8")) % self.dim
            vector[bucket] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector  # empty text / unknown characters -> all-zero vector

        return [value / norm for value in vector]


class FakeLLMClient:
    """Returns a canned answer and records every prompt it was given.

    Recording the prompts is what lets a test assert on the prompt without a separate spy,
    and asserting on the prompt is the only way to test this layer: the model's output is
    a string, so almost every mistake in prompt construction is invisible in the result.
    """

    def __init__(self, answer: str = "A canned answer [1].") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


class FakeQALogRepository:
    """Records the audit rows that would have been written, and can be made to fail.

    ``error`` stands in for a database that cannot be written to: it is raised from
    ``create_log`` exactly where a real ``SQLAlchemyError`` would be, so the service's
    handling of a failed audit write is testable without breaking a real database.
    """

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.logs: list[dict] = []

    def create_log(
        self, question: str, answer: str, retrieved_chunk_ids: list[str], latency_ms: int
    ) -> None:
        if self.error is not None:
            raise self.error
        self.logs.append(
            {
                "question": question,
                "answer": answer,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "latency_ms": latency_ms,
            }
        )


class FakeRetrievalService:
    """Returns a fixed list of chunks instead of querying a database."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self.chunks = chunks if chunks is not None else []
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        self.calls.append((question, top_k))
        return list(self.chunks)
