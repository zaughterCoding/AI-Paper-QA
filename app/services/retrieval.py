"""Retrieve ranked passages without changing the final context budget."""

from typing import get_args

from sqlalchemy.orm import Session

from app.core.config import RetrievalMode
from app.rag.embeddings import EmbeddingClient
from app.rag.ranking import Reranker, get_reranker, reciprocal_rank_fusion
from app.repositories.chunks import ChunkRepository, RetrievedChunk

DEFAULT_TOP_K = 5
MAX_TOP_K = 20
MAX_CANDIDATES = 200


class RetrievalService:
    def __init__(self, session: Session, embedding_client: EmbeddingClient, *,
                 mode: RetrievalMode = "dense", candidate_k: int = 30,
                 reranker: Reranker | None = None) -> None:
        if mode not in get_args(RetrievalMode):
            raise ValueError(f"Unknown retrieval mode: {mode}")
        if not 1 <= candidate_k <= MAX_CANDIDATES:
            raise ValueError(f"candidate_k must be between 1 and {MAX_CANDIDATES}")
        self.chunks = ChunkRepository(session)
        self.embedding_client = embedding_client
        self.mode = mode
        self.candidate_k = candidate_k
        self.reranker = reranker

    def retrieve_candidates(self, question: str, limit: int) -> list[RetrievedChunk]:
        """Return the bounded candidate pool before reranking."""
        if not question.strip():
            raise ValueError("question must not be empty")
        if not 1 <= limit <= MAX_CANDIDATES:
            raise ValueError(f"limit must be between 1 and {MAX_CANDIDATES}")
        if self.mode == "fts":
            return self.chunks.search_lexical(question, limit)
        vector = self.embedding_client.embed_text(question)
        dense = self.chunks.search_similar(query_embedding=vector, top_k=limit)
        if self.mode.startswith("hybrid"):
            lexical = self.chunks.search_lexical(question, limit)
            return reciprocal_rank_fusion([dense, lexical])[:limit]
        return dense

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K) -> list[RetrievedChunk]:
        if not question.strip():
            raise ValueError("question must not be empty")
        if not 1 <= top_k <= MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}, got {top_k}")
        limit = max(self.candidate_k, top_k) if self.mode not in {"dense", "fts"} else top_k
        candidates = self.retrieve_candidates(question, limit)
        if self.mode.endswith("_rerank") and candidates:
            reranker = self.reranker if self.reranker is not None else get_reranker()
            return reranker.rerank(question, candidates, top_k)
        return candidates[:top_k]
