"""Rank fusion and local cross-encoder reranking."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from functools import lru_cache
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import CrossEncoder
from torch import nn

from app.core.config import get_settings

if TYPE_CHECKING:
    from app.repositories.chunks import RetrievedChunk


def reciprocal_rank_fusion(rankings: list[list[RetrievedChunk]]) -> list[RetrievedChunk]:
    scores = defaultdict(float)
    chunks = {}
    for ranking in rankings:
        seen = set()
        for rank, chunk in enumerate(ranking, 1):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks.setdefault(chunk.chunk_id, chunk)
            scores[chunk.chunk_id] += 1 / (60 + rank)
    ordered = sorted(chunks.values(), key=lambda c: (-scores[c.chunk_id], c.title, c.chunk_index, str(c.chunk_id)))
    return [replace(c, score=scores[c.chunk_id], score_type="rrf") for c in ordered]


class Reranker:
    def __init__(self, model_name: str, revision: str) -> None:
        self.model = CrossEncoder(model_name, revision=revision, max_length=512,
                                  device="cpu", trust_remote_code=False)

    def rerank(self, question: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not chunks:
            return []
        pairs = [(question, f"{chunk.title}\n{chunk.text}") for chunk in chunks]
        scores = np.asarray(self.model.predict(pairs, batch_size=16, show_progress_bar=False,
                                              activation_fn=nn.Identity()))
        if scores.shape != (len(chunks),) or not np.isfinite(scores).all():
            raise RuntimeError("Reranker returned invalid scores")
        ranked = sorted(zip(chunks, scores), key=lambda pair: -float(pair[1]))
        return [replace(c, score=float(s), score_type="cross_encoder") for c, s in ranked[:top_k]]


@lru_cache(maxsize=1)
def _load_reranker() -> Reranker:
    settings = get_settings()
    return Reranker(settings.reranker_model, settings.reranker_revision)


_load_lock = Lock()


def get_reranker() -> Reranker:
    # Serialize cache misses across concurrent first requests.
    with _load_lock:
        return _load_reranker()
