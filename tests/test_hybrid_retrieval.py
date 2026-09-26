"""Hybrid retrieval, fusion and reranking contracts."""

import importlib.util
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import time
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from app.repositories.chunks import ChunkRepository, RetrievedChunk
from app.services.retrieval import RetrievalService
from tests.fakes import FakeEmbeddingClient
from tests.test_retrieval import add_chunk, make_document


def ranking():
    assert importlib.util.find_spec("app.rag.ranking") is not None, "ranking is not implemented"
    from app.rag import ranking as module
    return module


def chunk(index, score=0.5):
    return RetrievedChunk(uuid.UUID(int=index + 1), uuid.UUID(int=1), "Paper", index, f"evidence {index}", score)


def test_rrf_uses_ranks_and_counts_each_chunk_once_per_list():
    fuse = ranking().reciprocal_rank_fusion
    a, b, c = chunk(0, 1000), chunk(1, -50), chunk(2, 1)
    result = fuse([[a, b, a], [b, c]])
    assert [r.chunk_id for r in result] == [b.chunk_id, a.chunk_id, c.chunk_id]
    assert result[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert result[1].score == pytest.approx(1 / 61)
    assert result[0].score_type == "rrf"
    assert a.score == 1000


def test_reranker_scores_query_title_and_passage_then_limits_results():
    module = ranking()
    reranker = module.Reranker.__new__(module.Reranker)
    calls = []
    def predict(pairs, **kwargs):
        calls.append(pairs)
        return np.array([-2.0, 3.0])
    reranker.model = SimpleNamespace(predict=predict)
    result = reranker.rerank("question", [chunk(0), chunk(1)], 1)
    assert result[0].chunk_index == 1
    assert result[0].score == 3.0
    assert result[0].score_type == "cross_encoder"
    assert calls == [[("question", "Paper\nevidence 0"), ("question", "Paper\nevidence 1")]]
    assert reranker.rerank("question", [], 1) == []
    assert len(calls) == 1


@pytest.mark.parametrize("scores", [[float("nan"), 1], [1], [[1, 2], [3, 4]]])
def test_reranker_rejects_invalid_model_output(scores):
    module = ranking()
    reranker = module.Reranker.__new__(module.Reranker)
    reranker.model = SimpleNamespace(predict=lambda *a, **k: np.array(scores))
    with pytest.raises(RuntimeError):
        reranker.rerank("question", [chunk(0), chunk(1)], 2)


def test_concurrent_first_requests_share_one_reranker(monkeypatch):
    module = ranking()
    start = Barrier(4)
    constructed = []

    def construct(*args):
        time.sleep(0.02)
        instance = object()
        constructed.append(instance)
        return instance

    def load(_):
        start.wait(timeout=5)
        return module.get_reranker()

    monkeypatch.setattr(module, "Reranker", construct)
    module._load_reranker.cache_clear()
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(load, range(4)))
        assert len(constructed) == 1
        assert all(result is constructed[0] for result in results)
    finally:
        module._load_reranker.cache_clear()


def test_fts_matches_any_non_stopword_and_excludes_unindexed_chunks(db_session):
    repo = ChunkRepository(db_session)
    assert hasattr(repo, "search_lexical"), "lexical search is not implemented"
    embedder = FakeEmbeddingClient()
    doc = make_document(db_session, title="Optimizer Study")
    add_chunk(db_session, doc, 0, "AdamW uses a learning rate of 2e-5.", embedder.embed_text("AdamW"))
    add_chunk(db_session, doc, 1, "Unrelated botanical observations.", embedder.embed_text("botany"))
    add_chunk(db_session, doc, 2, "AdamW AdamW AdamW", None)
    results = repo.search_lexical("Which learning rate does AdamW use?", 5)
    assert [r.chunk_index for r in results] == [0]
    assert results[0].score_type == "fts"
    assert repo.search_lexical("the and of", 5) == []
    assert repo.search_lexical("?!", 5) == []
    assert repo.search_lexical("'; DROP TABLE chunks; --", 5) == []
    with pytest.raises(ValueError): repo.search_lexical("AdamW", 0)


def test_hybrid_deduplicates_candidates_before_reranking(db_session):
    embedder = FakeEmbeddingClient()
    calls = []
    reranker = SimpleNamespace(rerank=lambda q, candidates, k: calls.append(candidates) or candidates[:k])
    service = RetrievalService(db_session, embedder, mode="hybrid_rerank", candidate_k=3, reranker=reranker)
    a, b, c = chunk(0), chunk(1), chunk(2)
    service.chunks = SimpleNamespace(search_similar=lambda **kw: [a, b], search_lexical=lambda *args: [b, c])
    results = service.retrieve("question", 2)
    assert len(results) == 2
    assert len(calls[0]) == 3
    assert len({r.chunk_id for r in calls[0]}) == 3
    assert calls[0][0].chunk_id == b.chunk_id


def test_fts_does_not_encode_and_empty_candidates_do_not_load_reranker(db_session, monkeypatch):
    embedder = SimpleNamespace(embed_text=lambda q: pytest.fail("FTS must not embed"))
    service = RetrievalService(db_session, embedder, mode="fts")
    assert service.retrieve("unmatched", 5) == []
    hybrid = RetrievalService(db_session, FakeEmbeddingClient(), mode="hybrid_rerank")
    monkeypatch.setattr("app.services.retrieval.get_reranker", lambda: pytest.fail("Empty corpus must not load model"))
    assert hybrid.retrieve("unmatched", 5) == []


@pytest.mark.parametrize("kwargs", [{"mode": "typo"}, {"candidate_k": 0}, {"candidate_k": 201}])
def test_retrieval_rejects_invalid_configuration(db_session, kwargs):
    with pytest.raises(ValueError): RetrievalService(db_session, FakeEmbeddingClient(), **kwargs)
