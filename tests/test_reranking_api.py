"""API integration with the configured reranking pipeline."""

from dataclasses import replace
from types import SimpleNamespace

from tests.fakes import FakeEmbeddingClient, FakeLLMClient
from tests.test_api_ask import app_client, import_document


def test_ask_uses_configured_reranker_and_identifies_its_scores(db_session, monkeypatch):
    calls = []

    def rerank(question, candidates, k):
        calls.append((question, len(candidates), k))
        return [replace(c, score=-2.5, score_type="cross_encoder") for c in reversed(candidates)][:k]

    monkeypatch.setattr("app.services.retrieval.get_reranker", lambda: SimpleNamespace(rerank=rerank))
    with app_client(db_session, FakeEmbeddingClient(), FakeLLMClient(), "hybrid_rerank") as client:
        import_document(client)
        response = client.post("/ask", json={"question": "token10", "top_k": 2})
    assert response.status_code == 200
    assert calls == [("token10", 3, 2)]
    assert len(response.json()["sources"]) == 2
    assert all(s["score_type"] == "cross_encoder" and s["score"] == -2.5 for s in response.json()["sources"])
