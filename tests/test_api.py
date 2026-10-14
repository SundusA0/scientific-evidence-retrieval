from fastapi.testclient import TestClient
from sci_evidence import api
from sci_evidence.data import tiny_demo_dataset


def test_local_api_search_and_home(monkeypatch):
    monkeypatch.setattr(api, "load_scifact", lambda *a, **kw: tiny_demo_dataset())
    monkeypatch.delenv("SCI_EVIDENCE_RERANKER", raising=False)
    with TestClient(api.app) as client:
        home = client.get("/")
        assert home.status_code == 200
        assert "retrieval lab" in home.text
        status = client.get("/health").json()
        assert status["methods"] == ["bm25"]
        assert status["documents"] == 5
        assert api.app.state.engine.batch_size == 32
        response = client.post("/search", json={"query": "solar panels", "method": "bm25", "top_k": 2})
        assert response.status_code == 200
        result = response.json()
        assert result["results"][0]["id"] == "d1"
        assert result["elapsed_ms"] >= 0
        unavailable = client.post("/search", json={"query": "solar", "method": "rerank"})
        assert unavailable.status_code == 400
        empty = client.post("/search", json={"query": "", "top_k": 1})
        assert empty.status_code == 422


def test_reranker_api_uses_20_candidates(monkeypatch):
    from sci_evidence.serving import SearchOutput

    monkeypatch.setattr(
        api, "load_scifact",
        lambda *args, **kwargs: tiny_demo_dataset()
    )
    monkeypatch.delenv("SCI_EVIDENCE_RERANKER", raising=False)

    observed = {}

    def fake_search(self, query, *, method="bm25",
                    top_k=5, candidate_k=50):
        observed["candidate_k"] = candidate_k
        return SearchOutput(
            method=method, hits=[], elapsed_ms=1.0,
            candidate_k=candidate_k
        )

    monkeypatch.setattr(api.SearchEngine, "search", fake_search)

    with TestClient(api.app) as client:
        response = client.post(
            "/search",
            json={"query": "caffeine", "method": "rerank", "top_k": 5}
        )

    assert response.status_code == 200
    assert observed["candidate_k"] == 20
    assert response.json()["candidate_k"] == 20
