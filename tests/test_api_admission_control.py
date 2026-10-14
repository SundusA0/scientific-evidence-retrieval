"""Opt-in reranker admission control: reject overload without blocking BM25."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient

from sci_evidence import api
from sci_evidence.data import tiny_demo_dataset
from sci_evidence.serving import SearchOutput


class FakeEngine:
    def __init__(self, docs, **kwargs):
        self.docs = {doc.id: doc for doc in docs}
        self.model = object()
        self.modes = ["bm25", "rerank"]
        self.device_name = "cpu"
        self.reranker_model = "fake-model"
        self.batch_size = kwargs.get("batch_size", 16)
        self.started = Event()
        self.release = Event()
        self.block = False
        self.raise_error = False

    def search(self, query, *, method="bm25", top_k=5, candidate_k=50):
        if method == "rerank":
            if self.raise_error:
                raise ValueError("deliberate error")
            if self.block:
                self.started.set()
                if not self.release.wait(timeout=5):
                    raise RuntimeError("Timed out waiting for test release")
        return SearchOutput(method=method, hits=[], elapsed_ms=1.0,
                            candidate_k=candidate_k if method == "rerank" else None)


@pytest.fixture
def fake_api(monkeypatch):
    monkeypatch.setattr(api, "load_scifact", lambda *a, **kw: tiny_demo_dataset())
    monkeypatch.setattr(api, "SearchEngine", FakeEngine)


def test_capacity_rejects_invalid_settings(monkeypatch):
    for value in ("invalid", "-1", "1.5"):
        monkeypatch.setenv("SCI_EVIDENCE_MAX_INFLIGHT_RERANK", value)
        with pytest.raises(RuntimeError, match="SCI_EVIDENCE_MAX_INFLIGHT_RERANK"):
            api.configured_rerank_capacity()


def test_capacity_disabled_by_default(monkeypatch, fake_api):
    monkeypatch.delenv("SCI_EVIDENCE_MAX_INFLIGHT_RERANK", raising=False)
    with TestClient(api.app) as client:
        assert client.get("/health").json()["max_inflight_rerank"] == 0
        assert client.post("/search", json={"query": "coffee", "method": "rerank"}).status_code == 200


def test_overload_rejected_and_bm25_unaffected(monkeypatch, fake_api):
    monkeypatch.setenv("SCI_EVIDENCE_MAX_INFLIGHT_RERANK", "1")
    with TestClient(api.app) as client:
        assert client.get("/health").json()["max_inflight_rerank"] == 1
        engine = api.app.state.engine
        engine.block = True
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                first = executor.submit(client.post, "/search", json={"query": "coffee", "method": "rerank"})
                assert engine.started.wait(timeout=3), "first inference did not start"
                rejected = client.post("/search", json={"query": "solar", "method": "rerank"})
                assert rejected.status_code == 503
                assert rejected.headers["Retry-After"] == "1"
                assert client.post("/search", json={"query": "solar", "method": "bm25"}).status_code == 200
                engine.release.set()
                assert first.result(timeout=3).status_code == 200
            assert client.post("/search", json={"query": "solar", "method": "rerank"}).status_code == 200
        finally:
            engine.release.set()


def test_error_releases_slot(monkeypatch, fake_api):
    monkeypatch.setenv("SCI_EVIDENCE_MAX_INFLIGHT_RERANK", "1")
    with TestClient(api.app) as client:
        engine = api.app.state.engine
        engine.raise_error = True
        err = client.post("/search", json={"query": "solar", "method": "rerank"})
        assert err.status_code == 400
        engine.raise_error = False
        ok = client.post("/search", json={"query": "solar", "method": "rerank"})
        assert ok.status_code == 200
