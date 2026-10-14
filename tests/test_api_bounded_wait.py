"""Bounded queue policy for reranking under contention."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
import time

import pytest
from fastapi.testclient import TestClient

from sci_evidence import api
from sci_evidence.data import tiny_demo_dataset
from sci_evidence.serving import SearchOutput


class BlockingEngine:
    def __init__(self, docs, **kwargs):
        self.docs = {d.id: d for d in docs}
        self.model = object()
        self.modes = ["bm25", "rerank"]
        self.device_name = "cpu"
        self.reranker_model = "dummy"
        self.batch_size = 32
        self.started = Event()
        self.release = Event()

    def search(self, query, *, method="bm25", top_k=5, candidate_k=50):
        if method == "rerank" and query == "hold":
            self.started.set()
            if not self.release.wait(timeout=5):
                raise RuntimeError("Timeout waiting in fake model")
        return SearchOutput(method=method, hits=[], elapsed_ms=5,
                            candidate_k=candidate_k if method == "rerank" else None)


@pytest.fixture
def fake_api(monkeypatch):
    monkeypatch.setattr(api, "load_scifact", lambda *a, **kw: tiny_demo_dataset())
    monkeypatch.setattr(api, "SearchEngine", BlockingEngine)
    monkeypatch.setenv("SCI_EVIDENCE_MAX_INFLIGHT_RERANK", "1")
    monkeypatch.setenv("SCI_EVIDENCE_RERANK_WAIT_MS", "1000")
    monkeypatch.setenv("SCI_EVIDENCE_MAX_QUEUED_RERANK", "1")


def test_wait_configuration_validation(monkeypatch):
    for var, value in (("SCI_EVIDENCE_RERANK_WAIT_MS", "bad"),
                       ("SCI_EVIDENCE_RERANK_WAIT_MS", "-1"),
                       ("SCI_EVIDENCE_RERANK_WAIT_MS", "5001"),
                       ("SCI_EVIDENCE_MAX_QUEUED_RERANK", "-1"),
                       ("SCI_EVIDENCE_MAX_QUEUED_RERANK", "1001")):
        monkeypatch.delenv("SCI_EVIDENCE_RERANK_WAIT_MS", raising=False)
        monkeypatch.delenv("SCI_EVIDENCE_MAX_QUEUED_RERANK", raising=False)
        monkeypatch.setenv(var, value)
        with pytest.raises(RuntimeError):
            api.configured_rerank_wait()


def test_queue_wait_and_full_response(fake_api):
    with TestClient(api.app) as client:
        engine = api.app.state.engine
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(client.post, "/search", json={"query": "hold", "method": "rerank"})
            assert engine.started.wait(timeout=2)
            second = executor.submit(client.post, "/search", json={"query": "queued", "method": "rerank"})
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and client.get('/health').json()['waiting_rerank'] != 1:
                time.sleep(0.01)
            assert client.get('/health').json()['waiting_rerank'] == 1
            assert client.post('/search', json={"query":"overflow", "method":"rerank"}).status_code == 503
            assert client.post('/search', json={"query":"solar", "method":"bm25"}).status_code == 200
            engine.release.set()
            assert first.result(timeout=2).status_code == 200
            assert second.result(timeout=2).status_code == 200
        assert client.get('/health').json()['waiting_rerank'] == 0


def test_queue_timeout_releases_waiter(fake_api, monkeypatch):
    monkeypatch.setenv("SCI_EVIDENCE_RERANK_WAIT_MS", "25")
    with TestClient(api.app) as client:
        engine = api.app.state.engine
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                first = executor.submit(client.post, '/search', json={"query":"hold", "method":"rerank"})
                assert engine.started.wait(timeout=2)
                timed_out = client.post('/search', json={"query":"queued", "method":"rerank"})
                assert timed_out.status_code == 503
                assert timed_out.headers['Retry-After'] == '1'
                assert client.get('/health').json()['waiting_rerank'] == 0
                engine.release.set()
                assert first.result(timeout=2).status_code == 200
            assert client.post('/search', json={"query":"new", "method":"rerank"}).status_code == 200
        finally:
            engine.release.set()


def test_zero_wait_remains_immediate(monkeypatch, fake_api):
    monkeypatch.setenv("SCI_EVIDENCE_RERANK_WAIT_MS", "0")
    with TestClient(api.app) as client:
        engine = api.app.state.engine
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                first = executor.submit(client.post, '/search', json={"query":"hold", "method":"rerank"})
                assert engine.started.wait(timeout=2)
                overloaded = client.post('/search', json={"query":"other", "method":"rerank"})
                assert overloaded.status_code == 503
                engine.release.set()
                assert first.result(timeout=2).status_code == 200
        finally:
            engine.release.set()
