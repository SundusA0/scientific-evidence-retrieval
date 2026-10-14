"""Deterministic unit tests for the admission-aware load generator."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "admission_benchmark", ROOT / "scripts" / "12_benchmark_rerank_admission.py"
)
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)


class FakeResponse:
    def __init__(self, status, payload=None, retry_after=None):
        self.status_code = status
        self._payload = payload
        self.headers = {"Retry-After": retry_after} if retry_after else {}
        self.text = "synthetic response"

    def json(self):
        return self._payload


class FakeClient:
    def post(self, url, json):
        if "busy" in json["query"]:
            return FakeResponse(503, retry_after="1")
        return FakeResponse(200, {"method": "rerank", "results": [{}] * json["top_k"],
                                  "elapsed_ms": 10.0})


def test_status_classification_and_retry_after():
    client = FakeClient()
    ok = b.send(client, "http://localhost", "normal", 2)
    busy = b.send(client, "http://localhost", "busy", 2)
    assert ok["outcome"] == "accepted"
    assert ok["server_ms"] == 10.0
    assert busy["outcome"] == "rejected"
    assert busy["retry_after"] == "1"


def test_level_counts_rejections_separately():
    row = b.run_level(FakeClient(), "http://localhost",
                      ["normal", "busy", "normal", "busy"],
                      concurrency=2, top_k=2, warmup=0)
    assert (row["accepted"], row["rejected_503"], row["errors"]) == (2, 2, 0)
    assert row["rejection_fraction"] == 0.5
    assert row["accepted_http_latency"]["n"] == 2
    assert row["rejected_http_latency"]["n"] == 2
    assert row["accepted_server_search_latency"]["n"] == 2
    assert row["retry_after_values"] == ["1"]


def test_no_rejections():
    row = b.run_level(FakeClient(), "http://localhost", ["normal"] * 3,
                      concurrency=1, top_k=2, warmup=1)
    assert row["accepted"] == 3
    assert row["rejected_503"] == 0
    assert row["rejected_http_latency"] is None
    assert row["accepted_requests_per_second"] > 0
