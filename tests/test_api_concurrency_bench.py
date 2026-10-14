"""Unit tests for the HTTP concurrency benchmark, no model downloads needed."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "09_benchmark_api_concurrency.py"
spec = importlib.util.spec_from_file_location("api_concurrency_bench", SCRIPT)
bench = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(bench)


def test_load_train_queries_only(tmp_path):
    (tmp_path / "qrels").mkdir()
    (tmp_path / "qrels" / "train.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq2\td2\t1\n", encoding="utf-8")
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "held out"}) + "\n" +
        json.dumps({"_id": "q2", "text": "training query"}) + "\n",
        encoding="utf-8",
    )
    assert bench.load_train_queries(tmp_path, 42) == ["training query"]


def test_describe_and_quantile():
    assert bench.quantile([10, 20, 30], .5) == 20
    assert bench.describe([]) is None
    result = bench.describe([1.0, 2.0, 3.0])
    assert result["n"] == 3
    assert result["p95_ms"] == 2.9


def test_concurrency_over_real_local_http():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            body = json.dumps({"method": payload["method"],
                               "results": [{}] * payload["top_k"],
                               "elapsed_ms": 1.25}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        with httpx.Client(timeout=5, trust_env=False) as client:
            results = bench.run_level(client, url, "bm25", ["coffee", "sleep", "tea"],
                                      concurrency=2, top_k=2, warmup=1)
        assert results["successful"] == 3
        assert results["errors"] == 0
        assert results["http_latency"]["n"] == 3
        assert results["server_search_latency"]["p50_ms"] == 1.25
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
