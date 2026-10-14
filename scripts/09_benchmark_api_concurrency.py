#!/usr/bin/env python3
"""Measure closed-loop HTTP concurrency on the local SciFact FastAPI application.

Run the API separately, then this client. Each concurrency level sends the same
training-side query workload. Measures full client-observed HTTP latency as
well as server-reported search time (excluding HTTP serialization/queuing).
These are NOT ranking-quality, distributed-serving, or open-loop load tests.

    python scripts/09_benchmark_api_concurrency.py --method bm25 --requests 200
    python scripts/09_benchmark_api_concurrency.py --method rerank --requests 200
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import platform
import random
from pathlib import Path
import sys
from time import perf_counter

import httpx


def load_train_queries(root: Path, seed: int) -> list[str]:
    qrels_path = root / "qrels" / "train.tsv"
    queries_path = root / "queries.jsonl"
    if not qrels_path.is_file() or not queries_path.is_file():
        raise ValueError(f"Missing SciFact train queries in {root}; run: python -m sci_evidence download")
    with qrels_path.open(encoding="utf-8", newline="") as handle:
        ids = {r["query-id"] for r in csv.DictReader(handle, delimiter="\t")}
    with queries_path.open(encoding="utf-8") as handle:
        queries = [r["text"] for line in handle if line.strip()
                   if (r := json.loads(line))["_id"] in ids]
    if not queries:
        raise ValueError("SciFact training-side query list is empty")
    random.Random(seed).shuffle(queries)
    return queries


def quantile(values: list[float], fraction: float) -> float:
    if not values or not 0 <= fraction <= 1:
        raise ValueError("Need nonempty values and a quantile in [0,1]")
    a = sorted(values)
    p = (len(a) - 1) * fraction
    lo = int(p)
    hi = min(lo + 1, len(a) - 1)
    return a[lo] + (p - lo) * (a[hi] - a[lo])


def describe(values: list[float]) -> dict | None:
    if not values:
        return None
    return {"n": len(values), "mean_ms": round(sum(values) / len(values), 3),
            "p50_ms": round(quantile(values, 0.50), 3),
            "p95_ms": round(quantile(values, 0.95), 3),
            "p99_ms": round(quantile(values, 0.99), 3),
            "min_ms": round(min(values), 3), "max_ms": round(max(values), 3)}


def send(client: httpx.Client, url: str, method: str, query: str, top_k: int) -> dict:
    started = perf_counter()
    try:
        response = client.post(url + "/search", json={
            "query": query, "method": method, "top_k": top_k,
        })
        elapsed_ms = (perf_counter() - started) * 1000
        response.raise_for_status()
        result = response.json()
        if result.get("method") != method or not isinstance(result.get("results"), list):
            raise ValueError("Unexpected API response shape or method")
        if len(result["results"]) != top_k:
            raise ValueError("Unexpected result count")
        return {"ok": True, "http_ms": elapsed_ms,
                "server_ms": float(result["elapsed_ms"])}
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:180]}"}


def run_level(client: httpx.Client, url: str, method: str,
              queries: list[str], concurrency: int, top_k: int,
              warmup: int) -> dict:
    if concurrency < 1 or not queries:
        raise ValueError("Require concurrency >= 1 and at least one query")
    # Warm-up is outside the timed window, so no model load/index creation
    # should be included in the measured HTTP calls.
    for i in range(warmup):
        status = send(client, url, method, queries[i % len(queries)], top_k)
        if not status["ok"]:
            raise RuntimeError("Warmup failed: " + status["error"])

    start = perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send, client, url, method, q, top_k)
                   for q in queries]
        statuses = [f.result() for f in as_completed(futures)]
    wall_seconds = perf_counter() - start
    successful = [r for r in statuses if r["ok"]]
    errors = [r["error"] for r in statuses if not r["ok"]]
    return {
        "concurrency": concurrency,
        "requests": len(queries), "successful": len(successful),
        "errors": len(errors), "error_examples": errors[:3],
        "wall_seconds": round(wall_seconds, 3),
        "successful_requests_per_second": round(len(successful) / wall_seconds, 3),
        "http_latency": describe([r["http_ms"] for r in successful]),
        "server_search_latency": describe([r["server_ms"] for r in successful]),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--method", choices=("bm25", "rerank"), default="bm25")
    p.add_argument("--data", type=Path, default=Path("data/scifact"))
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--requests", type=int, default=200,
                   help="Measured HTTP requests PER concurrency level")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--output", type=Path)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if (args.requests < 1 or args.warmup < 0 or not 1 <= args.top_k <= 20
            or args.timeout <= 0 or not args.concurrency
            or any(c < 1 or c > 32 for c in args.concurrency)
            or len(set(args.concurrency)) != len(args.concurrency)):
        raise SystemExit("Require requests >= 1, warmup >= 0, top-k 1..20, "
                         "timeout > 0, and distinct concurrency levels 1..32")
    try:
        queries = load_train_queries(args.data, args.seed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    workload = [queries[i % len(queries)] for i in range(args.requests)]
    url = args.url.rstrip("/")
    limits = httpx.Limits(max_connections=max(args.concurrency),
                          max_keepalive_connections=max(args.concurrency))
    try:
        with httpx.Client(timeout=args.timeout, limits=limits, trust_env=False) as client:
            health_response = client.get(url + "/health")
            health_response.raise_for_status()
            health = health_response.json()
            if args.method not in health.get("methods", []):
                raise SystemExit(f"Server does not support {args.method!r}; "
                                 "start uvicorn with SCI_EVIDENCE_RERANKER for reranking")
            print(f"API ready: {health.get('documents')} docs; "
                  f"methods={health.get('methods')}", flush=True)
            results = []
            for c in args.concurrency:
                print(f"Measuring {args.method}: concurrency={c}, "
                      f"requests={args.requests}...", flush=True)
                row = run_level(client, url, args.method, workload, c,
                                args.top_k, args.warmup)
                results.append(row)
                p = row["http_latency"] or {}
                print(f"  {row['successful']}/{row['requests']} OK | "
                      f"throughput {row['successful_requests_per_second']:.2f} req/s | "
                      f"HTTP p50 {p.get('p50_ms', 'n/a')} ms, "
                      f"p95 {p.get('p95_ms', 'n/a')} ms, "
                      f"p99 {p.get('p99_ms', 'n/a')} ms", flush=True)
    except (httpx.HTTPError, ValueError) as exc:
        raise SystemExit(f"Cannot connect to API at {url}: {exc}. "
                         "Start the uvicorn server in a separate terminal.") from exc
    output = args.output or Path("results") / f"scifact_api_concurrency_{args.method}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "closed-loop HTTP concurrency (one local Uvicorn worker)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "client_platform": platform.platform(), "client_python": sys.version.split()[0],
        "url": url, "server_health": health,
        "parameters": {"method": args.method, "requests_per_level": args.requests,
                       "warmup_per_level": args.warmup, "top_k": args.top_k,
                       "seed": args.seed, "concurrency_levels": args.concurrency,
                       "client_timeout_s": args.timeout,
                       "workload": "SciFact train-side query texts (no qrels used for scoring)"},
        "results": results,
        "notes": "One local server and one load-generator process on the same computer. "
                 "Closed-loop fixed-request-count test, not an open-loop target-QPS test. "
                 "HTTP latency includes client/network/response overhead and queueing; "
                 "API-reported search time excludes those. Thread concurrency is not "
                 "GPU-wide batching, multi-GPU training, or production throughput. "
                 "p99 is especially noisy with small samples. No ranking-quality claims.",
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {output}")
    if any(row["errors"] for row in results):
        raise SystemExit("One or more requests failed; inspect error_examples in the JSON report")


if __name__ == "__main__":
    main()
