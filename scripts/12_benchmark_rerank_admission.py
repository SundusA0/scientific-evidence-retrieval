#!/usr/bin/env python3
"""Closed-loop reranker overload experiment: account separately for HTTP 200 and 503.

Local one-worker FastAPI service only. Each workload contains the same SciFact
TRAIN query texts at every concurrency setting. No retries: overload is a real
outcome, not a successful request, and offered work is a fixed request count.

Run with a separate model-serving Uvicorn process on port 8002:
    python scripts/12_benchmark_rerank_admission.py --url http://127.0.0.1:8002
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import sys
from time import perf_counter

import httpx


def percentile(values: list[float], fraction: float) -> float:
    if not values or not 0 <= fraction <= 1:
        raise ValueError("Expected nonempty values and quantile 0..1")
    a = sorted(values)
    pos = (len(a) - 1) * fraction
    low = int(pos)
    high = min(low + 1, len(a) - 1)
    return a[low] + (pos - low) * (a[high] - a[low])


def summary(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "n": len(values),
        "mean_ms": round(sum(values) / len(values), 3),
        "p50_ms": round(percentile(values, 0.50), 3),
        "p95_ms": round(percentile(values, 0.95), 3),
        "p99_ms": round(percentile(values, 0.99), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def load_train_queries(root: Path, seed: int) -> list[str]:
    qrels_file = root / "qrels" / "train.tsv"
    queries_file = root / "queries.jsonl"
    if not qrels_file.is_file() or not queries_file.is_file():
        raise ValueError(f"SciFact training queries not found in {root}")
    with qrels_file.open(encoding="utf-8", newline="") as f:
        judged = {r["query-id"] for r in csv.DictReader(f, delimiter="\t")}
    with queries_file.open(encoding="utf-8") as f:
        queries = [r["text"] for line in f if line.strip()
                   if (r := json.loads(line))["_id"] in judged]
    if not queries:
        raise ValueError("No SciFact training queries found")
    random.Random(seed).shuffle(queries)
    return queries


def send(client: httpx.Client, url: str, query: str, top_k: int) -> dict:
    start = perf_counter()
    try:
        response = client.post(url + "/search", json={
            "query": query, "method": "rerank", "top_k": top_k,
        })
    except httpx.RequestError as exc:
        return {"outcome": "error", "reason": f"{type(exc).__name__}: {str(exc)[:140]}",
                "http_ms": (perf_counter() - start) * 1000}

    duration = (perf_counter() - start) * 1000
    if response.status_code == 503:
        return {"outcome": "rejected", "http_ms": duration,
                "retry_after": response.headers.get("Retry-After")}
    if response.status_code != 200:
        return {"outcome": "error", "http_ms": duration,
                "reason": f"Unexpected HTTP {response.status_code}: {response.text[:100]}"}
    try:
        payload = response.json()
        if payload.get("method") != "rerank" or len(payload["results"]) != top_k:
            raise ValueError("Unexpected reranker response content")
        server_ms = float(payload["elapsed_ms"])
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return {"outcome": "error", "http_ms": duration,
                "reason": f"Invalid response: {exc}"}
    return {"outcome": "accepted", "http_ms": duration, "server_ms": server_ms}


def run_level(client: httpx.Client, url: str, queries: list[str],
              concurrency: int, top_k: int, warmup: int) -> dict:
    if concurrency < 1 or not queries or warmup < 0:
        raise ValueError("Concurrency and queries must be positive; warmup nonnegative")
    for i in range(warmup):
        outcome = send(client, url, queries[i % len(queries)], top_k)
        if outcome["outcome"] != "accepted":
            raise RuntimeError("Warmup request failed: " + repr(outcome))

    started = perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send, client, url, query, top_k) for query in queries]
        outcomes = [f.result() for f in as_completed(futures)]
    elapsed = perf_counter() - started
    accepted = [o for o in outcomes if o["outcome"] == "accepted"]
    rejected = [o for o in outcomes if o["outcome"] == "rejected"]
    errors = [o for o in outcomes if o["outcome"] == "error"]
    n = len(outcomes)
    return {
        "concurrency": concurrency,
        "requests": n,
        "accepted": len(accepted),
        "rejected_503": len(rejected),
        "errors": len(errors),
        "error_examples": [e["reason"] for e in errors[:3]],
        "rejection_fraction": round(len(rejected) / n, 4),
        "retry_after_values": sorted({str(o["retry_after"]) for o in rejected}),
        "wall_seconds": round(elapsed, 3),
        "accepted_requests_per_second": round(len(accepted) / elapsed, 3),
        "all_responses_per_second": round(n / elapsed, 3),
        "accepted_http_latency": summary([o["http_ms"] for o in accepted]),
        "rejected_http_latency": summary([o["http_ms"] for o in rejected]),
        "accepted_server_search_latency": summary([o["server_ms"] for o in accepted]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8002")
    p.add_argument("--expected-capacity", type=int, default=1,
                   help="Assert /health reports this capacity; 0 is unrestricted")
    p.add_argument("--data", type=Path, default=Path("data/scifact"))
    p.add_argument("--requests", type=int, default=200, help="Requests per concurrency level")
    p.add_argument("--concurrency", nargs="+", type=int, default=[1, 4, 8, 16])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--output", type=Path,
                   default=Path("results/scifact_admission_capacity1.json"))
    args = p.parse_args()
    if (args.requests < 1 or args.warmup < 0 or args.timeout <= 0 or args.expected_capacity < 0
            or not 1 <= args.top_k <= 20 or not args.concurrency
            or any(c < 1 or c > 32 for c in args.concurrency)
            or len(set(args.concurrency)) != len(args.concurrency)):
        p.error("Require positive requests/timeout, top-k 1..20, capacity >=0, and distinct concurrency 1..32")

    try:
        train_queries = load_train_queries(args.data, args.seed)
    except ValueError as exc:
        p.error(str(exc))
    workload = [train_queries[i % len(train_queries)] for i in range(args.requests)]
    url = args.url.rstrip("/")
    limits = httpx.Limits(max_connections=max(args.concurrency),
                          max_keepalive_connections=max(args.concurrency))
    try:
        with httpx.Client(timeout=args.timeout, limits=limits, trust_env=False) as client:
            health = client.get(url + "/health")
            health.raise_for_status()
            state = health.json()
            if "rerank" not in state.get("methods", []):
                p.error("Reranker unavailable on the selected server")
            if state.get("max_inflight_rerank") != args.expected_capacity:
                p.error(f"Server capacity {state.get('max_inflight_rerank')!r} != expected "
                        f"{args.expected_capacity}; check port and settings")
            print(f"Verified server: {state['documents']} documents, "
                  f"capacity={state['max_inflight_rerank']}, device={state['device']}", flush=True)
            results = []
            for c in args.concurrency:
                print(f"Measuring {args.requests} requests at concurrency={c}...", flush=True)
                r = run_level(client, url, workload, c, args.top_k, args.warmup)
                results.append(r)
                latency = r["accepted_http_latency"] or {}
                print(f"  Accepted {r['accepted']} | 503 {r['rejected_503']} | errors {r['errors']}"
                      f" | accepted {r['accepted_requests_per_second']:.2f}/s"
                      f" | accepted p50 {latency.get('p50_ms', 'n/a')} ms", flush=True)
    except httpx.HTTPError as exc:
        raise SystemExit(f"Could not reach {url}: {exc}. Check the Uvicorn terminal.") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "experiment": "closed-loop reranker admission-control benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "client_platform": platform.platform(),
        "python": sys.version.split()[0],
        "server_url": url,
        "server_health": state,
        "parameters": {
            "requested_concurrency": args.concurrency,
            "requests_per_level": args.requests,
            "warmup_per_level": args.warmup,
            "top_k": args.top_k,
            "seed": args.seed,
            "expected_capacity": args.expected_capacity,
            "timeout_s": args.timeout,
        },
        "results": results,
        "notes": "Local closed-loop fixed-request-count benchmark. Every request counted; HTTP 503 "
                 "is an expected overload outcome and is never retried. Only HTTP 200 contributes "
                 "to accepted throughput or success latency. Both load generator and single Uvicorn "
                 "worker run on one computer. Accepted latency may look artificially low because "
                 "many requests are rejected. No open-loop arrival process, SLO analysis, or "
                 "production-tail-latency inference; p95/p99 are noisy with few successes.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Saved:", args.output)
    if any(r["errors"] for r in results):
        raise SystemExit("Unexpected errors occurred; inspect the JSON report")


if __name__ == "__main__":
    main()
