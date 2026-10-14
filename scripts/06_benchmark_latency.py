#!/usr/bin/env python3
"""Measure warm, sequential, server-side retrieval latency on TRAIN queries.

Model load/index construction, network, and UI rendering are excluded.
Reranker mode includes BM25 candidate retrieval plus Transformer scoring.
Not a distributed throughput, stress, or concurrency benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import sys
from time import perf_counter

from sci_evidence.data import load_scifact
from sci_evidence.latency import summarize_latencies
from sci_evidence.serving import SearchEngine


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("bm25", "rerank"), default="bm25")
    p.add_argument("--model", default="runs/scifact_full_e1")
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--data", type=Path, default=Path("data/scifact"))
    p.add_argument("--queries", type=int, default=50)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--candidate-k", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output", type=Path)
    return p


def main():
    args = parser().parse_args()
    if args.queries < 1 or args.warmup < 0 or args.batch_size < 1 or not 1 <= args.top_k <= args.candidate_k:
        raise SystemExit("Require queries >= 1, warmup >= 0, 1 <= top-k <= candidate-k")
    ds = load_scifact(args.data, split="train")
    queries = list(ds.queries.values())
    random.Random(args.seed).shuffle(queries)
    if args.queries > len(queries):
        raise SystemExit(f"Too many queries: maximum {len(queries)} TRAIN queries")
    init = perf_counter()
    engine = SearchEngine(ds.documents,
                          reranker_model=args.model if args.method == "rerank" else None,
                          device=args.device, batch_size=args.batch_size)
    setup_s = perf_counter() - init
    for i in range(args.warmup):
        engine.search(queries[i % len(queries)], method=args.method,
                      top_k=args.top_k, candidate_k=args.candidate_k)
    samples = []
    for i, query in enumerate(queries[:args.queries], 1):
        result = engine.search(query, method=args.method, top_k=args.top_k,
                               candidate_k=args.candidate_k)
        samples.append(result.elapsed_ms)
        if i % 25 == 0 or i == args.queries:
            print(f"Measured {i}/{args.queries}", flush=True)
    metrics = summarize_latencies(samples)
    corpus_sha256 = hashlib.sha256((args.data / "corpus.jsonl").read_bytes()).hexdigest()
    model_weights = Path(args.model) / "model.safetensors"
    model_sha256 = hashlib.sha256(model_weights.read_bytes()).hexdigest() if args.method == "rerank" and model_weights.exists() else None
    payload = {
        "corpus_sha256": corpus_sha256,
        "local_model_safetensors_sha256": model_sha256,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "BEIR SciFact TRAIN queries (not test)",
        "method": args.method,
        "model": args.model if args.method == "rerank" else None,
        "device": args.device if args.method == "rerank" else "cpu",
        "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version.split()[0],
        "parameters": {"warmup_queries": args.warmup, "measured_queries": args.queries,
                       "seed": args.seed, "top_k": args.top_k,
                       "candidate_k": args.candidate_k if args.method == "rerank" else None,
                       "batch_size": args.batch_size if args.method == "rerank" else None},
        "index_and_model_load_seconds": round(setup_s, 3),
        "latency": metrics,
        "notes": "Sequential warm-query latency. Includes BM25 and (if rerank) cross-encoder forward pass, excludes indexing/model load, HTTP, browser and concurrency. CPU/MPS timings are environment-dependent. Model selection uses validation, not these timings or the final test metrics.",
    }
    output = args.output or Path("results") / f"scifact_latency_{args.method}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n{args.method.upper()} | {args.queries} SciFact TRAIN queries | device: {payload['device']}")
    print(f"Setup (excluded): {setup_s:.2f}s")
    print(f"p50: {metrics['p50_ms']:.2f}ms | p95: {metrics['p95_ms']:.2f}ms | mean: {metrics['mean_ms']:.2f}ms")
    print(f"Sequential throughput: {metrics['sequential_queries_per_sec']:.2f} queries/s")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
