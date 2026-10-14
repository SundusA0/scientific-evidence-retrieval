#!/usr/bin/env python3
"""Measure potential throughput benefits from batching *multiple queries* on MPS/CPU.

This is a sequential, in-process inference experiment, NOT real HTTP batching:
there is no arrival process, waiting queue, thread pool, or overload behaviour.
The same fixed, train-side SciFact queries and BM25 top-k candidates are used
at every group size; candidate generation is excluded from the timed region.
The timed region includes tokenizer work, device transfers, model forward,
score transfer and per-query ranking. Model/index initialization is excluded.

Usage:
    python scripts/11_benchmark_cross_request_batching.py \\
      --model Sundus08/scifact-minilm-cross-encoder --device mps \\
      --queries 80 --candidate-k 20 --group-sizes 1 2 4 8
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import random
from statistics import fmean
from time import perf_counter


def group_ranges(count: int, group_size: int) -> list[tuple[int, int]]:
    """Return a partition covering every item exactly once."""
    if count < 1 or group_size < 1:
        raise ValueError("count and group_size must be positive")
    return [(start, min(count, start + group_size))
            for start in range(0, count, group_size)]


def group_stats(latency_ms: list[float], n_queries: int) -> dict[str, float | int]:
    """Summarize observed *batch* time and amortized inference throughput."""
    if not latency_ms or n_queries < 1 or any(not math.isfinite(v) or v <= 0 for v in latency_ms):
        raise ValueError("Need positive finite times and a positive query count")
    ordered = sorted(latency_ms)

    def quantile(p: float) -> float:
        pos = (len(ordered) - 1) * p
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] + (pos - lo) * (ordered[hi] - ordered[lo])

    total = sum(latency_ms)
    return {
        "n_batches": len(latency_ms),
        "n_query_inferences": n_queries,
        "batch_p50_ms": quantile(0.5),
        "batch_p95_ms": quantile(0.95),
        "mean_batch_ms": fmean(latency_ms),
        "amortized_ms_per_query": total / n_queries,
        "inference_queries_per_second": n_queries * 1000 / total,
    }


def sync_device(engine) -> None:
    if engine.device_name == "mps":
        engine.torch.mps.synchronize()
    elif engine.device_name == "cuda":
        engine.torch.cuda.synchronize()


def infer_group(engine, work: list[tuple[int, str, list[str], list[str]]],
                top_k: int) -> tuple[float, dict[int, tuple[list[float], list[str]]]]:
    """Score flattened (query, document) pairs, then split rankings per query."""
    tokenizer = engine.tokenizer
    torch = engine.torch
    model = engine.model
    assert tokenizer is not None and torch is not None and model is not None

    questions: list[str] = []
    passages: list[str] = []
    slices: list[tuple[int, int, list[str], int]] = []
    for index, question, ids, documents in work:
        if len(ids) != len(documents) or not ids:
            raise ValueError("Each query needs matching nonempty ids and texts")
        start = len(questions)
        questions.extend([question] * len(ids))
        passages.extend(documents)
        slices.append((start, len(questions), ids, index))

    sync_device(engine)
    t = perf_counter()
    tokens = tokenizer(
        questions, passages, return_tensors="pt", padding=True,
        truncation=True, max_length=engine.max_length,
    )
    tokens = {key: val.to(engine.device) for key, val in tokens.items()}
    with torch.inference_mode():
        scores = model(**tokens).logits.float().reshape(-1).cpu().tolist()
    sync_device(engine)
    answers: dict[int, tuple[list[float], list[str]]] = {}
    for start, stop, ids, index in slices:
        local = scores[start:stop]
        order = sorted(range(len(ids)), key=lambda i: (-local[i], i))
        answers[index] = (local, [ids[i] for i in order[:top_k]])
    elapsed_ms = (perf_counter() - t) * 1000
    return elapsed_ms, answers


def compare_predictions(reference: dict, current: dict) -> dict[str, float]:
    """Compare per-pair logits and top-k identities, not exact floating precision."""
    if set(reference) != set(current):
        raise ValueError("Query sets differ")
    deltas = []
    matches = 0
    for key in reference:
        a_scores, a_ids = reference[key]
        b_scores, b_ids = current[key]
        if len(a_scores) != len(b_scores):
            raise ValueError("Different number of candidates")
        deltas.extend(abs(a - b) for a, b in zip(a_scores, b_scores))
        matches += a_ids == b_ids
    return {
        "max_abs_logit_difference": max(deltas, default=0.0),
        "top_k_exact_match_fraction": matches / len(reference),
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/scifact"))
    p.add_argument("--model", default="Sundus08/scifact-minilm-cross-encoder")
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--queries", type=int, default=80)
    p.add_argument("--candidate-k", type=int, default=20)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--group-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--warmup", type=int, default=3,
                   help="Warmup batches per group size")
    p.add_argument("--repeats", type=int, default=2,
                   help="Complete passes over identical queries at each group size")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path,
                   default=Path("results/scifact_cross_request_batching_pilot.json"))
    return p


def main() -> None:
    from sci_evidence.data import load_scifact
    from sci_evidence.serving import SearchEngine

    args = parser().parse_args()
    if (args.queries < 1 or args.candidate_k < 1 or not 1 <= args.top_k <= args.candidate_k
            or args.repeats < 1 or args.warmup < 0 or not args.group_sizes
            or 1 not in args.group_sizes or any(g < 1 or g > 8 for g in args.group_sizes)
            or len(set(args.group_sizes)) != len(args.group_sizes)):
        raise SystemExit("Require queries >=1, 1 <= top-k <= candidate-k, repeats >=1, "
                         "warmup >=0, and distinct group sizes from 1 through 8 including 1")
    if not (args.data / "corpus.jsonl").is_file():
        raise SystemExit("SciFact not found; run: python -m sci_evidence download")
    dataset = load_scifact(args.data, split="train")
    queries = list(dataset.queries.values())
    random.Random(args.seed).shuffle(queries)
    if args.queries > len(queries) or args.candidate_k > len(dataset.documents):
        raise SystemExit("Requested queries or candidate count exceeds the dataset")
    queries = queries[:args.queries]

    print(f"Loading {args.model} on {args.device}...", flush=True)
    engine = SearchEngine(dataset.documents, reranker_model=args.model,
                          device=args.device, batch_size=args.candidate_k)
    print("Preparing identical BM25 candidate lists for every setting...", flush=True)
    work = []
    for i, query in enumerate(queries):
        ids = [doc_id for doc_id, _ in engine.bm25.search(query, top_k=args.candidate_k)]
        work.append((i, query, ids, [engine.docs[doc_id].full_text for doc_id in ids]))

    results = []
    reference = None
    for group_size in sorted(args.group_sizes):
        for _ in range(args.warmup):
            infer_group(engine, work[:group_size], args.top_k)
        durations = []
        first_pass = {}
        for repeat in range(args.repeats):
            for start, stop in group_ranges(len(work), group_size):
                elapsed, outputs = infer_group(engine, work[start:stop], args.top_k)
                durations.append(elapsed)
                if repeat == 0:
                    first_pass.update(outputs)
        if group_size == 1:
            reference = first_pass
        assert reference is not None
        row = {
            "group_size": group_size,
            "maximum_pairs_per_forward": group_size * args.candidate_k,
            **group_stats(durations, len(work) * args.repeats),
            **compare_predictions(reference, first_pass),
        }
        results.append(row)
        print(f"{group_size:>2} queries/batch ({row['maximum_pairs_per_forward']:>3} pairs) "
              f"| {row['inference_queries_per_second']:.2f} q/s "
              f"| amortized {row['amortized_ms_per_query']:.2f} ms/query "
              f"| batch p95 {row['batch_p95_ms']:.2f} ms "
              f"| top-{args.top_k} agreement {row['top_k_exact_match_fraction']:.1%}",
              flush=True)
    report = {
        "experiment": "Sequential simulated cross-request inference batching, not an HTTP server",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "SciFact training-side queries; fixed BM25 candidates",
        "model": args.model,
        "device": args.device,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "parameters": {
            "queries": args.queries, "candidate_k": args.candidate_k,
            "top_k": args.top_k, "group_sizes": sorted(args.group_sizes),
            "repeats": args.repeats, "warmup_batches_per_size": args.warmup,
            "seed": args.seed,
        },
        "results": results,
        "notes": "Each group represents simultaneous *availability* of queries, not simulated "
                 "arrival times or a serving queue. Includes tokenizer, H2D, model forward, D2H "
                 "and per-query ranking; excludes BM25, model/index setup, network, and "
                 "queueing delay. A batch with G queries scores G*candidate_k pairs. "
                 "Amortized latency is not individual HTTP response latency. "
                 "Larger groups may change padding and floating-point logits. "
                 "Single-machine, exploratory performance data only.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
