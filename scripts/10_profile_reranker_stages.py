#!/usr/bin/env python3
"""Profile serial BM25 + cross-encoder inference stages on SciFact TRAIN queries.

This is an *in-process*, sequential, warm-query diagnostic, NOT an HTTP,
concurrency, production throughput, or relevance-quality benchmark. Measures
BM25 candidate selection, tokenizer preparation, input device transfer, neural
forward pass, score transfer, and final ordering separately. GPU synchronizations
are inserted at stage boundaries to attribute GPU time; this changes the normal
execution schedule, so stage times are approximate diagnostics, not a drop-in
replacement for end-to-end latency measurements.

Example:
  python scripts/10_profile_reranker_stages.py --device mps \\
    --model Sundus08/scifact-minilm-cross-encoder --queries 50
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
from time import perf_counter


STAGE_KEYS = (
    "bm25_ms", "tokenize_ms", "transfer_inputs_ms", "forward_ms",
    "scores_to_cpu_ms", "rank_ms", "total_ms",
)


def summarize_stages(samples: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Summarize completed per-query stage times, validating sample structure."""
    if not samples:
        raise ValueError("At least one profiling sample is required")
    if any(set(sample) != set(STAGE_KEYS) for sample in samples):
        raise ValueError("Each profiling sample must contain exactly the expected stages")
    from sci_evidence.latency import summarize_latencies
    return {key: summarize_latencies([sample[key] for sample in samples])
            for key in STAGE_KEYS}


def sync_device(engine: SearchEngine) -> None:
    if engine.device_name == "mps":
        engine.torch.mps.synchronize()
    elif engine.device_name == "cuda":
        engine.torch.cuda.synchronize()


def profile_query(engine: SearchEngine, query: str, *, candidate_k: int,
                  top_k: int) -> tuple[dict[str, float], list[str]]:
    """Reproduce the current single-request inference path, with stage barriers."""
    assert engine.model is not None and engine.tokenizer is not None
    assert engine.torch is not None and engine.device is not None

    sync_device(engine)
    query_start = perf_counter()

    t = perf_counter()
    candidates = engine.bm25.search(query, top_k=candidate_k)
    bm25_ms = (perf_counter() - t) * 1000

    ids = [doc_id for doc_id, _ in candidates]
    t = perf_counter()
    tokens = engine.tokenizer(
        [query] * len(ids),
        [engine.docs[doc_id].full_text for doc_id in ids],
        padding=True, truncation=True, max_length=engine.max_length,
        return_tensors="pt",
    )
    tokenize_ms = (perf_counter() - t) * 1000

    t = perf_counter()
    tokens = {name: value.to(engine.device) for name, value in tokens.items()}
    sync_device(engine)
    transfer_inputs_ms = (perf_counter() - t) * 1000

    t = perf_counter()
    with engine.torch.inference_mode():
        logits = engine.model(**tokens).logits.squeeze(-1)
    sync_device(engine)
    forward_ms = (perf_counter() - t) * 1000

    t = perf_counter()
    scores = logits.float().cpu().reshape(-1).tolist()
    scores_to_cpu_ms = (perf_counter() - t) * 1000

    t = perf_counter()
    indices = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
    winners = [ids[i] for i in indices[:top_k]]
    rank_ms = (perf_counter() - t) * 1000

    total_ms = (perf_counter() - query_start) * 1000
    return {
        "bm25_ms": bm25_ms, "tokenize_ms": tokenize_ms,
        "transfer_inputs_ms": transfer_inputs_ms, "forward_ms": forward_ms,
        "scores_to_cpu_ms": scores_to_cpu_ms, "rank_ms": rank_ms,
        "total_ms": total_ms,
    }, winners


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/scifact"))
    p.add_argument("--model", default="Sundus08/scifact-minilm-cross-encoder")
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--queries", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--candidate-k", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path,
                   default=Path("results/scifact_reranker_stage_profile.json"))
    return p


def main() -> None:
    from sci_evidence.data import load_scifact
    from sci_evidence.serving import SearchEngine

    args = parser().parse_args()
    if (args.queries < 1 or args.warmup < 0 or args.batch_size < 1
            or not 1 <= args.top_k <= args.candidate_k):
        raise SystemExit("Require queries >=1, warmup >=0, batch-size >=1, "
                         "and 1 <= top-k <= candidate-k")
    if args.candidate_k > args.batch_size:
        raise SystemExit("This profiler supports candidate-k <= batch-size only, "
                         "matching the current one-batch API configuration")
    if not (args.data / "corpus.jsonl").exists():
        raise SystemExit(f"Missing {args.data / 'corpus.jsonl'}; download SciFact first")

    ds = load_scifact(args.data, split="train")
    queries = list(ds.queries.values())
    random.Random(args.seed).shuffle(queries)
    if args.queries > len(queries):
        raise SystemExit(f"Only {len(queries)} train queries available")
    if args.candidate_k > len(ds.documents):
        raise SystemExit("candidate-k exceeds corpus size")

    print(f"Loading index and model ({args.model}) on {args.device}...", flush=True)
    t = perf_counter()
    engine = SearchEngine(
        ds.documents, reranker_model=args.model,
        device=args.device, batch_size=args.batch_size,
    )
    setup_seconds = perf_counter() - t
    print(f"Ready: {len(ds.documents):,} abstracts; setup {setup_seconds:.2f}s", flush=True)

    for i in range(args.warmup):
        engine.search(queries[i % len(queries)], method="rerank",
                      top_k=args.top_k, candidate_k=args.candidate_k)

    rows: list[dict[str, float]] = []
    for i, query in enumerate(queries[:args.queries], 1):
        stages, winners = profile_query(engine, query,
                                        candidate_k=args.candidate_k,
                                        top_k=args.top_k)
        # Confirm once that staged inference preserves the original ranking path.
        if i == 1:
            reference = engine.search(query, method="rerank",
                                      top_k=args.top_k,
                                      candidate_k=args.candidate_k)
            if winners != [hit.doc_id for hit in reference.hits]:
                raise RuntimeError("Staged inference disagrees with the serving path")
        rows.append(stages)
        if i % 10 == 0 or i == args.queries:
            print(f"Profiled {i}/{args.queries} queries", flush=True)

    result = summarize_stages(rows)
    report = {
        "experiment": "sequential in-process reranker stage breakdown",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "BEIR SciFact train queries (no test labels)",
        "model": args.model, "device": args.device,
        "platform": platform.platform(), "machine": platform.machine(),
        "setup_seconds_excluded": round(setup_seconds, 3),
        "parameters": {"queries": args.queries, "warmup": args.warmup,
                       "candidate_k": args.candidate_k, "batch_size": args.batch_size,
                       "top_k": args.top_k, "seed": args.seed},
        "stages": result,
        "per_query": rows,
        "notes": "Measured sequentially with explicit GPU synchronizations at stage "
                 "boundaries; extra barriers alter scheduling relative to normal "
                 "serving. Not an HTTP, multi-request, or concurrency benchmark. "
                 "No relevance judgments used for profiling or model selection. "
                 "p95 from 50 queries is exploratory; full test set is not used.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nStage breakdown (p50 / p95 milliseconds):")
    for stage in STAGE_KEYS:
        value = result[stage]
        print(f"  {stage:20s} {value['p50_ms']:8.2f} / {value['p95_ms']:8.2f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
