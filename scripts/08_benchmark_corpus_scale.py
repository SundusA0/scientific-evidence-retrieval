#!/usr/bin/env python3
"""Measure how the existing in-memory BM25 retriever scales on BEIR TREC-COVID.

This is a *systems* benchmark, NOT an evaluation of relevance quality. Index
construction, warm sequential lookup, and peak process RSS are reported
separately. Corpus subsets are prefixes of the same BEIR corpus for reproducible
size comparisons. The test queries are used only as query workloads; no qrels
or relevance scores are used for model selection here.

Prerequisite: download the BEIR trec-covid.zip into data/trec-covid.
Examples:
    python scripts/08_benchmark_corpus_scale.py --max-docs 5000
    python scripts/08_benchmark_corpus_scale.py --max-docs 20000
    python scripts/08_benchmark_corpus_scale.py --max-docs 0  # full corpus
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import random
import resource
import sys
from pathlib import Path
from time import perf_counter

from sci_evidence.data import Document
from sci_evidence.latency import summarize_latencies
from sci_evidence.retrieval import BM25Retriever


def peak_process_rss_mib() -> float:
    """Peak whole-process resident memory (not the BM25 index alone).

    On macOS ru_maxrss is bytes; on Linux it is KiB.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/trec-covid"))
    parser.add_argument("--max-docs", type=int, default=5000,
                        help="Corpus prefix to index; 0 indexes all available documents")
    parser.add_argument("--queries", type=int, default=20,
                        help="Number of TREC-COVID query texts used for timing")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_docs < 0 or args.queries < 1 or args.warmup < 0 or args.top_k < 1:
        raise SystemExit("Require max-docs >= 0, queries >= 1, warmup >= 0, top-k >= 1")

    corpus_file = args.data / "corpus.jsonl"
    if not corpus_file.is_file():
        raise SystemExit(f"Missing {corpus_file}. Download and extract BEIR TREC-COVID first.")

    query_file = args.data / "queries.jsonl"
    if not query_file.is_file():
        raise SystemExit(f"Missing {query_file}.")

    start_load = perf_counter()
    docs: list[Document] = []
    with corpus_file.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            docs.append(Document(id=str(row["_id"]), title=row.get("title", ""),
                                 text=row.get("text", "")))
            if args.max_docs and len(docs) >= args.max_docs:
                break
    queries = []
    with query_file.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                queries.append(json.loads(line)["text"])
    load_seconds = perf_counter() - start_load
    if not docs:
        raise SystemExit("Empty corpus")
    if args.top_k > len(docs):
        raise SystemExit("top-k must not exceed the number of indexed documents")
    if args.queries > len(queries):
        raise SystemExit(f"Only {len(queries)} query texts available; lower --queries")
    random.Random(args.seed).shuffle(queries)
    rss_after_load_mib = peak_process_rss_mib()

    print(f"Loaded {len(docs):,} documents in {load_seconds:.2f}s; indexing", flush=True)
    start_index = perf_counter()
    index = BM25Retriever(docs)
    index_seconds = perf_counter() - start_index
    rss_after_index_mib = peak_process_rss_mib()
    print(f"Index built in {index_seconds:.2f}s; peak process RSS: "
          f"{rss_after_index_mib:.1f} MiB", flush=True)

    for i in range(args.warmup):
        index.search(queries[i % len(queries)], top_k=args.top_k)

    latencies_ms = []
    for i, query in enumerate(queries[:args.queries], start=1):
        start_query = perf_counter()
        index.search(query, top_k=args.top_k)
        latencies_ms.append((perf_counter() - start_query) * 1000)
        if i % 10 == 0 or i == args.queries:
            print(f"Timed {i}/{args.queries} queries", flush=True)

    latency = summarize_latencies(latencies_ms)
    n_docs = len(docs)
    out = args.output or Path("results") / f"trec_covid_bm25_scale_{n_docs}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with corpus_file.open("r", encoding="utf-8") as stream:
        available_docs = sum(bool(line.strip()) for line in stream)
    with corpus_file.open("rb") as corpus_handle:
        corpus_sha256 = hashlib.file_digest(corpus_handle, "sha256").hexdigest()
    payload = {
        "dataset": "BEIR TREC-COVID",
        "experiment": "BM25 corpus-size and warm sequential query benchmark",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python": sys.version.split()[0],
        "corpus_sha256": corpus_sha256,
        "parameters": {
            "indexed_documents": n_docs,
            "available_documents": available_docs,
            "corpus_subset": "first N documents in corpus file" if args.max_docs else "full corpus",
            "sampled_queries": args.queries,
            "warmup_queries": args.warmup,
            "seed": args.seed,
            "top_k": args.top_k,
            "bm25_k1": index.k1,
            "bm25_b": index.b,
        },
        "corpus_load_seconds": round(load_seconds, 3),
        "index_build_seconds": round(index_seconds, 3),
        "peak_process_rss_mib_after_load": round(rss_after_load_mib, 2),
        "peak_process_rss_mib_after_index": round(rss_after_index_mib, 2),
        "peak_process_rss_mib_after_queries": round(peak_process_rss_mib(), 2),
        "unique_index_terms": len(index.postings),
        "mean_document_tokens": round(index.avg_len, 2),
        "warm_sequential_query_latency": latency,
        "notes": "Timing excludes corpus load and index build; includes only in-process BM25 lookup. "
                 "Peak RSS covers the entire Python process (including the corpus subset), "
                 "not index-only allocation. This is not concurrent throughput, and is not an IR "
                 "relevance evaluation. The query workload uses TREC-COVID query texts without qrels. "
                 "Subsets are not representative random samples; never compare quality across sizes.",
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nBM25 on {n_docs:,} documents")
    print(f"Index build: {index_seconds:.2f}s | peak RSS: {payload['peak_process_rss_mib_after_queries']:.1f} MiB")
    print(f"Warm p50: {latency['p50_ms']:.2f} ms | p95: {latency['p95_ms']:.2f} ms | "
          f"sequential queries/s: {latency['sequential_queries_per_sec']:.1f}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
