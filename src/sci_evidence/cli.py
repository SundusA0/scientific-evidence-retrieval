"""Command line entry point for experiments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import download_scifact, load_scifact, tiny_demo_dataset
from .evaluation import benchmark
from .retrieval import BM25Retriever, DenseRetriever

DEFAULT_DATA = Path("data/scifact")
DEFAULT_RESULTS = Path("results")


def _build_retriever(args, docs):
    if args.method == "bm25":
        return BM25Retriever(docs)
    return DenseRetriever(docs, model_name=args.model, device=args.device,
                          batch_size=args.batch_size)


def _add_method_options(p):
    p.add_argument("--method", choices=("bm25", "dense"), default="bm25")
    p.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default=None,
                   help="Only used with dense; default is automatic")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)


def build_parser():
    p = argparse.ArgumentParser(prog="evidence",
                                description="Scientific-evidence retrieval research prototype")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="Download BEIR SciFact (no model weights)")
    d.add_argument("--data", type=Path, default=DEFAULT_DATA)

    sub.add_parser("demo", help="Run offline toy example (synthetic, not a benchmark)")

    b = sub.add_parser("benchmark", help="Evaluate on held-out SciFact test queries")
    _add_method_options(b)
    b.add_argument("--split", choices=("test", "train"), default="test")
    b.add_argument("--top-k", type=int, default=10)
    b.add_argument("--limit", type=int, default=None,
                   help="Smoke test only; limited result is NOT the full benchmark")
    b.add_argument("--output", type=Path, default=None)

    s = sub.add_parser("search", help="Search scientific abstracts")
    _add_method_options(s)
    s.add_argument("--query", required=True)
    s.add_argument("--top-k", type=int, default=5)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "download":
        download_scifact(args.data)
        return
    if args.command == "demo":
        ds = tiny_demo_dataset()
        r = BM25Retriever(ds.documents)
        metrics, _ = benchmark(r, ds)
        print("OFFLINE DEMO — synthetic toy data, not a real scientific benchmark")
        for qid, query in ds.queries.items():
            print(f"\n{query}")
            for doc_id, score in r.search(query, top_k=3):
                doc = next(d for d in ds.documents if d.id == doc_id)
                print(f"  {doc_id}: {doc.title} | BM25 score={score:.3f}")
        print(f"\nToy metrics: {json.dumps(metrics, indent=2)}")
        return

    ds = load_scifact(args.data, split=getattr(args, "split", "test"))
    print(f"Loaded {len(ds.documents)} scientific abstracts and {len(ds.queries)} judged queries")
    retriever = _build_retriever(args, ds.documents)
    if args.command == "search":
        by_id = {d.id: d for d in ds.documents}
        for rank, (doc_id, score) in enumerate(retriever.search(args.query, args.top_k), 1):
            doc = by_id[doc_id]
            print(f"{rank}. [{doc_id}] {doc.title}\n   Score: {score:.4f}\n   {doc.text[:220]}...\n")
        return

    metrics, per_query = benchmark(retriever, ds, k=args.top_k, limit=args.limit)
    output = args.output or DEFAULT_RESULTS / (f"scifact_{args.split}_{args.method}" +
                (f"_first{args.limit}" if args.limit else "") + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "BEIR SciFact", "split": args.split, "method": args.method,
        "model": args.model if args.method == "dense" else None,
        "parameters": {"k": args.top_k, "limit": args.limit, "n_queries_evaluated": len(per_query),
                       "n_documents": len(ds.documents)},
        "metrics": metrics,
        "notes": "Unjudged papers are treated as nonrelevant. Limited runs are not comparable to full benchmarks.",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "per_query": per_query,
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nSciFact retrieval benchmark")
    print(f"Method: {args.method} | Split: {args.split} | Queries: {len(per_query)}")
    for name, val in metrics.items():
        print(f"{name}: {val:.4f}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
