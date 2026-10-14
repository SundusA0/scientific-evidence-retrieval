#!/usr/bin/env python3
"""Evaluate BM25 -> cross-encoder reranking on SciFact validation or test.

Training uses only SciFact TRAIN qrels. Validate on a held-out portion of TRAIN;
run --split test *after* choosing parameters, without tuning on test scores.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean
import time

from sci_evidence.data import SciFactDataset, load_scifact
from sci_evidence.evaluation import evaluate_one
from sci_evidence.retrieval import BM25Retriever
from sci_evidence.reranking import candidate_upper_bound_recall, partition_train_queries


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/scifact"))
    p.add_argument("--model", default="runs/scifact_reranker",
                   help="Local fine-tuned model folder, or pretrained model ID")
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--candidate-k", type=int, default=50)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--limit", type=int, default=None, help="Smoke test, not full evaluation")
    p.add_argument("--output", type=Path, default=None)
    return p


def make_dataset(args) -> SciFactDataset:
    if args.split == "test":
        return load_scifact(args.data, split="test")
    ds = load_scifact(args.data, split="train")
    _, validation_ids = partition_train_queries(ds, seed=args.seed,
                                                validation_fraction=args.validation_fraction)
    return SciFactDataset(ds.documents,
                          {qid: ds.queries[qid] for qid in validation_ids},
                          {qid: ds.qrels[qid] for qid in validation_ids})


def main() -> None:
    args = parser().parse_args()
    if args.top_k < 1 or args.candidate_k < args.top_k or args.batch_size < 1:
        raise SystemExit("Require: 1 <= top-k <= candidate-k, batch-size >= 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("limit must be positive")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        raise SystemExit("Missing ML dependencies; install project [ml,dev] extras") from exc
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable; use --device cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; use --device cpu")

    ds = make_dataset(args)
    items = list(ds.queries.items())
    if args.limit is not None:
        items = items[:args.limit]
    print(f"Evaluating {len(items)} {args.split} queries against {len(ds.documents)} documents")
    bm25 = BM25Retriever(ds.documents)
    docs = {doc.id: doc for doc in ds.documents}

    print(f"Loading reranker: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    if model.config.num_labels != 1:
        raise SystemExit("Expected a model with exactly one ranking logit")
    device = torch.device(args.device)
    model.to(device).eval()

    rows = []
    start = time.monotonic()
    for idx, (qid, query) in enumerate(items, start=1):
        candidates = bm25.search(query, top_k=args.candidate_k)
        candidate_ids = [did for did, _ in candidates]
        scores: list[float] = []
        with torch.inference_mode():
            for offset in range(0, len(candidate_ids), args.batch_size):
                chunk = candidate_ids[offset:offset + args.batch_size]
                tokens = tokenizer([query] * len(chunk), [docs[did].full_text for did in chunk],
                                   return_tensors="pt", padding=True, truncation=True,
                                   max_length=args.max_length)
                tokens = {k: v.to(device) for k, v in tokens.items()}
                logits = model(**tokens).logits.squeeze(-1)
                scores.extend(logits.float().cpu().reshape(-1).tolist())
        rank_indices = sorted(range(len(candidate_ids)), key=lambda i: (-scores[i], i))
        reranked = [candidate_ids[i] for i in rank_indices[:args.top_k]]
        bm25_top = candidate_ids[:args.top_k]
        relevance = ds.qrels[qid]
        baseline = evaluate_one(bm25_top, relevance, k=args.top_k)
        neural = evaluate_one(reranked, relevance, k=args.top_k)
        rows.append({"query_id": qid, "bm25": baseline, "reranked": neural,
                     f"candidate_recall@{args.candidate_k}": candidate_upper_bound_recall(candidate_ids, relevance)})
        if idx % 25 == 0 or idx == len(items):
            print(f"Scored {idx}/{len(items)} queries", flush=True)

    names = (f"ndcg@{args.top_k}", f"recall@{args.top_k}")
    baseline = {k: mean(row["bm25"][k] for row in rows) for k in names}
    neural = {k: mean(row["reranked"][k] for row in rows) for k in names}
    ceiling = mean(row[f"candidate_recall@{args.candidate_k}"] for row in rows)
    elapsed = time.monotonic() - start
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "split": args.split,
        "parameters": {"candidate_k": args.candidate_k, "top_k": args.top_k,
                       "n_queries": len(rows), "seed": args.seed,
                       "validation_fraction": args.validation_fraction,
                       "limit": args.limit, "device": args.device, "max_length": args.max_length},
        "bm25": baseline, "bm25_plus_reranker": neural,
        "delta_reranker_minus_bm25": {k: neural[k] - baseline[k] for k in names},
        f"candidate_recall@{args.candidate_k}": ceiling,
        "evaluation_seconds": round(elapsed, 2), "per_query": rows,
        "notes": "Unjudged documents counted as nonrelevant. Limited query runs are smoke tests only."
    }
    output = args.output or Path("results") / (
        f"scifact_{args.split}_rerank_{Path(args.model).name}"
        + (f"_first{args.limit}" if args.limit else "") + ".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("\nRetrieval comparison on", args.split)
    for metric in names:
        print(f"{metric}: BM25 {baseline[metric]:.4f} | reranked {neural[metric]:.4f} "
              f"| delta {neural[metric] - baseline[metric]:+.4f}")
    print(f"Candidate recall@{args.candidate_k}: {ceiling:.4f}")
    print(f"Elapsed reranking/evaluation time: {elapsed:.1f} seconds")
    print("Saved", output)


if __name__ == "__main__":
    main()
