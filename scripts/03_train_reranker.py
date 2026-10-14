#!/usr/bin/env python3
"""Fine-tune a small pretrained cross-encoder on SciFact TRAIN judgments only.

Quick hardware smoke test (25 training queries):
python scripts/03_train_reranker.py --max-train-queries 25 --out runs/scifact_smoke --device mps

Full run (all train-side queries except a fixed validation subset):
python scripts/03_train_reranker.py --out runs/scifact_reranker --device mps

Warning: BM25-mined unjudged documents are weak negatives, not verified false claims.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import random
from pathlib import Path
import time

from sci_evidence.data import load_scifact
from sci_evidence.retrieval import BM25Retriever
from sci_evidence.reranking import mine_train_pairs, partition_train_queries


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/scifact"))
    p.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--out", type=Path, default=Path("runs/scifact_reranker"))
    p.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--candidate-k", type=int, default=50)
    p.add_argument("--negatives-per-positive", type=int, default=3)
    p.add_argument("--max-train-queries", type=int, default=None,
                   help="Smoke test only; uses first N train queries (do not claim full training)")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=256)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 8:
        raise SystemExit("epochs, batch-size must be positive and max-length >= 8")
    if args.max_train_queries is not None and args.max_train_queries < 1:
        raise SystemExit("max-train-queries must be positive")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"Output directory {args.out} is not empty; use a new --out path")

    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError as exc:
        raise SystemExit("Missing ML libraries. Install the project with [ml,dev] extras.") from exc

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable; try --device cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable; try --device cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "mps":
        torch.mps.manual_seed(args.seed)

    ds = load_scifact(args.data, split="train")
    train_ids, validation_ids = partition_train_queries(
        ds, seed=args.seed, validation_fraction=args.validation_fraction
    )
    if args.max_train_queries is not None:
        train_ids = train_ids[:args.max_train_queries]
    print(f"SciFact training-side query split: {len(train_ids)} train / "
          f"{len(validation_ids)} held-out validation")
    print("Mining hard negatives from BM25 top-k, using TRAIN judgments only...")
    bm25 = BM25Retriever(ds.documents)
    examples = mine_train_pairs(ds, train_ids, bm25,
                                candidate_k=args.candidate_k,
                                negatives_per_positive=args.negatives_per_positive)
    counts = Counter(int(ex.label) for ex in examples)
    print(f"Training pairs: {len(examples)} ({counts[1]} positives, "
          f"{counts[0]} unjudged BM25 negatives)")
    if not examples or not counts[0] or not counts[1]:
        raise SystemExit("Not enough positive and negative examples to train")

    print(f"Loading cross-encoder: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    if model.config.num_labels != 1:
        raise SystemExit("Expected a one-logit sequence-classification reranker")
    device = torch.device(args.device)
    model.to(device)
    model.train()

    def collate(batch):
        tokenized = tokenizer(
            [ex.query for ex in batch], [ex.document for ex in batch],
            padding=True, truncation=True, max_length=args.max_length,
            return_tensors="pt",
        )
        labels = torch.tensor([ex.label for ex in batch], dtype=torch.float32)
        return tokenized, labels

    loader = DataLoader(examples, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed),
                        collate_fn=collate, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        losses: list[float] = []
        for step, (tokens, labels) in enumerate(loader, 1):
            inputs = {k: v.to(device) for k, v in tokens.items()}
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**inputs).logits.squeeze(-1)
            loss = loss_fn(logits.float(), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if step == 1 or step % 25 == 0 or step == len(loader):
                print(f"Epoch {epoch}/{args.epochs} | batch {step}/{len(loader)} "
                      f"| mean BCE loss {sum(losses)/len(losses):.4f}", flush=True)
        print(f"Epoch {epoch} completed; average training loss {sum(losses)/len(losses):.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    model.eval().cpu().save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": args.model,
        "split": "SciFact train only",
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "n_train_queries": len(train_ids),
        "n_validation_queries_excluded": len(validation_ids),
        "n_pairs": len(examples),
        "n_positive_pairs": counts[1],
        "n_mined_negative_pairs": counts[0],
        "parameters": {"candidate_k": args.candidate_k,
                       "negatives_per_positive": args.negatives_per_positive,
                       "max_train_queries": args.max_train_queries,
                       "epochs": args.epochs, "batch_size": args.batch_size,
                       "learning_rate": args.learning_rate, "max_length": args.max_length,
                       "device": args.device},
        "training_seconds": round(time.monotonic() - start, 2),
        "warning": "BM25 hard negatives are unjudged, not confirmed irrelevant.",
    }
    (args.out / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"\nSaved fine-tuned reranker to {args.out}")
    print("Next: evaluate on validation queries BEFORE consulting the test split.")


if __name__ == "__main__":
    main()
