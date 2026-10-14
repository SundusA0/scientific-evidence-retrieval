#!/usr/bin/env python3
"""Paired analysis of saved SciFact BM25 vs reranker query results.

Run AFTER model selection and final test evaluation. No model downloads or
training are required; reads scripts/04_evaluate_reranker.py JSON output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from sci_evidence.statistics import paired_analysis


DEFAULT_INPUT = Path("results/scifact_test_rerank_scifact_full_e1.json")
DEFAULT_OUTPUT = Path("results/scifact_test_paired_statistics.json")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--iterations", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def extract_pairs(payload: dict, metric: str) -> tuple[list[float], list[float]]:
    rows = payload.get("per_query")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Input has no per_query measurements")
    seen: set[str] = set()
    bm25: list[float] = []
    reranked: list[float] = []
    for row in rows:
        qid = str(row["query_id"])
        if qid in seen:
            raise ValueError(f"Duplicate query_id: {qid}")
        seen.add(qid)
        bm25.append(float(row["bm25"][metric]))
        reranked.append(float(row["reranked"][metric]))
    return bm25, reranked


def main():
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Missing result file: {args.input}. Run the test reranker evaluator first.")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("split") != "test":
        raise SystemExit("This command expects the final held-out TEST report, not validation.")
    params = payload.get("parameters", {})
    if params.get("limit") is not None:
        raise SystemExit("Only the full test split can be used for this report (no --limit).")
    top_k = int(params.get("top_k", 10))
    rows = payload.get("per_query", [])
    if len(rows) != 300 or params.get("n_queries") != len(rows):
        raise SystemExit("Expected the complete 300-query SciFact test report.")

    output = {
        "source": str(args.input),
        "model": payload.get("model"),
        "split": "test",
        "n_queries": len(rows),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "Paired query bootstrap percentile 95% CI; two-sided sign-flip randomization test",
        "metrics": {},
        "limitations": [
            "SciFact relevance labels are incomplete; unjudged results count as nonrelevant.",
            "CIs and p-values assume queries are independent samples; paired sign-flip test assumes symmetry/exchangeability under its null.",
            "Two metrics are examined; no multiple-testing correction is applied.",
            "Model was selected on validation before evaluating the test set; do not tune subsequent versions on these test results.",
            "A confidence interval including zero is not proof of no effect.",
        ],
    }
    for i, metric in enumerate((f"ndcg@{top_k}", f"recall@{top_k}")):
        baseline, neural = extract_pairs(payload, metric)
        result = paired_analysis(baseline, neural, iterations=args.iterations, seed=args.seed + i)
        output["metrics"][metric] = result
        lo, hi = result["bootstrap_95pct_ci"]
        print(f"\n{metric} | {result['n_queries']} paired test queries")
        print(f"BM25: {result['baseline_mean']:.4f} | reranked: {result['reranked_mean']:.4f}")
        print(f"Mean delta: {result['mean_delta']:+.4f}")
        print(f"Paired 95% bootstrap CI: [{lo:+.4f}, {hi:+.4f}]")
        print(f"Exploratory two-sided sign-flip p: {result['sign_flip_two_sided_p']:.4f}")
        print(f"Queries improved / tied / worsened: {result['queries_improved']} / "
              f"{result['queries_tied']} / {result['queries_worsened']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("\nSaved", args.output)
    print("Interpret significance cautiously: two exploratory metrics, incomplete labels.")


if __name__ == "__main__":
    main()
