#!/usr/bin/env python3
"""Check saved test result integrity without downloading data or rerunning ML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from sci_evidence.verification import verify_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bm25", type=Path, default=Path("results/scifact_test_bm25.json"))
    p.add_argument("--reranker", type=Path, default=Path("results/scifact_test_rerank_scifact_full_e1.json"))
    args = p.parse_args()
    try:
        baseline = json.loads(args.bm25.read_text(encoding="utf-8"))
        reranked = json.loads(args.reranker.read_text(encoding="utf-8"))
        metrics = verify_report(baseline, reranked)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise SystemExit(f"Verification FAILED: {exc}") from exc
    print("PASS — 300 matched SciFact test queries; stored means and deltas agree with per-query values")
    for name, values in metrics.items():
        print(f"{name}: BM25 {values['bm25']:.4f} | reranker {values['reranked']:.4f} | delta {values['delta']:+.4f}")
    print("This checks saved-result consistency, not independent reproducibility or statistical significance.")


if __name__ == "__main__":
    main()
