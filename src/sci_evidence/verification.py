"""Validate that saved test reports agree with their per-query measurements."""

from __future__ import annotations

import math
from statistics import fmean


def verify_report(baseline: dict, reranker: dict, tol: float = 1e-9) -> dict:
    """Verify saved SciFact full-test reports; does not recompute retrieval."""
    if baseline.get("split") != "test" or reranker.get("split") != "test":
        raise ValueError("Both reports must be for the test split")
    if baseline.get("parameters", {}).get("limit") is not None or reranker.get("parameters", {}).get("limit") is not None:
        raise ValueError("Reports must include all queries (no --limit)")
    a = baseline.get("per_query", [])
    b = reranker.get("per_query", [])
    if len(a) != 300 or len(b) != 300:
        raise ValueError("Expected 300 per-query results in both reports")
    def by_id(rows):
        result = {str(r["query_id"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicated query IDs")
        return result
    base = by_id(a)
    neural = by_id(b)
    if set(base) != set(neural):
        raise ValueError("BM25 and reranker reports have different query IDs")
    verified: dict = {}
    for name in ("ndcg@10", "recall@10"):
        base_values = [float(r[name]) for r in base.values()]
        paired_values = [float(r["bm25"][name]) for r in neural.values()]
        rerank_values = [float(r["reranked"][name]) for r in neural.values()]
        if not all(math.isfinite(v) for v in base_values + paired_values + rerank_values):
            raise ValueError(f"Nonfinite {name} metric")
        base_mean = fmean(base_values)
        rerank_base_mean = fmean(paired_values)
        neural_mean = fmean(rerank_values)
        if abs(base_mean - rerank_base_mean) > tol:
            raise ValueError(f"BM25 per-query discrepancy for {name}")
        if abs(base_mean - float(baseline["metrics"][name])) > tol:
            raise ValueError(f"Stored BM25 overall mismatch for {name}")
        if abs(rerank_base_mean - float(reranker["bm25"][name])) > tol:
            raise ValueError(f"Stored paired BM25 mismatch for {name}")
        if abs(neural_mean - float(reranker["bm25_plus_reranker"][name])) > tol:
            raise ValueError(f"Stored reranker overall mismatch for {name}")
        if abs((neural_mean - base_mean) - float(reranker["delta_reranker_minus_bm25"][name])) > tol:
            raise ValueError(f"Stored delta mismatch for {name}")
        verified[name] = {"bm25": base_mean, "reranked": neural_mean,
                          "delta": neural_mean - base_mean}
    return verified
