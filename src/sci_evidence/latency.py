"""Latency summaries; unit tests operate without network, torch, or models."""

from __future__ import annotations

from statistics import fmean
from .statistics import percentile


def summarize_latencies(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms or any(x < 0 or not (x < float("inf")) for x in samples_ms):
        raise ValueError("Latencies must be nonnegative, finite and nonempty")
    ordered = sorted(samples_ms)
    total = sum(samples_ms)
    return {
        "n_queries": len(samples_ms),
        "mean_ms": fmean(samples_ms),
        "p50_ms": percentile(ordered, 0.50),
        "p95_ms": percentile(ordered, 0.95),
        "min_ms": ordered[0], "max_ms": ordered[-1],
        "sequential_queries_per_sec": len(samples_ms) / (total / 1000) if total else 0.0,
    }
