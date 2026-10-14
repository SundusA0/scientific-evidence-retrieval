"""Paired, query-level resampling for retrieval evaluation.

Each evaluation query is the paired experimental unit. This module computes
percentile bootstrap intervals for the mean query delta and a two-sided paired
sign-flip randomization p-value. These are exploratory inference estimates,
not evidence of causal impact or a correction for incomplete relevance labels.
"""

from __future__ import annotations

import math
import random
from statistics import fmean
from typing import Sequence


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linearly interpolated sample quantile (0 <= fraction <= 1)."""
    if not sorted_values or not 0.0 <= fraction <= 1.0:
        raise ValueError("Need nonempty values and a quantile in [0, 1]")
    position = (len(sorted_values) - 1) * fraction
    lo = int(math.floor(position))
    hi = int(math.ceil(position))
    return sorted_values[lo] + (position - lo) * (sorted_values[hi] - sorted_values[lo])


def paired_analysis(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int | list[float]]:
    """Estimate mean paired delta, bootstrap 95% CI and sign-flip p-value.

    Assumes independent queries; the comparison is *paired* within each query.
    Sign-flip testing uses an exchangeability/symmetry null and Monte Carlo
    correction (extreme + 1) / (iterations + 1). Does not assert significance.
    """
    a, b = list(baseline), list(treatment)
    if len(a) != len(b) or not a:
        raise ValueError("Paired arrays must be nonempty and have equal lengths")
    if iterations < 100:
        raise ValueError("iterations must be >= 100")
    if any(not math.isfinite(float(v)) for v in a + b):
        raise ValueError("Nonfinite metric in paired values")

    differences = [float(new) - float(old) for old, new in zip(a, b)]
    n = len(differences)
    observed = fmean(differences)
    rng = random.Random(seed)
    bootstrap = []
    for _ in range(iterations):
        bootstrap.append(sum(differences[rng.randrange(n)] for __ in range(n)) / n)
    bootstrap.sort()
    ci = [percentile(bootstrap, 0.025), percentile(bootstrap, 0.975)]

    extreme = 0
    threshold = abs(observed) - 1e-15
    for _ in range(iterations):
        permuted = sum(d if rng.getrandbits(1) else -d for d in differences) / n
        if abs(permuted) >= threshold:
            extreme += 1
    p_value = (extreme + 1) / (iterations + 1)
    return {
        "n_queries": n,
        "baseline_mean": fmean(a),
        "reranked_mean": fmean(b),
        "mean_delta": observed,
        "bootstrap_95pct_ci": ci,
        "sign_flip_two_sided_p": p_value,
        "queries_improved": sum(d > 1e-12 for d in differences),
        "queries_tied": sum(abs(d) <= 1e-12 for d in differences),
        "queries_worsened": sum(d < -1e-12 for d in differences),
        "iterations": iterations,
        "seed": seed,
    }
