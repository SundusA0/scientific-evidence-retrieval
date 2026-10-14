"""Fast, dependency-free checks for paired retrieval statistical analysis."""

import math

import pytest

from sci_evidence.statistics import paired_analysis, percentile


def test_percentile_interpolation():
    assert percentile([0, 1, 2], 0.25) == 0.5
    assert percentile([0, 1, 2], 1.0) == 2


def test_identical_systems_have_zero_delta_and_p_one():
    result = paired_analysis([0.2, 0.4, 0.1], [0.2, 0.4, 0.1], iterations=200)
    assert result["mean_delta"] == 0
    assert result["bootstrap_95pct_ci"] == [0, 0]
    assert result["sign_flip_two_sided_p"] == 1
    assert result["queries_tied"] == 3


def test_fixed_seed_is_reproducible():
    args = ([0.2, 0.3, 0.5, 0.8], [0.4, 0.25, 0.7, 0.6])
    assert paired_analysis(*args, iterations=250, seed=9) == paired_analysis(*args, iterations=250, seed=9)


def test_paired_delta_is_mean_difference():
    result = paired_analysis([0, 0, 0], [0.1, 0.2, 0.3], iterations=200)
    assert math.isclose(result["mean_delta"], 0.2)
    assert result["queries_improved"] == 3
    assert result["queries_worsened"] == 0


@pytest.mark.parametrize("x,y", [([], []), ([1], [1, 2])])
def test_reject_mismatched_or_empty_pairs(x, y):
    with pytest.raises(ValueError):
        paired_analysis(x, y)


def test_reject_nonfinite_scores():
    with pytest.raises(ValueError):
        paired_analysis([0.0], [float("nan")])


def test_reject_too_few_iterations():
    with pytest.raises(ValueError):
        paired_analysis([0.0], [0.1], iterations=99)
