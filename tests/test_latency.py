import pytest
from sci_evidence.latency import summarize_latencies


def test_percentiles_are_interpolated_and_sequential_throughput():
    data = summarize_latencies([10.0, 20.0, 30.0, 40.0, 50.0])
    assert data["p50_ms"] == 30.0
    assert data["p95_ms"] == pytest.approx(48.0)
    assert data["sequential_queries_per_sec"] == pytest.approx(5 / 0.15)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        summarize_latencies([])
    with pytest.raises(ValueError):
        summarize_latencies([-1])
    with pytest.raises(ValueError):
        summarize_latencies([float("nan")])
