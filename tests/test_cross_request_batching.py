"""Pure tests for the cross-request GPU batching feasibility experiment."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "11_benchmark_cross_request_batching.py"
spec = importlib.util.spec_from_file_location("cross_request_batch", SCRIPT)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_group_ranges_cover_items_once():
    assert experiment.group_ranges(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert experiment.group_ranges(8, 8) == [(0, 8)]
    with pytest.raises(ValueError):
        experiment.group_ranges(10, 0)


def test_batch_summary_and_agreement():
    metrics = experiment.group_stats([100.0, 200.0], n_queries=4)
    assert metrics["inference_queries_per_second"] == pytest.approx(4000 / 300)
    assert metrics["amortized_ms_per_query"] == 75.0
    assert metrics["batch_p50_ms"] == 150.0
    with pytest.raises(ValueError):
        experiment.group_stats([float("nan")], n_queries=1)

    a = {0: ([0.3, 0.1], ["a", "b"]), 1: ([1.0, -1.0], ["x", "y"])}
    b = {0: ([0.30001, 0.1], ["a", "b"]), 1: ([0.8, -1.0], ["y", "x"])}
    result = experiment.compare_predictions(a, b)
    assert result["max_abs_logit_difference"] == pytest.approx(0.2)
    assert result["top_k_exact_match_fraction"] == 0.5


def test_group_inference_splits_scores_back_into_queries():
    from contextlib import nullcontext
    from types import SimpleNamespace

    class FakeTensor:
        def __init__(self, values):
            self.values = values
        def to(self, _device):
            return self
        def float(self):
            return self
        def reshape(self, _n):
            return self
        def cpu(self):
            return self
        def tolist(self):
            return list(self.values)

    seen = {}
    def tokenize(questions, passages, **kwargs):
        seen["queries"] = questions
        seen["texts"] = passages
        return {"input_ids": FakeTensor([1, 2, 3, 4])}

    engine = SimpleNamespace(
        device_name="cpu", device="cpu", max_length=256,
        tokenizer=tokenize,
        torch=SimpleNamespace(inference_mode=nullcontext),
        model=lambda **kw: SimpleNamespace(logits=FakeTensor([2.0, 1.0, -1.0, 3.0])),
    )
    items = [
        (0, "first", ["a", "b"], ["text a", "text b"]),
        (1, "second", ["c", "d"], ["text c", "text d"]),
    ]
    elapsed, outputs = experiment.infer_group(engine, items, top_k=1)
    assert elapsed >= 0
    assert seen["queries"] == ["first", "first", "second", "second"]
    assert outputs[0] == ([2.0, 1.0], ["a"])
    assert outputs[1] == ([-1.0, 3.0], ["d"])
