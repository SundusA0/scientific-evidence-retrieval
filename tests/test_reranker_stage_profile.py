"""Lightweight checks for the stage-profile summary; no model download needed."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "10_profile_reranker_stages.py"
spec = importlib.util.spec_from_file_location("reranker_profile", PATH)
assert spec is not None and spec.loader is not None
profiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiler)


def test_stage_summary():
    sample = {name: 1.0 for name in profiler.STAGE_KEYS}
    stats = profiler.summarize_stages([sample, {name: 3.0 for name in profiler.STAGE_KEYS}])
    assert stats["forward_ms"]["n_queries"] == 2
    assert stats["forward_ms"]["p50_ms"] == 2.0
    assert stats["total_ms"]["p95_ms"] == pytest.approx(2.9)


def test_stage_summary_rejects_incomplete_or_negative():
    with pytest.raises(ValueError):
        profiler.summarize_stages([])
    with pytest.raises(ValueError):
        profiler.summarize_stages([{"total_ms": 1.0}])
    with pytest.raises(ValueError):
        profiler.summarize_stages([{name: -1.0 for name in profiler.STAGE_KEYS}])
