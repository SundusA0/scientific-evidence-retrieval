from pathlib import Path

import pytest
from sci_evidence.data import tiny_demo_dataset
from sci_evidence.serving import SearchEngine


def test_bm25_search_reuses_engine_and_returns_documents():
    engine = SearchEngine(tiny_demo_dataset().documents)
    a = engine.search("solar panels", top_k=2)
    assert a.method == "bm25"
    assert a.hits[0].doc_id == "d1"
    assert a.hits[0].title == "Solar panels"
    assert a.elapsed_ms >= 0
    assert a.candidate_k is None
    assert engine.modes == ["bm25"]


def test_bm25_empty_or_invalid_query():
    engine = SearchEngine(tiny_demo_dataset().documents)
    with pytest.raises(ValueError, match="nonempty"):
        engine.search("  ")
    with pytest.raises(ValueError, match="top_k"):
        engine.search("solar", top_k=0)
    with pytest.raises(ValueError, match="not loaded"):
        engine.search("solar", method="rerank")
    with pytest.raises(ValueError, match="method"):
        engine.search("solar", method="other")


def test_package_includes_html():
    path = Path(__file__).resolve().parents[1] / "src/sci_evidence/static/index.html"
    assert "Scientific Evidence" in path.read_text()
