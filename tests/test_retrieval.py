from sci_evidence.data import tiny_demo_dataset
from sci_evidence.retrieval import BM25Retriever, tokenize


def test_tokenize():
    assert tokenize("ML, Solar PANELS!") == ["ml", "solar", "panels"]


def test_bm25_retrieves_expected_toy_document():
    ds = tiny_demo_dataset()
    retriever = BM25Retriever(ds.documents)
    assert retriever.search("sunlight panels", top_k=1)[0][0] == "d1"
    assert retriever.search("caffeine delay sleep", top_k=1)[0][0] == "d2"


def test_bm25_returns_zero_score_ties_deterministically():
    ds = tiny_demo_dataset()
    retriever = BM25Retriever(ds.documents)
    assert retriever.search("unobtainium", top_k=3) == [
        ("d1", 0.0), ("d2", 0.0), ("d3", 0.0)
    ]


def test_bm25_rejects_invalid_k():
    import pytest
    with pytest.raises(ValueError):
        BM25Retriever(tiny_demo_dataset().documents).search("anything", 0)
