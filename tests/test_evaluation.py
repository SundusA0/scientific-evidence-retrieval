import math
import pytest

from sci_evidence.data import tiny_demo_dataset
from sci_evidence.evaluation import benchmark, evaluate_one
from sci_evidence.retrieval import BM25Retriever


def test_perfect_results_are_one():
    assert evaluate_one(["d1", "d2"], {"d1": 1, "d2": 1}, k=2) == {
        "ndcg@2": 1.0, "recall@2": 1.0
    }


def test_one_relevant_at_second_rank():
    scores = evaluate_one(["other", "d1"], {"d1": 1}, k=2)
    assert scores["ndcg@2"] == pytest.approx(1 / math.log2(3))
    assert scores["recall@2"] == 1


def test_no_relevant_in_top_k():
    assert evaluate_one(["other"], {"d1": 1}, k=1) == {
        "ndcg@1": 0, "recall@1": 0
    }


def test_offline_full_benchmark():
    dataset = tiny_demo_dataset()
    overall, per_query = benchmark(BM25Retriever(dataset.documents), dataset)
    assert len(per_query) == 2
    assert overall["recall@10"] == 1.0
    assert overall["ndcg@10"] == 1.0


def test_limit_must_be_positive():
    dataset = tiny_demo_dataset()
    with pytest.raises(ValueError):
        benchmark(BM25Retriever(dataset.documents), dataset, limit=0)
