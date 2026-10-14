"""Offline regression tests; download-free and no ML libraries imported."""

import pytest

from sci_evidence.data import Document, SciFactDataset, tiny_demo_dataset
from sci_evidence.retrieval import BM25Retriever
from sci_evidence.reranking import (
    candidate_upper_bound_recall, mine_train_pairs, partition_train_queries,
)


def fixture():
    docs = [Document(str(i), f"Paper {i}", f"Text on topic {i % 3}") for i in range(20)]
    queries = {str(i): f"What is topic {i % 3}?" for i in range(10)}
    qrels = {str(i): {str(i): 1} for i in range(10)}
    return SciFactDataset(docs, queries, qrels)


def test_query_split_disjoint_and_complete():
    ds = fixture()
    tr, val = partition_train_queries(ds)
    assert len(tr) + len(val) == len(ds.queries)
    assert set(tr).isdisjoint(val)
    assert set(tr) | set(val) == set(ds.queries)


def test_query_split_deterministic_and_seeded():
    ds = fixture()
    assert partition_train_queries(ds, seed=42) == partition_train_queries(ds, seed=42)
    assert partition_train_queries(ds, seed=42) != partition_train_queries(ds, seed=4)


def test_pair_mining_never_labels_known_positives_negative():
    ds = tiny_demo_dataset()
    pairs = mine_train_pairs(ds, ds.queries, BM25Retriever(ds.documents),
                             candidate_k=5, negatives_per_positive=2)
    assert pairs
    assert {(p.query_id, p.doc_id) for p in pairs if p.label == 1} == {('q1','d1'),('q2','d2')}
    for p in pairs:
        if p.label == 0:
            assert p.doc_id not in ds.qrels[p.query_id]
        assert p.document


def test_validation_labels_are_never_used_for_training():
    ds = fixture()
    tr, val = partition_train_queries(ds)
    pairs = mine_train_pairs(ds, tr, BM25Retriever(ds.documents), candidate_k=8)
    assert all(p.query_id in tr and p.query_id not in val for p in pairs)


def test_candidate_recall():
    assert candidate_upper_bound_recall(['a', 'b'], {'a': 1, 'c': 2}) == 0.5
    assert candidate_upper_bound_recall(['a', 'c'], {'a': 1, 'c': 2}) == 1
    with pytest.raises(ValueError):
        candidate_upper_bound_recall(['a'], {})


def test_invalid_train_split():
    with pytest.raises(ValueError):
        partition_train_queries(fixture(), validation_fraction=1.0)
