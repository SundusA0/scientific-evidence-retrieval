"""Train-side splits, BM25 hard-negative mining, and reranker evaluation helpers.

All training labels must come from the SciFact *train* split. Unjudged
candidates are treated as negative *training examples*, not established
scientific falsehoods; some may actually be relevant.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

from .data import Document, SciFactDataset
from .retrieval import BM25Retriever


@dataclass(frozen=True)
class Pair:
    query: str
    document: str
    label: float
    query_id: str
    doc_id: str


def partition_train_queries(
    dataset: SciFactDataset, *, seed: int = 42, validation_fraction: float = 0.1
) -> tuple[list[str], list[str]]:
    """Deterministic query-level split, using *only* SciFact train qrels."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    ids = sorted(dataset.queries)
    if len(ids) < 2:
        raise ValueError("Need at least two labelled training queries")
    random.Random(seed).shuffle(ids)
    n_val = min(len(ids) - 1, max(1, round(len(ids) * validation_fraction)))
    return ids[n_val:], ids[:n_val]


def mine_train_pairs(
    dataset: SciFactDataset,
    query_ids: Iterable[str],
    retriever: BM25Retriever,
    *,
    candidate_k: int = 50,
    negatives_per_positive: int = 3,
) -> list[Pair]:
    """Positive train qrels, plus hard negatives among BM25's top hits."""
    if candidate_k < 1:
        raise ValueError("candidate_k must be positive")
    if negatives_per_positive < 1:
        raise ValueError("negatives_per_positive must be positive")
    docs: dict[str, Document] = {doc.id: doc for doc in dataset.documents}
    result: list[Pair] = []
    for query_id in query_ids:
        query = dataset.queries[query_id]
        positives = set(dataset.qrels[query_id])
        known = sorted(d for d in positives if d in docs)
        if not known:
            continue
        for did in known:
            result.append(Pair(query, docs[did].full_text, 1.0, query_id, did))
        candidates = retriever.search(query, top_k=candidate_k)
        negative_ids = [did for did, _ in candidates if did not in positives]
        for did in negative_ids[: negatives_per_positive * len(known)]:
            result.append(Pair(query, docs[did].full_text, 0.0, query_id, did))
    return result


def candidate_upper_bound_recall(
    candidates: list[str], relevant: dict[str, int]
) -> float:
    """Fraction of known relevant docs in first-stage candidate pool."""
    if not relevant:
        raise ValueError("relevance judgments cannot be empty")
    return sum(did in candidates for did in relevant) / len(relevant)
