"""Information-retrieval metrics using only positive judgments in SciFact."""

from __future__ import annotations

import math
from statistics import mean

from .data import SciFactDataset


def _dcg(gains: list[int]) -> float:
    return sum((2 ** g - 1) / math.log2(rank + 2) for rank, g in enumerate(gains))


def evaluate_one(results: list[str], relevant: dict[str, int], k: int = 10) -> dict[str, float]:
    if not relevant:
        raise ValueError("Expected at least one relevant document")
    gains = [relevant.get(doc_id, 0) for doc_id in results[:k]]
    ideal = sorted(relevant.values(), reverse=True)[:k]
    ndcg = _dcg(gains) / _dcg(ideal)
    recall = sum(1 for doc_id in results[:k] if relevant.get(doc_id, 0) > 0) / len(relevant)
    return {f"ndcg@{k}": ndcg, f"recall@{k}": recall}


def benchmark(retriever, dataset: SciFactDataset, k: int = 10,
              limit: int | None = None) -> tuple[dict[str, float], list[dict]]:
    items = list(dataset.queries.items())
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be >=1")
        items = items[:limit]
    if not items:
        raise ValueError("No evaluated queries")
    per_query = []
    for qid, query in items:
        ranked = retriever.search(query, top_k=k)
        metrics = evaluate_one([doc_id for doc_id, _ in ranked], dataset.qrels[qid], k=k)
        per_query.append({"query_id": qid, "query": query, **metrics})
    keys = (f"ndcg@{k}", f"recall@{k}")
    overall = {metric: mean(row[metric] for row in per_query) for metric in keys}
    return overall, per_query
