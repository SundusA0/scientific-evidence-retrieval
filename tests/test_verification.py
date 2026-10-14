from copy import deepcopy
import pytest
from sci_evidence.verification import verify_report


def make_reports():
    base = {"split": "test", "parameters": {"limit": None}, "per_query": [],
            "metrics": {"ndcg@10": 0.5, "recall@10": 0.4}}
    rerank = {"split": "test", "parameters": {"limit": None}, "per_query": [],
              "bm25": base["metrics"].copy(),
              "bm25_plus_reranker": {"ndcg@10": 0.6, "recall@10": 0.5},
              "delta_reranker_minus_bm25": {"ndcg@10": 0.1, "recall@10": 0.1}}
    for i in range(300):
        base["per_query"].append({"query_id": str(i), "ndcg@10": 0.5, "recall@10": 0.4})
        rerank["per_query"].append({"query_id": str(i), "bm25": {"ndcg@10": 0.5, "recall@10": 0.4},
                                    "reranked": {"ndcg@10": 0.6, "recall@10": 0.5}})
    return base, rerank


def test_results_verified():
    a,b = make_reports()
    assert verify_report(a,b)["ndcg@10"]["delta"] == pytest.approx(0.1)


def test_rejects_bad_or_mismatched_results():
    a,b = make_reports()
    b["per_query"][0]["bm25"]["ndcg@10"] = 0.9
    with pytest.raises(ValueError, match="discrepancy"):
        verify_report(a,b)
    a,b = make_reports()
    b["per_query"][0]["query_id"] = "other"
    with pytest.raises(ValueError, match="different query IDs"):
        verify_report(a,b)
