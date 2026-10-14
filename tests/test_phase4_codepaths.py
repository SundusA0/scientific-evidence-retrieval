from sci_evidence.data import tiny_demo_dataset
from sci_evidence.serving import SearchEngine


def test_reranker_endpoint_candidate_pool_and_scores(monkeypatch):
    engine = SearchEngine(tiny_demo_dataset().documents)
    # Stub ranking only; no Hugging Face downloads are needed in the CI suite.
    engine.model = object()
    observed = {}

    def fake_rerank(query, candidates, top_k):
        observed['candidates'] = candidates
        return [(candidates[-1][0], 2.5), (candidates[0][0], 1.1)][:top_k]

    monkeypatch.setattr(engine, '_rerank', fake_rerank)
    out = engine.search('caffeine sleep', method='rerank', top_k=2, candidate_k=5)
    assert out.method == 'rerank'
    assert len(observed['candidates']) == 5
    assert out.hits[0].score == 2.5
    assert out.candidate_k == 5
