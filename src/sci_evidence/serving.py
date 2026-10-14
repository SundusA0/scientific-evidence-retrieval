"""Reusable, measured retrieval pipeline for the local UI and latency experiment.

No test judgments are used to produce search results. Reranking follows the
same BM25 top-50 and max_length=256 design used by the Phase 2 evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .data import Document
from .retrieval import BM25Retriever


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    title: str
    abstract: str
    score: float


@dataclass(frozen=True)
class SearchOutput:
    method: str
    hits: list[SearchHit]
    elapsed_ms: float
    candidate_k: int | None


class SearchEngine:
    def __init__(self, docs: list[Document], *, reranker_model: str | None = None,
                 device: str = "cpu", batch_size: int = 16, max_length: int = 256):
        if not docs:
            raise ValueError("Cannot search an empty corpus")
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch_size and max_length must be positive")
        self.docs = {doc.id: doc for doc in docs}
        self.bm25 = BM25Retriever(docs)
        self.reranker_model = reranker_model
        self.device_name = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.device = None
        if reranker_model is not None:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            if device == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("Apple MPS not available. Try SCI_EVIDENCE_DEVICE=cpu")
            if device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA not available. Try SCI_EVIDENCE_DEVICE=cpu")
            if device not in ("cpu", "mps", "cuda"):
                raise ValueError("device must be one of cpu, mps, cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(reranker_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(reranker_model)
            if self.model.config.num_labels != 1:
                raise ValueError("Expected a cross-encoder with exactly one ranking logit")
            self.torch = torch
            self.device = torch.device(device)
            self.model.to(self.device).eval()

    @property
    def modes(self) -> list[str]:
        return ["bm25", "rerank"] if self.model is not None else ["bm25"]

    def _rerank(self, query: str, candidates: list[tuple[str, float]], top_k: int) -> list[tuple[str, float]]:
        assert self.model is not None and self.tokenizer is not None
        assert self.torch is not None and self.device is not None
        ids = [doc_id for doc_id, _ in candidates]
        scores: list[float] = []
        with self.torch.inference_mode():
            for begin in range(0, len(ids), self.batch_size):
                selected = ids[begin:begin + self.batch_size]
                tokens = self.tokenizer(
                    [query] * len(selected),
                    [self.docs[doc_id].full_text for doc_id in selected],
                    return_tensors="pt", padding=True, truncation=True,
                    max_length=self.max_length,
                )
                tokens = {k: v.to(self.device) for k, v in tokens.items()}
                logits = self.model(**tokens).logits.squeeze(-1)
                # Copy to CPU ensures asynchronous GPU work is complete before timing ends.
                scores.extend(logits.float().cpu().reshape(-1).tolist())
        indices = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
        return [(ids[i], float(scores[i])) for i in indices[:top_k]]

    def search(self, query: str, *, method: str = "bm25", top_k: int = 5,
               candidate_k: int = 50) -> SearchOutput:
        if not query or not query.strip():
            raise ValueError("Enter a nonempty query")
        if len(query) > 2000:
            raise ValueError("Query is too long (maximum 2000 characters)")
        if method not in ("bm25", "rerank"):
            raise ValueError("method must be 'bm25' or 'rerank'")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        if method == "rerank" and self.model is None:
            raise ValueError("Reranker is not loaded. Set SCI_EVIDENCE_RERANKER to a model path")
        if method == "rerank" and not top_k <= candidate_k <= len(self.docs):
            raise ValueError("Reranker candidate_k must be >= top_k and <= corpus size")
        start = perf_counter()
        if method == "bm25":
            ranked = self.bm25.search(query, top_k=top_k)
        else:
            candidates = self.bm25.search(query, top_k=candidate_k)
            ranked = self._rerank(query, candidates, top_k)
        elapsed_ms = (perf_counter() - start) * 1000.0
        hits = [SearchHit(doc_id=doc_id, title=self.docs[doc_id].title,
                          abstract=self.docs[doc_id].text, score=score)
                for doc_id, score in ranked]
        return SearchOutput(method=method, hits=hits, elapsed_ms=elapsed_ms,
                            candidate_k=candidate_k if method == "rerank" else None)
