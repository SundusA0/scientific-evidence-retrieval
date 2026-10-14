"""Deterministic BM25 and optional Sentence Transformers dense retrieval."""

from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import math
import re
from pathlib import Path

from .data import Document

WORD_PATTERN = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return WORD_PATTERN.findall(text.lower())


class BM25Retriever:
    """BM25 with smoothed positive IDF, k1=1.5 and b=0.75."""

    def __init__(self, docs: list[Document], k1: float = 1.5, b: float = 0.75):
        if not docs:
            raise ValueError("Corpus cannot be empty")
        self.docs = docs
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = []
        for i, doc in enumerate(docs):
            words = tokenize(doc.full_text)
            self.lengths.append(len(words))
            for term, freq in Counter(words).items():
                self.postings[term].append((i, freq))
        self.avg_len = sum(self.lengths) / len(self.lengths) or 1.0

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k must be >=1")
        scores: dict[int, float] = defaultdict(float)
        n = len(self.docs)
        for word in set(tokenize(query)):
            postings = self.postings.get(word, [])
            df = len(postings)
            if not df:
                continue
            idf = math.log1p((n - df + 0.5) / (df + 0.5))
            for i, tf in postings:
                length_ratio = self.lengths[i] / self.avg_len
                norm = tf + self.k1 * (1 - self.b + self.b * length_ratio)
                scores[i] += idf * (tf * (self.k1 + 1)) / norm
        ranked = heapq.nlargest(min(top_k, n), scores.items(),
                               key=lambda item: (item[1], -item[0]))
        # Deterministic zero-score ties when a query matches fewer than k documents.
        if len(ranked) < min(top_k, n):
            used = {i for i, _ in ranked}
            for i in range(n):
                if i not in used:
                    ranked.append((i, 0.0))
                    if len(ranked) >= min(top_k, n):
                        break
        return [(self.docs[i].id, float(score)) for i, score in ranked]


class DenseRetriever:
    """Cosine-similarity retriever; model download happens on first use."""

    def __init__(self, docs: list[Document], model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str | None = None, batch_size: int = 32):
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Dense retrieval needs: python -m pip install -e '.[ml]'") from exc
        if not docs:
            raise ValueError("Corpus cannot be empty")
        self.np = np
        self.docs = docs
        self.model_name = model_name
        self.batch_size = batch_size
        print(f"Loading embedding model {model_name} (first run downloads model weights)...")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"Encoding {len(docs)} documents; on a laptop this can take a while...")
        # Encodings are computed fresh each run to avoid mismatching stale caches.
        self.doc_embeddings = np.asarray(self.model.encode(
            [d.full_text for d in docs], batch_size=batch_size,
            normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if top_k < 1:
            raise ValueError("top_k must be >=1")
        vector = self.np.asarray(self.model.encode([query],
            normalize_embeddings=True, show_progress_bar=False)[0], dtype=self.np.float32)
        scores = self.doc_embeddings @ vector
        # For the small SciFact corpus a complete sort is simple and reproducible.
        order = self.np.argsort(-scores, kind="stable")[:top_k]
        return [(self.docs[int(i)].id, float(scores[int(i)])) for i in order]
