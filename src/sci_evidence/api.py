"""Optional FastAPI retrieval service. Install with: pip install -e '.[api]'"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:
    raise RuntimeError("API needs: python -m pip install -e '.[api]'") from exc

from .data import load_scifact
from .retrieval import BM25Retriever


@asynccontextmanager
async def lifespan(app: FastAPI):
    dataset = load_scifact(Path("data/scifact"), split="test")
    app.state.docs = {d.id: d for d in dataset.documents}
    app.state.retriever = BM25Retriever(dataset.documents)
    yield


app = FastAPI(title="Scientific Evidence Retrieval", version="0.1.0", lifespan=lifespan,
              description="Research prototype: ranks papers; does not validate scientific claims.")


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)


@app.get("/health")
def health():
    return {"status": "ok", "method": "bm25", "corpus": "BEIR SciFact"}


@app.post("/search")
def search(request: SearchRequest):
    ranked = app.state.retriever.search(request.query, top_k=request.top_k)
    documents = app.state.docs
    return {
        "query": request.query,
        "method": "bm25",
        "results": [
            {"rank": i, "id": doc_id, "score": score,
             "title": documents[doc_id].title, "text": documents[doc_id].text}
            for i, (doc_id, score) in enumerate(ranked, 1)
        ],
        "disclaimer": "Retrieval relevance is not a fact-check or scientific verification.",
    }
