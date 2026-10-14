"""Local search UI and JSON API. Install: python -m pip install -e '.[api]'"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field
    from typing import Literal
except ImportError as exc:
    raise RuntimeError("API needs: python -m pip install -e '.[api]'") from exc

from .data import load_scifact
from .serving import SearchEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_root = Path(os.getenv("SCI_EVIDENCE_DATA", "data/scifact"))
    try:
        # Split only determines which judged queries would be loaded; the UI
        # uses the scientific abstracts, not test judgments.
        dataset = load_scifact(data_root, split="test")
    except FileNotFoundError as exc:
        raise RuntimeError("SciFact dataset missing. Run: python -m sci_evidence download") from exc
    model_path = os.getenv("SCI_EVIDENCE_RERANKER") or None
    device = os.getenv("SCI_EVIDENCE_DEVICE", "cpu")
    app.state.engine = SearchEngine(dataset.documents, reranker_model=model_path, device=device)
    yield


app = FastAPI(
    title="Scientific Evidence Search", version="0.2.0", lifespan=lifespan,
    description="Research demo: retrieval relevance does not verify scientific claims."
)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    method: Literal["bm25", "rerank"] = "bm25"


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html",
                        media_type="text/html")


@app.get("/health")
def health():
    engine: SearchEngine = app.state.engine
    return {"status": "ok", "corpus": "BEIR SciFact", "documents": len(engine.docs),
            "methods": engine.modes, "device": engine.device_name if engine.model else "cpu",
            "reranker_model": engine.reranker_model}


@app.post("/search")
def search(request: SearchRequest):
    engine: SearchEngine = app.state.engine
    try:
        result = engine.search(request.query, method=request.method, top_k=request.top_k, candidate_k=20)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"query": request.query, "method": result.method,
            "elapsed_ms": round(result.elapsed_ms, 2), "candidate_k": result.candidate_k,
            "results": [{"rank": i, "id": hit.doc_id, "score": hit.score,
                         "title": hit.title, "text": hit.abstract}
                        for i, hit in enumerate(result.hits, 1)],
            "disclaimer": "Search ranks abstracts; it does not determine whether a claim is true."}
