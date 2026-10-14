"""Local search UI and JSON API. Install: python -m pip install -e '.[api]'"""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import os
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, Field
    from starlette.concurrency import run_in_threadpool
    from typing import Literal
except ImportError as exc:
    raise RuntimeError("API needs: python -m pip install -e '.[api]'") from exc

from .data import load_scifact
from .serving import SearchEngine


def configured_rerank_capacity() -> int:
    """0 means unlimited; positive values enable immediate rejection at capacity."""
    raw = os.getenv("SCI_EVIDENCE_MAX_INFLIGHT_RERANK", "0")
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("SCI_EVIDENCE_MAX_INFLIGHT_RERANK must be an integer >= 0") from exc
    if value < 0:
        raise RuntimeError("SCI_EVIDENCE_MAX_INFLIGHT_RERANK must be >= 0")
    return value


def configured_rerank_wait() -> tuple[int, int]:
    """Maximum queue wait in ms, and maximum number of waiting requests.

    Both default to zero, preserving the previous immediate-rejection policy.
    """
    try:
        wait_ms = int(os.getenv("SCI_EVIDENCE_RERANK_WAIT_MS", "0"))
        max_queued = int(os.getenv("SCI_EVIDENCE_MAX_QUEUED_RERANK", "0"))
    except ValueError as exc:
        raise RuntimeError("Rerank wait/queue settings must be integers") from exc
    if not (0 <= wait_ms <= 5000):
        raise RuntimeError("SCI_EVIDENCE_RERANK_WAIT_MS must be 0..5000")
    if not (0 <= max_queued <= 1000):
        raise RuntimeError("SCI_EVIDENCE_MAX_QUEUED_RERANK must be 0..1000")
    return wait_ms, max_queued


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
    app.state.engine = SearchEngine(dataset.documents, reranker_model=model_path, device=device, batch_size=32)
    # Admission is managed on the event loop, before GPU work is dispatched
    # to the worker thread pool. No threads are held by queued requests.
    capacity = configured_rerank_capacity()
    wait_ms, max_queued = configured_rerank_wait()
    app.state.max_inflight_rerank = capacity
    app.state.rerank_slots = asyncio.Semaphore(capacity) if capacity else None
    app.state.rerank_wait_ms = wait_ms
    app.state.max_queued_rerank = max_queued
    app.state.rerank_waiting = 0
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
            "reranker_model": engine.reranker_model,
            "max_inflight_rerank": app.state.max_inflight_rerank,
            "rerank_wait_ms": app.state.rerank_wait_ms,
            "max_queued_rerank": app.state.max_queued_rerank,
            "waiting_rerank": app.state.rerank_waiting}


@app.post("/search")
async def search(request: SearchRequest):
    engine: SearchEngine = app.state.engine
    slots = app.state.rerank_slots if request.method == "rerank" else None
    acquired = False
    if slots is not None:
        if not slots.locked():
            # Asyncio fast path completes without suspension, so admission is
            # atomic with respect to other request coroutines on this loop.
            await slots.acquire()
            acquired = True
        elif (app.state.rerank_wait_ms > 0
              and app.state.rerank_waiting < app.state.max_queued_rerank):
            # Bounded FIFO semaphore waiting, with an explicit wait deadline.
            # Updating this counter has no intervening await and is atomic
            # with respect to other coroutines on the same event loop.
            app.state.rerank_waiting += 1
            try:
                await asyncio.wait_for(
                    slots.acquire(), timeout=app.state.rerank_wait_ms / 1000
                )
                acquired = True
            except TimeoutError as exc:
                raise HTTPException(status_code=503,
                                    detail="Reranker queue wait expired; retry later",
                                    headers={"Retry-After": "1"}) from exc
            finally:
                app.state.rerank_waiting -= 1
        else:
            raise HTTPException(status_code=503,
                                detail="Reranker queue full; retry later",
                                headers={"Retry-After": "1"})
    try:
        # Keep CPU/GPU work off the event loop, including in unlimited mode.
        try:
            result = await run_in_threadpool(engine.search, request.query,
                                             method=request.method, top_k=request.top_k,
                                             candidate_k=20)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if acquired:
            slots.release()
    return {"query": request.query, "method": result.method,
            "elapsed_ms": round(result.elapsed_ms, 2), "candidate_k": result.candidate_k,
            "results": [{"rank": i, "id": hit.doc_id, "score": hit.score,
                         "title": hit.title, "text": hit.abstract}
                        for i, hit in enumerate(result.hits, 1)],
            "disclaimer": "Search ranks abstracts; it does not determine whether a claim is true."}
