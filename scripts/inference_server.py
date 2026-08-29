"""Remote GPU inference service: bge-m3 embedding (dense + sparse) + bge-reranker-v2-m3.

Serves the models used by the RAG subsystem. Runs on autodl-new5 (RTX 5090).
The local machine has no GPU, so all embedding/reranking calls go over HTTP here.

Endpoints:
  POST /v1/embeddings      -> {"dense": [...], "sparse": {"indices": [...], "values": [...]}}
  POST /v1/rerank          -> {"scores": [...], "order": [...]}
  GET  /health             -> {"status": "ok"}
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Use hf-mirror for model downloads on the remote (github/hf are slow there).
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# Load strictly from the local cache. The download script intentionally skips
# junk files (.DS_Store etc.) via ignore_patterns, so a snapshot_download
# completeness check would try to fetch those skipped files and 403.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_EMBED = "BAAI/bge-m3"
MODEL_RERANK = "BAAI/bge-reranker-v2-m3"

_embed_model = None
_rerank_model = None
_rerank_lock = threading.Lock()


def _local_snapshot(repo_id: str) -> str:
    """Resolve a repo id to its on-disk HF cache snapshot path.

    FlagEmbedding only skips its internal `snapshot_download` (which runs a
    completeness check and re-fetches ignored files like `.DS_Store`) when the
    path passed in already exists. Resolving the snapshot ourselves avoids that.
    """
    hf_home = os.environ.get("HF_HOME") or "/root/autodl-tmp/hf_cache"
    repo_dir = "models--" + repo_id.replace("/", "--")
    snapshots = os.path.join(hf_home, "hub", repo_dir, "snapshots")
    commits = sorted(os.listdir(snapshots))
    if not commits:
        raise RuntimeError(f"no cached snapshot for {repo_id} under {snapshots}")
    return os.path.join(snapshots, commits[-1])


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_embed() -> None:
    global _embed_model
    from FlagEmbedding import BGEM3FlagModel

    _embed_model = BGEM3FlagModel(_local_snapshot(MODEL_EMBED), use_fp16=True, devices=_device())


def _ensure_reranker() -> None:
    """Lazy-load the reranker on first use (its weights may still be downloading)."""
    global _rerank_model
    if _rerank_model is not None:
        return
    with _rerank_lock:
        if _rerank_model is None:
            from FlagEmbedding import FlagReranker

            _rerank_model = FlagReranker(_local_snapshot(MODEL_RERANK), use_fp16=True, devices=_device())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Only the embedding model is required at startup; the reranker loads on demand.
    _load_embed()
    yield


app = FastAPI(title="researchops-inference", version="0.1.0", lifespan=lifespan)


class EmbedRequest(BaseModel):
    inputs: list[str] = Field(min_length=1)


class EmbedResponse(BaseModel):
    dense: list[list[float]]
    sparse: list[dict[str, list[int | float]]]


class RerankRequest(BaseModel):
    query: str
    passages: list[str] = Field(min_length=1)


class RerankResponse(BaseModel):
    scores: list[float]
    order: list[int]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/embeddings", response_model=EmbedResponse)
def embeddings(req: EmbedRequest) -> EmbedResponse:
    out = _embed_model.encode(
        req.inputs, return_dense=True, return_sparse=True
    )
    dense = [v.tolist() for v in out["dense_vecs"]]
    sparse = [_lexical_to_sparse(w) for w in out["lexical_weights"]]
    return EmbedResponse(dense=dense, sparse=sparse)


@app.post("/v1/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    _ensure_reranker()
    pairs = [[req.query, p] for p in req.passages]
    scores = _rerank_model.compute_score(pairs, normalize=True)
    scores = [float(s) for s in scores]
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return RerankResponse(scores=scores, order=order)


def _lexical_to_sparse(weights: dict) -> dict[str, list[int | float]]:
    """Convert bge-m3 lexical_weights {token_id: weight} -> Qdrant sparse vector."""
    if not weights:
        return {"indices": [], "values": []}
    # Keys are token ids (may be numpy ints); values are floats.
    idx = [int(k) for k in weights]
    vals = [float(weights[k]) for k in weights]
    # bge-m3 expects float16 weights; Qdrant needs float32.
    arr = np.array(vals, dtype=np.float32)
    arr[arr < 0.0] = 0.0
    return {"indices": idx, "values": arr.tolist()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
