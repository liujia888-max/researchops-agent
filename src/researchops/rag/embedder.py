"""HTTP client for the remote inference service (bge-m3 + reranker).

The local dev machine has no GPU, so embedding/reranking are delegated to a small
FastAPI service on the GPU host (see scripts/inference_server.py). This client is
deliberately thin: it speaks the same JSON shape and keeps the rest of the RAG
pipeline agnostic to where inference actually runs (remote GPU, or local CPU in
a pinch).
"""

from __future__ import annotations

import httpx

from researchops.config import Settings
from researchops.rag.models import Chunk


class InferenceError(RuntimeError):
    """Raised when the remote inference service is unreachable or errors."""


class Embedder:
    """Talks to the bge-m3 / reranker service over HTTP."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._base = self._settings.inference_base_url.rstrip("/")

    async def embed(self, texts: list[str]) -> list[tuple[list[float], list[int], list[float]]]:
        """Return (dense, sparse_indices, sparse_values) for each text."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self._base}/v1/embeddings", json={"inputs": texts})
        if resp.status_code != 200:
            raise InferenceError(f"embedding failed {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        out = []
        for dense, sparse in zip(data["dense"], data["sparse"], strict=True):
            out.append(
                (dense, sparse["indices"], sparse["values"])
            )
        return out

    async def embed_chunks(self, chunks: list[Chunk]) -> None:
        """Embed chunk texts in place, filling dense/sparse_* fields."""
        if not chunks:
            return
        results = await self.embed([c.text for c in chunks])
        for chunk, (dense, idx, vals) in zip(chunks, results, strict=True):
            chunk.dense = dense
            chunk.sparse_indices = idx
            chunk.sparse_values = vals

    async def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Return a relevance score per passage (higher = more relevant)."""
        if not passages:
            return []
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base}/v1/rerank", json={"query": query, "passages": passages}
            )
        if resp.status_code != 200:
            raise InferenceError(f"rerank failed {resp.status_code}: {resp.text[:300]}")
        scores: list[float] = resp.json()["scores"]
        return scores
