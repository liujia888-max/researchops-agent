"""HTTP client for the remote inference service (bge-m3 + reranker).

The local dev machine has no GPU, so embedding/reranking are delegated to a small
FastAPI service on the GPU host (see scripts/inference_server.py). This client is
deliberately thin: it speaks the same JSON shape and keeps the rest of the RAG
pipeline agnostic to where inference actually runs (remote GPU, or local CPU in
a pinch).

When the service is unreachable and ``rag_fallback_local`` is enabled (the default),
``embed``/``rerank`` degrade to a zero-dependency feature-hash embedder so RAG still
works offline — retrieval becomes lexical instead of semantic, but the pipeline does
not error out.
"""

from __future__ import annotations

import logging

import httpx

from researchops.config import Settings
from researchops.rag import local_embedder as _local
from researchops.rag.models import Chunk

_logger = logging.getLogger(__name__)
_fallback_warned = False


class InferenceError(RuntimeError):
    """Raised when the remote inference service is unreachable or errors."""


def _warn_fallback(exc: BaseException) -> None:
    """Log the fallback decision once per process (avoids spamming the log)."""
    global _fallback_warned
    if _fallback_warned:
        return
    _fallback_warned = True
    _logger.warning(
        "inference service unreachable (%s); falling back to offline feature-hash "
        "embedding (lexical retrieval). Point INFERENCE_BASE_URL at a bge-m3 service "
        "for semantic quality.",
        exc,
    )


class Embedder:
    """Talks to the bge-m3 / reranker service over HTTP, with an offline fallback."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._base = self._settings.inference_base_url.rstrip("/")

    async def embed(self, texts: list[str]) -> list[tuple[list[float], list[int], list[float]]]:
        """Return (dense, sparse_indices, sparse_values) for each text."""
        try:
            return await self._embed_http(texts)
        except (InferenceError, httpx.HTTPError) as exc:
            if not self._settings.rag_fallback_local:
                raise
            _warn_fallback(exc)
            return _local.embed(texts)

    async def _embed_http(
        self, texts: list[str]
    ) -> list[tuple[list[float], list[int], list[float]]]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{self._base}/v1/embeddings", json={"inputs": texts})
        if resp.status_code != 200:
            raise InferenceError(f"embedding failed {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        return [
            (dense, sparse["indices"], sparse["values"])
            for dense, sparse in zip(data["dense"], data["sparse"], strict=True)
        ]

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
        try:
            return await self._rerank_http(query, passages)
        except (InferenceError, httpx.HTTPError) as exc:
            if not self._settings.rag_fallback_local:
                raise
            _warn_fallback(exc)
            return _local.rerank(query, passages)

    async def _rerank_http(self, query: str, passages: list[str]) -> list[float]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base}/v1/rerank", json={"query": query, "passages": passages}
            )
        if resp.status_code != 200:
            raise InferenceError(f"rerank failed {resp.status_code}: {resp.text[:300]}")
        scores: list[float] = resp.json()["scores"]
        return scores
