"""Hybrid retrieval: dense + sparse search fused with RRF, then cross-encoder rerank.

Pipeline:
1. Embed the query with bge-m3 -> dense vector + sparse (lexical) weights.
2. Run both branches against Qdrant (dense cosine + sparse BM25-style).
3. Fuse the two ranked lists with Reciprocal Rank Fusion (RRF).
4. Rerank the fused top-k with bge-reranker-v2-m3 for final precision.
"""

from __future__ import annotations

from qdrant_client import AsyncQdrantClient, models

from researchops.config import Settings
from researchops.rag.embedder import Embedder
from researchops.rag.models import Chunk, RetrievedChunk
from researchops.rag.qdrant_store import QdrantStore

# RRF constant: 60 is a common default that matches Qdrant's own RRF behavior.
RRF_K = 60


class Retriever:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._embedder = Embedder(self._settings)
        self._store = QdrantStore(self._settings)
        self._client = AsyncQdrantClient(url=self._settings.qdrant_url, check_compatibility=False)

    async def fuse(self, query: str, *, top_k: int | None = None) -> list[Chunk]:
        """Hybrid RRF fusion only (no rerank) — the candidate pool.

        Split out so the eval harness can compare the fused ordering against the
        reranked one (the "rerank delta" metric).
        """
        settings = self._settings
        top_k = top_k or settings.retrieval_top_k

        (dense, sparse_idx, sparse_val) = (await self._embedder.embed([query]))[0]

        # Hybrid query: dense + sparse branches fused by RRF inside Qdrant.
        result = await self._client.query_points(
            collection_name=self._store.collection,
            prefetch=[
                models.Prefetch(query=dense, using="dense", limit=top_k),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_idx, values=sparse_val),
                    using="sparse",
                    limit=top_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
        )

        return [_point_to_chunk(p) for p in result.points]

    async def retrieve(
        self, query: str, *, top_k: int | None = None, rerank_top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Fuse a candidate pool, then rerank it down to the final top-k.

        ``top_k`` sizes the fusion candidate pool; ``rerank_top_k`` is the final
        answer size (defaults to the config's ``retrieval_rerank_top_k``).
        """
        candidates = await self.fuse(query, top_k=top_k)
        rerank_top_k = rerank_top_k or self._settings.retrieval_rerank_top_k
        return await self.rerank(query, candidates, top_k=rerank_top_k)

    async def rerank(
        self, query: str, candidates: list[Chunk], *, top_k: int | None = None
    ) -> list[RetrievedChunk]:
        """Rerank candidate chunks with the cross-encoder and return top-k."""
        if not candidates:
            return []
        top_k = top_k or self._settings.retrieval_rerank_top_k
        scores = await self._embedder.rerank(query, [c.text for c in candidates])
        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda t: t[1], reverse=True
        )[:top_k]
        return [RetrievedChunk(chunk=c, score=float(s)) for c, s in ranked]

    async def close(self) -> None:
        await self._client.close()
        await self._store.close()


def _point_to_chunk(point: models.ScoredPoint) -> Chunk:
    payload = point.payload or {}
    return Chunk(
        text=payload.get("text", ""),
        doc_id=str(payload.get("doc_id", "")),
        page=int(payload.get("page", 0)),
        section=str(payload.get("section", "")),
        chunk_index=int(payload.get("chunk_index", 0)),
    )
