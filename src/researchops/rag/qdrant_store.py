"""Qdrant store: collection creation + upsert for dual-vector chunks.

Each point carries:
- a dense vector  (bge-m3 `dense_vecs`, cosine)  -> named vector "dense"
- a sparse vector (bge-m3 `lexical_weights`)      -> named vector "sparse"
Hybrid retrieval fuses both branches with RRF in the retriever.
"""

from __future__ import annotations

import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from researchops.config import Settings
from researchops.rag.models import Chunk

DENSE_DIM = 1024  # bge-m3 dense embedding dimension

# Namespace for deterministic point UUIDs (uuid5 of the human-readable chunk id).
_POINT_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _point_id(chunk_id: str) -> str:
    """Map a human-readable chunk id to a valid Qdrant point id (UUID string)."""
    return str(uuid.uuid5(_POINT_NS, chunk_id))


class QdrantStore:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        # Server is pinned at 1.12.x while the client is newer; the version check
        # is a hard stop otherwise. The wire API we use is stable across both.
        self._client = AsyncQdrantClient(url=self._settings.qdrant_url, check_compatibility=False)
        self.collection = self._settings.qdrant_collection

    async def ensure_collection(self) -> None:
        exists = await self._client.collection_exists(self.collection)
        if exists:
            return
        await self._client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=DENSE_DIM,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(),
            },
        )

    async def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        await self.ensure_collection()
        points = []
        for c in chunks:
            if c.dense is None:
                raise ValueError(f"chunk {c.id} has no dense embedding; run embedder first")
            points.append(
                models.PointStruct(
                    id=_point_id(c.id),
                    vector={
                        "dense": c.dense,
                        "sparse": models.SparseVector(
                            indices=c.sparse_indices,
                            values=c.sparse_values,
                        ),
                    },
                    payload={
                        "doc_id": c.doc_id,
                        "page": c.page,
                        "section": c.section,
                        "chunk_type": c.chunk_type,
                        "chunk_index": c.chunk_index,
                        "text": c.text,
                    },
                )
            )
        await self._client.upsert(collection_name=self.collection, points=points)

    async def list_documents(self) -> list[dict[str, object]]:
        """Return the distinct ingested documents and their chunk counts.

        Used by the web UI's "document library" panel. Scrolls the collection's
        ``doc_id`` payloads (no vectors) and groups by document.
        """
        await self.ensure_collection()
        counts: dict[str, int] = {}
        offset: Any = None
        while True:
            points, offset = await self._client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=["doc_id"],
                with_vectors=False,
            )
            if not points:
                break
            for p in points:
                doc_id = str((p.payload or {}).get("doc_id", "?"))
                counts[doc_id] = counts.get(doc_id, 0) + 1
            if offset is None:
                break
        return [
            {"doc_id": doc_id, "chunks": n}
            for doc_id, n in sorted(counts.items())
        ]

    async def delete_document(self, doc_id: str) -> int:
        """Delete every chunk belonging to ``doc_id``; return the count removed."""
        await self.ensure_collection()
        flt = models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
        )
        count = await self._client.count(collection_name=self.collection, count_filter=flt)
        await self._client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=flt),
        )
        return int(count.count)

    async def close(self) -> None:
        await self._client.close()
