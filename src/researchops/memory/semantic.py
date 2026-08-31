"""Semantic memory: recall by meaning (embedding cosine similarity), not keywords.

The v1 ``SqliteMemoryStore`` recalls lexically — an entry matches only if it shares
tokens with the query. ``SemanticMemoryStore`` keeps the same durable SQLite storage
but ranks by dense-embedding cosine similarity, so "denoising accuracy went up" can be
recalled by a query like "did the model get better?" that shares no words.

It reuses the RAG ``Embedder`` (bge-m3 over the remote inference service), so there is
no new infrastructure. Dense vectors are cached in memory and recomputed lazily for any
entry loaded from a previous session.
"""

from __future__ import annotations

import math
from typing import Protocol

from researchops.memory.store import MemoryEntry, SqliteMemoryStore


class _Embedder(Protocol):
    """The slice of ``Embedder`` semantic recall needs (so a fake can stand in)."""

    async def embed(
        self, texts: list[str]
    ) -> list[tuple[list[float], list[int], list[float]]]: ...


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SemanticMemoryStore:
    """Durable SQLite storage with dense-embedding recall.

    ``remember`` persists via ``SqliteMemoryStore`` and caches the dense vector;
    ``recall`` embeds the query and ranks every entry by cosine similarity. Structurally
    satisfies the ``MemoryStore`` protocol, so it drops into ``make_memory_search_tool``.
    """

    def __init__(self, embedder: _Embedder, *, path: str | None = None) -> None:
        self._embedder = embedder
        self._db = SqliteMemoryStore(path)
        self._vectors: dict[int, list[float]] = {}

    async def remember(self, text: str, *, kind: str = "note") -> int:
        entry_id = await self._db.remember(text, kind=kind)
        (dense, _idx, _vals) = (await self._embedder.embed([text]))[0]
        self._vectors[entry_id] = dense
        return entry_id

    async def recall(self, query: str, *, k: int = 5) -> list[MemoryEntry]:
        entries = await self._db.list_entries()
        if not entries:
            return []
        (query_vec, _idx, _vals) = (await self._embedder.embed([query]))[0]
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in entries:
            vector = self._vectors.get(entry.id)
            if vector is None:
                (vector, _i, _v) = (await self._embedder.embed([entry.text]))[0]
                self._vectors[entry.id] = vector
            scored.append((_cosine(query_vec, vector), entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[:k]]

    async def close(self) -> None:
        await self._db.close()
