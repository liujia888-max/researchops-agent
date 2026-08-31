"""Tests for semantic memory (embedding-based recall, no keyword overlap needed)."""

from __future__ import annotations

from researchops.memory import SemanticMemoryStore


class _ScriptedEmbedder:
    """Returns a fixed vector per text, so recall ordering is fully deterministic."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    async def embed(self, texts: list[str]) -> list[tuple[list[float], list[int], list[float]]]:
        return [(self._vectors[t], [], []) for t in texts]


async def test_semantic_recall_ranks_by_meaning_not_keywords(tmp_path) -> None:
    # The query "query" shares no words with either entry, so lexical recall would miss;
    # semantic recall ranks by cosine distance to the embedding instead.
    vectors = {
        "image quality improved": [1.0, 0.0, 0.0],
        "wavelet transformer denoising": [0.0, 1.0, 0.0],
        "query": [1.0, 0.0, 0.0],  # closest to the first entry
    }
    store = SemanticMemoryStore(_ScriptedEmbedder(vectors), path=str(tmp_path / "m.db"))
    try:
        await store.remember("image quality improved")
        await store.remember("wavelet transformer denoising")
        hits = await store.recall("query")
        assert [h.text for h in hits] == [
            "image quality improved",
            "wavelet transformer denoising",
        ]
    finally:
        await store.close()


async def test_semantic_recall_respects_k_and_empty(tmp_path) -> None:
    vectors = {"entry": [1.0], "query": [1.0]}
    store = SemanticMemoryStore(_ScriptedEmbedder(vectors), path=str(tmp_path / "m.db"))
    try:
        assert await store.recall("query") == []  # empty before anything is remembered
        await store.remember("entry")
        assert len(await store.recall("query", k=1)) == 1
    finally:
        await store.close()


async def test_semantic_memory_is_durable_across_store_reopen(tmp_path) -> None:
    """Vectors are recomputed lazily for entries persisted by an earlier session."""
    vectors = {"entry": [1.0, 0.0], "query": [1.0, 0.0]}
    path = str(tmp_path / "m.db")
    store = SemanticMemoryStore(_ScriptedEmbedder(vectors), path=path)
    await store.remember("entry")
    await store.close()

    reopened = SemanticMemoryStore(_ScriptedEmbedder(vectors), path=path)
    try:
        hits = await reopened.recall("query")
        assert [h.text for h in hits] == ["entry"]
    finally:
        await reopened.close()
