"""Tests for the episodic memory store and its agent tool."""

from __future__ import annotations

from researchops.agent.tools import make_memory_search_tool
from researchops.memory import SqliteMemoryStore


async def test_remember_and_recall(tmp_path) -> None:
    store = SqliteMemoryStore(str(tmp_path / "memory.db"))
    try:
        await store.remember(
            "Experiment restormer_blind on CBSD68 sigma=25 PSNR 31.79", kind="experiment"
        )
        await store.remember("Baseline model_v3_rgb on CBSD68 sigma=25 PSNR 29.96")
        hits = await store.recall("restormer blind psnr")
        assert hits, "expected at least one hit"
        assert "31.79" in hits[0].text
        assert hits[0].kind == "experiment"
    finally:
        await store.close()


async def test_recall_ranks_more_matching_terms_first(tmp_path) -> None:
    store = SqliteMemoryStore(str(tmp_path / "memory.db"))
    try:
        await store.remember("wavelet")
        await store.remember("wavelet transformer detail")
        hits = await store.recall("wavelet transformer")
        assert [h.text for h in hits] == ["wavelet transformer detail", "wavelet"]
    finally:
        await store.close()


async def test_recall_returns_empty_on_no_match(tmp_path) -> None:
    store = SqliteMemoryStore(str(tmp_path / "memory.db"))
    try:
        await store.remember("GPU RTX 5090 utilization")
        assert await store.recall("quantum chemistry") == []
    finally:
        await store.close()


async def test_recall_respects_k(tmp_path) -> None:
    store = SqliteMemoryStore(str(tmp_path / "memory.db"))
    try:
        for i in range(5):
            await store.remember(f"psnr result {i}")
        assert len(await store.recall("psnr", k=3)) == 3
    finally:
        await store.close()


async def test_remember_returns_increasing_ids(tmp_path) -> None:
    store = SqliteMemoryStore(str(tmp_path / "memory.db"))
    try:
        first = await store.remember("first")
        second = await store.remember("second")
        assert second == first + 1
    finally:
        await store.close()


async def test_memory_search_tool_formats_results(tmp_path) -> None:
    store = SqliteMemoryStore(str(tmp_path / "memory.db"))
    try:
        await store.remember("Restormer CBSD68 sigma=25 PSNR 31.79", kind="experiment")
        tool = make_memory_search_tool(store)
        output = await tool.handler("restormer psnr")
        assert "31.79" in output
        assert "experiment" in output
        empty = await tool.handler("nothing here")
        assert empty == "No relevant past experiments or notes found in memory."
    finally:
        await store.close()
