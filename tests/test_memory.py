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


async def test_list_entries_returns_all_in_order(tmp_path) -> None:
    store = SqliteMemoryStore(str(tmp_path / "memory.db"))
    try:
        await store.remember("first")
        await store.remember("second", kind="experiment")
        entries = await store.list_entries()
        assert [e.text for e in entries] == ["first", "second"]
        assert entries[1].kind == "experiment"
        assert entries[0].id < entries[1].id
    finally:
        await store.close()


async def test_recall_matches_pure_chinese_via_bigrams(tmp_path) -> None:
    """Chinese queries recall by character bigrams — no latin keyword required."""
    store = SqliteMemoryStore(str(tmp_path / "memory.db"))
    try:
        await store.remember("图像去噪模型使用 MSE 与 SSIM 联合训练")
        hits = await store.recall("图像去噪的损失函数是什么")
        assert hits, "expected the Chinese entry to match via CJK bigrams"
        assert "MSE" in hits[0].text
    finally:
        await store.close()


async def test_remember_stores_task_column(tmp_path) -> None:
    store = SqliteMemoryStore(str(tmp_path / "memory.db"))
    try:
        await store.remember("report text", task="复现 Restormer CBSD68")
        entries = await store.list_entries()
        assert entries[0].task == "复现 Restormer CBSD68"
        assert entries[0].text == "report text"
        # task words participate in recall even though the text lacks them
        assert await store.recall("restormer") != []
    finally:
        await store.close()


async def test_v1_database_is_migrated_in_place(tmp_path) -> None:
    """A pre-v2 database (no task column) gains it on open and keeps recalling."""
    import sqlite3

    path = tmp_path / "memory.db"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE memories (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "text TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'note', created_at TEXT NOT NULL)"
    )
    db.execute(
        "INSERT INTO memories (text, kind, created_at) "
        "VALUES ('Task: old question\\nResult: old answer', 'note', '2026-01-01')"
    )
    db.commit()
    db.close()

    store = SqliteMemoryStore(str(path))
    try:
        await store.remember("new result", task="new question")
        entries = await store.list_entries()
        assert [e.task for e in entries] == ["", "new question"]
        # the v1 blob is still recallable through its text
        assert await store.recall("old question") != []
    finally:
        await store.close()
