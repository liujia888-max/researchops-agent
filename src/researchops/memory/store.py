"""Episodic memory store: ``remember`` writes, ``recall`` retrieves relevant entries.

The v1 backend is SQLite with lexical recall — every entry is scored by how many of the
query's tokens it contains, so ``recall`` returns the most on-topic past entries without
needing an embedding service. It is deliberately self-contained (no ORM coupling to the
experiment DB) and unit-testable offline. A semantic backend can implement the same
``MemoryStore`` protocol over the existing embedder/Qdrant later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import aiosqlite

from researchops.config import get_settings

_TOKEN = re.compile(r"[a-z0-9]+")
_CJK = re.compile(r"[一-鿿]+")


def _tokens(text: str) -> set[str]:
    """Lowercased latin tokens plus CJK character bigrams.

    Latin words are matched whole; Chinese has no whitespace word boundaries in the
    raw text, so each CJK run is shingled into overlapping 2-char bigrams. A query
    like 「图像去噪」 then matches stored entries sharing the same character pairs,
    without pulling in a tokenizer dependency.
    """
    lowered = text.lower()
    tokens = set(_TOKEN.findall(lowered))
    for run in _CJK.findall(lowered):
        tokens.update(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


@dataclass(frozen=True)
class MemoryEntry:
    """One remembered fact: free text plus a coarse kind tag."""

    id: int
    text: str
    kind: str  # "experiment" | "note" | ...
    created_at: str
    task: str = ""  # the original user task, when the entry came from an agent run


class MemoryStore(Protocol):
    """The memory interface both backends implement (lexical now, semantic later)."""

    async def remember(self, text: str, *, kind: str = "note", task: str = "") -> int: ...

    async def recall(self, query: str, *, k: int = 5) -> list[MemoryEntry]: ...

    async def close(self) -> None: ...


class SqliteMemoryStore:
    """SQLite-backed episodic memory with lexical recall.

    ``path`` defaults to the configured ``Settings.memory_path`` (a git-ignored file
    under ``.researchops/``). The table is created lazily on first use, so the store
    can be constructed without touching disk.
    """

    def __init__(self, path: str | None = None) -> None:
        self._path = path if path is not None else get_settings().memory_path
        self._conn: aiosqlite.Connection | None = None

    async def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            self._conn = await aiosqlite.connect(self._path)
            await self._conn.execute(
                "CREATE TABLE IF NOT EXISTS memories ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "text TEXT NOT NULL, "
                "kind TEXT NOT NULL DEFAULT 'note', "
                "created_at TEXT NOT NULL)"
            )
            # v2 schema: the original task gets its own column so a history UI can
            # list "questions asked" without parsing the free text. v1 databases
            # (task stored inside text as "Task: ...\nResult: ...") are migrated in place.
            cursor = await self._conn.execute("PRAGMA table_info(memories)")
            columns = {row[1] for row in await cursor.fetchall()}
            if "task" not in columns:
                await self._conn.execute(
                    "ALTER TABLE memories ADD COLUMN task TEXT NOT NULL DEFAULT ''"
                )
            await self._conn.commit()
        return self._conn

    async def remember(self, text: str, *, kind: str = "note", task: str = "") -> int:
        db = await self._db()
        now = datetime.now(UTC).isoformat()
        cursor = await db.execute(
            "INSERT INTO memories (text, kind, task, created_at) VALUES (?, ?, ?, ?)",
            (text, kind, task, now),
        )
        await db.commit()
        return int(cursor.lastrowid or 0)

    async def recall(self, query: str, *, k: int = 5) -> list[MemoryEntry]:
        db = await self._db()
        cursor = await db.execute("SELECT id, text, kind, task, created_at FROM memories")
        rows = await cursor.fetchall()

        query_tokens = _tokens(query)
        scored: list[tuple[float, MemoryEntry]] = []
        for row in rows:
            entry = MemoryEntry(
                id=int(row[0]),
                text=str(row[1]),
                kind=str(row[2]),
                task=str(row[3]),
                created_at=str(row[4]),
            )
            entry_tokens = _tokens(f"{entry.task} {entry.text}")
            if not entry_tokens or not query_tokens:
                continue
            overlap = len(query_tokens & entry_tokens)
            if overlap == 0:
                continue
            # Fraction of query terms covered — favors entries that match many terms
            # without rewarding raw length.
            scored.append((overlap / len(query_tokens), entry))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[:k]]

    async def list_entries(self) -> list[MemoryEntry]:
        """Return every stored entry (in insertion order).

        Used by the semantic backend to re-score entries with embeddings and by the
        web UI to browse the memory log (past tasks and their reports).
        """
        db = await self._db()
        cursor = await db.execute("SELECT id, text, kind, task, created_at FROM memories")
        rows = await cursor.fetchall()
        return [
            MemoryEntry(
                id=int(r[0]),
                text=str(r[1]),
                kind=str(r[2]),
                task=str(r[3]),
                created_at=str(r[4]),
            )
            for r in rows
        ]

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
