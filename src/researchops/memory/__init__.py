"""Long-term (episodic) memory for the agent.

The agent's long-term memory is a queryable log of what it has learned — experiment
outcomes, metrics, and decisions — so later tasks can build on earlier ones instead of
re-deriving them. ``SqliteMemoryStore`` is the zero-infra backend (lexical recall, no
embedding service needed); a semantic backend (embedding over Qdrant) is the documented
upgrade path when a GPU/embedding service is available.
"""

from researchops.memory.store import MemoryEntry, MemoryStore, SqliteMemoryStore

__all__ = ["MemoryEntry", "MemoryStore", "SqliteMemoryStore"]
