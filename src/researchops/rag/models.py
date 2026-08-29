"""RAG data model: a retrievable chunk of a paper, plus its embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """A self-contained text fragment from a paper, with provenance metadata."""

    text: str
    doc_id: str
    page: int
    section: str = ""
    chunk_index: int = 0

    # Embeddings are filled in by the embedder; kept None until then so chunking
    # and embedding stay decoupled (and testable without a GPU).
    dense: list[float] | None = None
    sparse_indices: list[int] = field(default_factory=list)
    sparse_values: list[float] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Deterministic point id in Qdrant: doc/page/chunk_index."""
        return f"{self.doc_id}:{self.page}:{self.chunk_index}"


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieval, with its fused/reranked score and provenance."""

    chunk: Chunk
    score: float
