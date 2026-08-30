"""End-to-end ingestion: PDF -> parse -> chunk -> embed -> Qdrant."""

from __future__ import annotations

from pathlib import Path

from researchops.rag.chunking import chunk_pages
from researchops.rag.embedder import Embedder
from researchops.rag.models import Chunk
from researchops.rag.parser import parse_pdf
from researchops.rag.qdrant_store import QdrantStore
from researchops.rag.tables import extract_table_rows


def _doc_id_from_path(path: Path) -> str:
    # Slugify the file name into a stable document id.
    stem = path.stem
    return "".join(c if c.isalnum() else "_" for c in stem).strip("_") or "doc"


async def ingest_pdf(path: str | Path, *, batch_size: int = 16) -> int:
    """Ingest one paper PDF and return the number of chunks upserted."""
    path = Path(path)
    doc_id = _doc_id_from_path(path)

    pages = parse_pdf(str(path))
    chunks = chunk_pages(pages, doc_id=doc_id)

    # Structured table rows (method x dataset x sigma) become their own chunks,
    # so "Restormer on CBSD68 at sigma=25" stays retrievable as one self-contained
    # unit instead of dissolving into prose number soup. Their chunk_index keeps
    # counting after the prose chunks, so every id stays unique.
    table_rows = extract_table_rows(str(path))
    base_index = len(chunks)
    for i, row in enumerate(table_rows):
        chunks.append(
            Chunk(
                text=row.text,
                doc_id=doc_id,
                page=row.page,
                section=row.caption or "Table",
                chunk_type="table_row",
                chunk_index=base_index + i,
            )
        )

    embedder = Embedder()
    store = QdrantStore()

    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        await embedder.embed_chunks(batch)
        await store.upsert(batch)
        total += len(batch)

    await store.close()
    return total
