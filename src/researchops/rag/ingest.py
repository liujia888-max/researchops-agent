"""End-to-end ingestion: document -> parse -> chunk -> embed -> Qdrant.

``ingest_document`` is the single entrypoint: it dispatches to the right parser by
file extension (PDF via PyMuPDF, Word/``.docx`` and plain text via the stdlib), runs
the matching chunker, then embeds and upserts into Qdrant. ``ingest_pdf`` is kept as
a thin alias for the CLI's PDF path.
"""

from __future__ import annotations

from pathlib import Path

from researchops.rag.chunking import chunk_pages, chunk_text
from researchops.rag.embedder import Embedder
from researchops.rag.models import Chunk
from researchops.rag.parser import parse_document
from researchops.rag.qdrant_store import QdrantStore
from researchops.rag.tables import extract_table_rows


def doc_id_from_path(path: str | Path) -> str:
    # Slugify the file name into a stable document id.
    stem = Path(path).stem
    return "".join(c if c.isalnum() else "_" for c in stem).strip("_") or "doc"


async def ingest_document(path: str | Path, *, batch_size: int = 16) -> int:
    """Ingest a document (pdf/docx/txt/md) and return the number of chunks upserted."""
    path = Path(path)
    doc_id = doc_id_from_path(path)
    pages = parse_document(str(path))

    if path.suffix.lower() == ".pdf":
        chunks = chunk_pages(pages, doc_id=doc_id)
        # Structured table rows (method x dataset x sigma) become their own chunks,
        # so "Restormer on CBSD68 at sigma=25" stays retrievable as one self-contained
        # unit instead of dissolving into prose number soup. PDF-only: the table
        # extractor relies on PyMuPDF's layout engine.
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
    else:
        # Non-PDF docs parse to a single page; chunk by paragraph instead of line.
        chunks = chunk_text(pages[0].text if pages else "", doc_id=doc_id)

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


async def ingest_pdf(path: str | Path, *, batch_size: int = 16) -> int:
    """Back-compatible alias: ingest one paper PDF."""
    return await ingest_document(path, batch_size=batch_size)
