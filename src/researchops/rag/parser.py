"""PDF parsing (local PyMuPDF fallback path).

Design: the RAG pipeline prefers MinerU (remote, structure-rich) for authoritative
parsing; this PyMuPDF parser is the always-available fallback that turns a PDF into
per-page text. It is intentionally simple and dependency-light so it runs anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ParsedPage:
    page: int  # 1-based
    text: str


class PDFParseError(RuntimeError):
    """Raised when a PDF cannot be opened or parsed."""


def parse_pdf(path: str) -> list[ParsedPage]:
    """Extract text per page from a PDF using PyMuPDF."""
    import pymupdf  # PyMuPDF

    try:
        doc: Any = pymupdf.open(path)  # type: ignore[no-untyped-call]
    except Exception as exc:  # noqa: BLE001 — surface a clean error to callers
        raise PDFParseError(f"cannot open PDF {path}: {exc}") from exc

    pages: list[ParsedPage] = []
    try:
        for i, page in enumerate(doc):
            pymupdf_page: Any = page
            text = pymupdf_page.get_text("text")
            pages.append(ParsedPage(page=i + 1, text=text))
    finally:
        doc.close()
    return pages
