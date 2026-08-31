"""PDF parsing (local PyMuPDF fallback path).

Design: the RAG pipeline prefers MinerU (remote, structure-rich) for authoritative
parsing; this PyMuPDF parser is the always-available fallback that turns a PDF into
per-page text. It is intentionally simple and dependency-light so it runs anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ParsedPage:
    page: int  # 1-based
    text: str


class PDFParseError(RuntimeError):
    """Raised when a PDF cannot be opened or parsed."""


class UnsupportedFormatError(RuntimeError):
    """Raised when a document's extension is not a supported upload format."""


# Extensions we accept for upload. Word `.doc` (the legacy binary format) is *not*
# supported — only `.docx` (the OOXML zip container) — because it needs no parser
# beyond the stdlib. Tell users to "save as .docx" instead.
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


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


def parse_docx(path: str) -> list[ParsedPage]:
    """Extract paragraph text from a Word ``.docx`` using only the stdlib.

    A ``.docx`` is a zip container; the prose lives in ``word/document.xml`` as
    ``<w:p>`` paragraphs of ``<w:t>`` text runs. We flatten those into one text
    block with paragraphs separated by blank lines, which the plain-text chunker
    then splits on. Formatting (bold, tables, images) is dropped — this is a
    lightweight text extractor, not a layout-preserving renderer.
    """
    import xml.etree.ElementTree as ET
    import zipfile

    try:
        with zipfile.ZipFile(path) as zf:
            raw = zf.read("word/document.xml")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise UnsupportedFormatError(f"cannot read {path} as a .docx: {exc}") from exc

    root = ET.fromstring(raw)
    paragraphs: list[str] = []
    for p in root.iter(_W_NS + "p"):
        runs = [t.text or "" for t in p.iter(_W_NS + "t")]
        line = "".join(runs).strip()
        if line:
            paragraphs.append(line)
    return [ParsedPage(page=1, text="\n\n".join(paragraphs))]


def _decode(data: bytes) -> str:
    """Decode bytes with a forgiving chain of encodings (UTF-8 first)."""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def parse_text(path: str) -> list[ParsedPage]:
    """Read a plain-text / markdown file as a single page of prose.

    Line endings are normalized to ``\\n`` so Windows-authored files (CRLF) chunk
    the same as Unix ones.
    """
    return [ParsedPage(page=1, text=_decode(Path(path).read_bytes()).replace("\r\n", "\n"))]


def parse_document(path: str) -> list[ParsedPage]:
    """Dispatch to the right parser by file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path)
    if suffix == ".docx":
        return parse_docx(path)
    if suffix in {".txt", ".md", ".markdown"}:
        return parse_text(path)
    raise UnsupportedFormatError(
        f"unsupported document format {suffix or '(none)'!r}; "
        f"supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )
