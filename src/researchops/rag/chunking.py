"""Structure-aware chunking for paper text.

Strategy (Phase 1, deterministic and testable without a GPU):
- Scan each page line by line (arxiv two-column PDFs rarely have blank lines
  between paragraphs, so blank-line splitting does not work here).
- Detect section headings with a small heuristic (numbered lines like
  "3.1. Title", all-caps short lines, or known section words) and tag the
  current section.
- Pack consecutive body lines into chunks up to `max_chars`, carrying the
  section + page provenance so citations stay traceable to the source page.
- Overlap by `overlap_chars` between adjacent chunks for cross-boundary context.

This is intentionally heuristic (papers are messy); MinerU would replace the
*parsing* step upstream, not this chunking policy.
"""

from __future__ import annotations

import re

from researchops.rag.models import Chunk
from researchops.rag.parser import ParsedPage

_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*[\.\)]?\s+)?(?:[A-Z][A-Za-z\s\-]{1,60})$"
)

# Common section titles seen in papers, matched case-insensitively at line start.
_SECTION_WORDS = (
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methodology",
    "approach",
    "proposed method",
    "experiments",
    "experimental results",
    "results",
    "ablation",
    "ablation studies",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "references",
    "appendix",
)


def _is_heading(block: str) -> bool:
    stripped = block.strip()
    if not stripped or len(stripped) > 70:
        return False
    # Explicit numbering like "3.1 Method", "3.1. Method", "4 Experiments".
    # Require a capital letter after the number so table digits ("2 2",
    # "31.57 dB") are not mistaken for section headings.
    if re.match(r"^\d+(?:\.\d+)*[\.\)]?\s+[A-Z]", stripped):
        return True
    # Short all-caps line with no sentence-ending punctuation. Must contain a
    # letter so pure number lines ("29.71") are excluded.
    if (
        len(stripped) >= 3
        and stripped.upper() == stripped
        and re.search(r"[A-Za-z]", stripped)
        and not re.search(r"[.!?,;:]$", stripped)
    ):
        return True
    # Known section title (title-case like "Abstract" / "Related Work").
    lower = stripped.lower()
    if any(lower == w or lower.startswith(w + " ") for w in _SECTION_WORDS):
        return True
    return False


def chunk_pages(
    pages: list[ParsedPage],
    *,
    doc_id: str,
    max_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """Turn parsed pages into provenance-tagged chunks."""
    chunks: list[Chunk] = []

    def make_chunk(text: str, page_num: int, cur_section: str) -> None:
        text = text.strip()
        if text:
            chunks.append(
                Chunk(
                    text=text,
                    doc_id=doc_id,
                    page=page_num,
                    section=cur_section,
                    chunk_index=len(chunks),
                )
            )

    section = ""
    for page in pages:
        # Body lines are sentence fragments in two-column PDFs; join with spaces
        # to reconstruct continuous prose. Blank lines are dropped.
        lines = [ln.strip() for ln in page.text.splitlines() if ln.strip()]

        buf: list[str] = []
        buf_len = 0

        def flush(page_num: int, cur_section: str) -> None:
            nonlocal buf, buf_len
            if not buf:
                return
            text = " ".join(buf)
            make_chunk(text, page_num, cur_section)
            # Keep a tail of the previous chunk as overlap context.
            tail = text[-overlap_chars:] if overlap_chars > 0 else ""
            buf = [tail] if tail else []
            buf_len = len(tail)

        for line in lines:
            if _is_heading(line):
                flush(page.page, section)
                section = line
                continue

            # A single line longer than max_chars (rare, e.g. an unbroken
            # paragraph) must still be split.
            if len(line) > max_chars:
                flush(page.page, section)
                for piece in _split_long_block(line, max_chars, overlap_chars):
                    make_chunk(piece, page.page, section)
                continue

            if buf and buf_len + len(line) + 1 > max_chars:
                flush(page.page, section)
            buf.append(line)
            buf_len += len(line) + 1

        flush(page.page, section)

    return chunks


def _split_long_block(block: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split one oversized block into <= max_chars pieces with word-boundary overlap."""
    if overlap_chars <= 0:
        return [block[i : i + max_chars] for i in range(0, len(block), max_chars)]

    pieces: list[str] = []
    start = 0
    n = len(block)
    while start < n:
        end = min(start + max_chars, n)
        pieces.append(block[start:end])
        if end >= n:
            break
        start = end - overlap_chars
    return pieces
