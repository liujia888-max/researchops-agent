from __future__ import annotations

from researchops.rag.chunking import _is_heading, chunk_pages, chunk_text
from researchops.rag.models import Chunk
from researchops.rag.parser import ParsedPage


def test_heading_detection_numbered() -> None:
    assert _is_heading("3.1 Method")
    assert _is_heading("4 Experiments")
    assert _is_heading("2 Related Work")


def test_heading_detection_uppercase() -> None:
    assert _is_heading("ABSTRACT")
    assert _is_heading("INTRODUCTION")


def test_non_heading_plain_text() -> None:
    assert not _is_heading("This is a normal sentence that is much longer than a heading.")
    assert not _is_heading("We propose a method that ends with a period.")


def test_chunk_pages_tags_section_and_page() -> None:
    pages = [
        ParsedPage(
            page=1,
            text="ABSTRACT\n\nWe propose a denoising method.\n\n1 Introduction\n\nImage denoising is important.",
        )
    ]
    chunks = chunk_pages(pages, doc_id="paper")
    assert chunks, "expected at least one chunk"
    # Provenance is carried on every chunk.
    for c in chunks:
        assert c.doc_id == "paper"
        assert c.page == 1
        assert c.id.startswith("paper:1:")


def test_chunk_never_exceeds_max_chars() -> None:
    # A single very long paragraph must be split, never exceed max_chars.
    long_para = "word " * 2000  # 10k chars
    pages = [ParsedPage(page=1, text=long_para)]
    chunks = chunk_pages(pages, doc_id="paper", max_chars=500, overlap_chars=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 500


def test_chunk_id_deterministic() -> None:
    c = Chunk(text="x", doc_id="doc", page=2, section="S", chunk_index=7)
    assert c.id == "doc:2:7"


def test_chunk_text_splits_on_paragraphs() -> None:
    text = "Introduction\n\nWe propose a method. It works well.\n\nConclusion\n\nWe show strong results."
    chunks = chunk_text(text, doc_id="doc")
    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert c.doc_id == "doc"
        assert c.page == 1
    # The "Conclusion" heading tags the chunk that follows it.
    assert any(c.section == "Conclusion" for c in chunks)


def test_chunk_text_never_exceeds_max_chars() -> None:
    long_para = "word " * 2000  # 10k chars, one paragraph
    chunks = chunk_text(long_para, doc_id="doc", max_chars=500, overlap_chars=50)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 500
