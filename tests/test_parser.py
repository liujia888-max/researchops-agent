from __future__ import annotations

import zipfile

import pytest

from researchops.rag.parser import (
    UnsupportedFormatError,
    parse_document,
    parse_docx,
    parse_text,
)


def _make_docx(path: str) -> None:
    doc = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        "<w:p><w:r><w:t>Introduction</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>We propose a denoising method.</w:t></w:r></w:p>"
        "</w:body>"
        "</w:document>"
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("word/document.xml", doc)


def test_parse_text_reads_utf8(tmp_path) -> None:
    p = tmp_path / "notes.txt"
    p.write_text("hello world", encoding="utf-8")
    pages = parse_text(str(p))
    assert [pg.text for pg in pages] == ["hello world"]


def test_parse_text_decodes_gb18030(tmp_path) -> None:
    p = tmp_path / "notes.txt"
    p.write_bytes("复现".encode("gb18030"))
    pages = parse_text(str(p))
    assert pages[0].text == "复现"


def test_parse_docx_extracts_paragraphs(tmp_path) -> None:
    p = tmp_path / "paper.docx"
    _make_docx(str(p))
    pages = parse_docx(str(p))
    assert len(pages) == 1
    assert pages[0].page == 1
    assert "Introduction" in pages[0].text
    assert "We propose a denoising method." in pages[0].text


def test_parse_document_dispatches_by_extension(tmp_path) -> None:
    p = tmp_path / "a.md"
    p.write_text("# Title\n\nbody", encoding="utf-8")
    pages = parse_document(str(p))
    assert pages[0].text == "# Title\n\nbody"


def test_parse_document_rejects_unsupported(tmp_path) -> None:
    p = tmp_path / "legacy.doc"
    p.write_bytes(b"not a real docx")
    with pytest.raises(UnsupportedFormatError):
        parse_document(str(p))
