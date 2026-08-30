"""Structured table extraction for paper PDFs (Phase-1 fallback for MinerU).

PyMuPDF's plain ``get_text`` flattens SOTA-comparison tables into an
unstructured number stream — column headers and row labels are dropped, so
"Restormer on CBSD68 at sigma=25" can no longer be mapped to the cell 31.79.
This module re-parses each detected table from ``find_tables``' cell grid (which
preserves the row/column alignment, including "-" placeholders) and re-emits it
as self-contained, searchable rows that spell out every "dataset x sigma ->
value" association.

Scope: the common "method x dataset x sigma" result-table shape (denoising /
restoration SOTA tables). Tables that don't match this shape are skipped; their
content is still covered by the prose chunks. MinerU would replace this
heuristic upstream.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A decimal number (a result value). Method citations are bracketed integers
# like "[101]", so a bare decimal is unambiguously a value, never a method name.
_NUM = re.compile(r"-?\d+\.\d+")
_SIGMA = re.compile(r"σ=(\d+)")
_CITATION_TAIL = re.compile(r"\s*\[\d+\]\s*$")
_CAPTION = re.compile(r"^Table\s+\d+[\.:]")
# First-column labels that are not dataset names.
_NON_DATA_LABELS = {"method", "dataset", "network", "dim", "patch size"}


@dataclass
class TableRow:
    """One (method, dataset) cell group recovered from a result table."""

    page: int
    method: str
    dataset: str
    sigma_values: list[tuple[str, str]]  # [(sigma, value), ...]
    caption: str = ""

    @property
    def text(self) -> str:
        cells = ", ".join(f"σ={s} PSNR {v}" for s, v in self.sigma_values)
        base = f"{self.method} on {self.dataset}: {cells}"
        return f"{self.caption}. {base}".strip() if self.caption else base


def extract_table_rows(path: str) -> list[TableRow]:
    """Recover structured (method x dataset x sigma) rows from a paper PDF."""
    import pymupdf

    doc = pymupdf.open(path)
    rows: list[TableRow] = []
    try:
        for page in doc:
            page_num = page.number + 1
            for table in page.find_tables().tables:
                caption = _find_caption(page, table.bbox)
                rows.extend(_parse_table(table.extract(), page_num, caption))
    finally:
        doc.close()
    return rows


def _find_caption(page, bbox) -> str:
    """Return the "Table N. ..." caption line just above a table's bbox."""
    import pymupdf

    x0, y0, x1, y1 = bbox
    rect = pymupdf.Rect(x0 - 10, max(0, y0 - 80), x1 + 10, y0 + 2)
    for ln in page.get_text("text", clip=rect).splitlines():
        ln = ln.strip()
        if _CAPTION.match(ln):
            return ln[:80]
    return ""


def _parse_table(data: list[list[str | None]], page: int, caption: str) -> list[TableRow]:
    if not data:
        return []

    # 1. Locate the sigma sub-header row and derive the column count from it.
    sigma_idx: int | None = None
    sigmas: list[str] = []
    n_datasets = 0
    for i, row in enumerate(data):
        toks: list[str] = []
        for cell in row:
            if cell:
                toks.extend(_SIGMA.findall(cell))
        if toks:
            sigmas = list(dict.fromkeys(toks))
            if not sigmas or len(toks) % len(sigmas) != 0:
                return []
            n_datasets = len(toks) // len(sigmas)
            sigma_idx = i
            break
    if sigma_idx is None or sigma_idx == 0 or n_datasets == 0:
        return []

    # 2. Dataset column labels from the row directly above the sigma row.
    header = [c or "" for c in (data[sigma_idx - 1] if sigma_idx > 0 else [])]
    datasets: list[str] = []
    for c in header:
        if c and c.lower() not in _NON_DATA_LABELS:
            datasets.append(_CITATION_TAIL.sub("", c).strip())
    if len(datasets) != n_datasets:
        return []

    vals_per_row = n_datasets * len(sigmas)

    # 3. Data rows after the sigma row.
    rows: list[TableRow] = []
    for row in data[sigma_idx + 1 :]:
        cells = [c for c in row if c is not None and c.strip()]
        if not cells:
            continue
        if len(cells) == 1:
            # Collapsed: the whole method list landed in the first column, one
            # "method  v1 v2 ... vN" per line.
            for line in cells[0].split("\n"):
                line = line.strip()
                if not line:
                    continue
                name, values = _split_name_values(line)
                if name and len(values) == vals_per_row:
                    rows.extend(_emit(name, values, datasets, sigmas, page, caption))
        else:
            # Column-split: method names in the first cell, one value group per
            # subsequent cell; the "\n" index aligns them across columns.
            names = cells[0].split("\n")
            value_cols = [c.split("\n") for c in cells[1:]]
            for j, raw_name in enumerate(names):
                name = raw_name.strip()
                if not name:
                    continue
                values: list[str] = []
                for col in value_cols:
                    if j < len(col):
                        values.extend(col[j].split())
                if len(values) == vals_per_row:
                    rows.extend(_emit(name, values, datasets, sigmas, page, caption))

    return rows


def _emit(
    name: str,
    values: list[str],
    datasets: list[str],
    sigmas: list[str],
    page: int,
    caption: str,
) -> list[TableRow]:
    out: list[TableRow] = []
    n_sigmas = len(sigmas)
    for ci, ds in enumerate(datasets):
        cells: list[tuple[str, str]] = []
        for si, s in enumerate(sigmas):
            idx = ci * n_sigmas + si
            cells.append((s, values[idx] if idx < len(values) else "-"))
        out.append(TableRow(page=page, method=name, dataset=ds, sigma_values=cells, caption=caption))
    return out


def _split_name_values(line: str) -> tuple[str, list[str]]:
    """Split "<method> <v1> <v2> ..." into (name, [values]).

    Values are decimal numbers or "-" placeholders; method names may carry a
    bracketed citation ("[101]"). Scan tokens left-to-right and treat the first
    "-" or bare decimal as the start of the value list.
    """
    tokens = line.split()
    for i, tok in enumerate(tokens):
        if tok == "-" or _NUM.match(tok):
            return " ".join(tokens[:i]).strip(), tokens[i:]
    return line, []
