"""Cited answer generation + grounding validation.

The retrieval stage hands back a handful of chunks; this module turns them into
a *cited* answer whose every claim traces back to a source page:

1. `build_prompt` renders the retrieved chunks as numbered passages, each header
   carrying provenance (`doc_id`, page, section) so the model can cite `[n]`.
2. `generate_cited_answer` calls the LLM and parses the answer.
3. `extract_citation_indices` pulls the `[n]` markers out of the answer.
4. `resolve_citations` maps those markers back to the actual chunks (the
   "citation -> document -> page" traceability the plan calls for).
5. `validate_grounding` checks, deterministically, that every citation is
   in-range and lexically anchored in its chunk. This is a *cheap* sanity gate
   that runs without an LLM; the stronger NLI-style faithfulness score belongs
   to the RAGAS eval (see ``src/researchops/eval/``).

Everything here is dependency-injected (the LLM is passed in), so it is fully
unit-testable on a machine with no GPU and no network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from researchops.llm.providers import BaseLLM, ChatMessage
from researchops.rag.models import Chunk

# Citation markers look like [1], [2,3], or a run of [1][2]. We only expand
# comma-separated integers inside a single pair of brackets; ranges ([1-3]) are
# not part of the prompt contract and are treated as no match.
_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# A small stopword set for the lexical grounding heuristic. It only needs to be
# good enough to skip function words, not exhaustive.
_STOPWORDS = frozenset(
    """
    a an and are as at be but by for from has have in is it its of on or that
    the this to was were which with without model method paper using proposed
    propose we our results table figure shows show
    """.split()
)


@dataclass
class Citation:
    """A citation marker resolved to the chunk it points at."""

    index: int  # 1-based marker number, i.e. the `n` in `[n]`
    chunk: Chunk


@dataclass
class CitedAnswer:
    """A generated answer together with its citations and grounding verdicts."""

    answer: str
    chunks: list[Chunk]  # context passages, indexed 1..N by the prompt
    citations: list[Citation]
    grounded: list[bool]  # aligned with `citations`
    dangling_indices: list[int] = field(default_factory=list)  # out-of-range [n]


def build_prompt(query: str, chunks: list[Chunk]) -> list[ChatMessage]:
    """Render the retrieved chunks as numbered passages and ask for cited output.

    The system prompt is deliberately explicit: cite *only* passages that
    support the statement, and prefer saying "not covered" to hallucinating.
    """
    passages = []
    for i, c in enumerate(chunks, 1):
        # Provenance goes into the header so the model can cite, and so a human
        # (or an eval harness) can trace `[i]` back to a page/section.
        header = f"[{i}] doc={c.doc_id} page={c.page} section={c.section or '-'}"
        passages.append(f"{header}\n{c.text.strip()}")

    context = "\n\n".join(passages)

    system = (
        "You are a research assistant answering questions about scientific papers. "
        "Answer using ONLY the numbered passages below. Cite every factual claim "
        "inline with its passage number in square brackets, e.g. [1] or [2][3]. "
        "If the passages do not support an answer, say so instead of guessing."
    )
    user = (
        f"Passages:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer (with [n] citations):"
    )
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]


def extract_citation_indices(answer: str) -> list[int]:
    """Return the sorted, de-duplicated citation numbers found in ``answer``.

    Handles ``[1]``, ``[2,3]`` and adjacent markers ``[1][2]``. Markers whose
    number is out of range are kept so `validate_grounding` can flag them
    instead of silently dropping them.
    """
    found: list[int] = []
    for group in _CITE_RE.findall(answer):
        for part in group.split(","):
            part = part.strip()
            if part.isdigit():
                found.append(int(part))
    # Preserve first-seen order but drop duplicates.
    seen: set[int] = set()
    unique: list[int] = []
    for n in found:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return sorted(unique)


def resolve_citations(answer: str, chunks: list[Chunk]) -> list[Citation]:
    """Map the `[n]` markers in ``answer`` to the chunks they reference.

    Out-of-range markers are skipped here (they are surfaced by
    `validate_grounding`); valid ones become `Citation(index, chunk)`.
    """
    return [
        Citation(index=i, chunk=chunks[i - 1])
        for i in extract_citation_indices(answer)
        if 1 <= i <= len(chunks)
    ]


def validate_grounding(answer: str, chunks: list[Chunk]) -> list[bool]:
    """Deterministic grounding check for each citation in ``answer``.

    A citation is "grounded" when:
      * its number is in range (1..N), AND
      * the sentence that carries the marker shares at least one non-stopword
        content token with the cited chunk.

    The token-overlap test is a weak, cheap signal — not a substitute for NLI.
    Its job is to catch the obvious failures (an out-of-range `[9]`, or a claim
    attributed to a chunk that shares no vocabulary with it). The real
    faithfulness measurement happens in the RAGAS eval.
    """
    indices = extract_citation_indices(answer)
    sentences = _split_sentences(answer)
    verdicts: list[bool] = []
    for idx in indices:
        if not (1 <= idx <= len(chunks)):
            verdicts.append(False)
            continue
        verdicts.append(_sentence_overlaps_chunk(sentences, idx, chunks[idx - 1]))
    return verdicts


def _sentence_overlaps_chunk(sentences: list[str], index: int, chunk: Chunk) -> bool:
    marker = f"[{index}]"
    chunk_tokens = _content_tokens(chunk.text)
    for sent in sentences:
        if marker not in sent:
            continue
        return any(tok in chunk_tokens for tok in _content_tokens(sent))
    # Marker appears nowhere: nothing to check, treat as ungrounded.
    return False


def _content_tokens(text: str) -> set[str]:
    """Lowercased, non-stopword alphabetic tokens of length > 3."""
    return {
        t.lower()
        for t in re.findall(r"[A-Za-z]{4,}", text)
        if t.lower() not in _STOPWORDS
    }


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation, keeping the punctuation attached."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


async def generate_cited_answer(
    query: str, chunks: list[Chunk], llm: BaseLLM, *, max_tokens: int = 1024
) -> CitedAnswer:
    """Run the full cited-answer pipeline: prompt -> generate -> parse -> validate."""
    messages = build_prompt(query, chunks)
    resp = await llm.chat(messages, temperature=0.0, max_tokens=max_tokens)
    answer = resp.content

    # `validate_grounding` returns one verdict per citation index found, so it is
    # aligned with `extract_citation_indices`. Filter both to in-range citations
    # (so `citations` and `grounded` stay aligned) and surface out-of-range ones
    # as dangling (hallucinated) citations.
    indices = extract_citation_indices(answer)
    verdicts = validate_grounding(answer, chunks)
    citations: list[Citation] = []
    grounded: list[bool] = []
    dangling: list[int] = []
    for i, ok in zip(indices, verdicts, strict=True):
        if 1 <= i <= len(chunks):
            citations.append(Citation(index=i, chunk=chunks[i - 1]))
            grounded.append(ok)
        else:
            dangling.append(i)

    return CitedAnswer(
        answer=answer,
        chunks=chunks,
        citations=citations,
        grounded=grounded,
        dangling_indices=dangling,
    )
