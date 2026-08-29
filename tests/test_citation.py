from __future__ import annotations

from researchops.llm.providers import BaseLLM, ChatResponse
from researchops.rag.citation import (
    build_prompt,
    extract_citation_indices,
    generate_cited_answer,
    resolve_citations,
    validate_grounding,
)
from researchops.rag.models import Chunk


def _chunk(text: str, **kw: object) -> Chunk:
    defaults = dict(doc_id="restormer", page=3, section="Method")
    defaults.update(kw)
    return Chunk(text=text, **defaults)


class _FakeLLM(BaseLLM):
    name = "fake"

    def __init__(self, content: str) -> None:
        super().__init__("http://fake.local", "k", "fake-model")
        self._content = content

    async def chat(self, messages, *, temperature=0.7, max_tokens=1024):
        return ChatResponse(content=self._content, model="fake-model")


def test_build_prompt_numbers_passages_with_provenance() -> None:
    chunks = [_chunk("First passage body."), _chunk("Second passage body.", page=8, section="Conclusion")]
    messages = build_prompt("What is Restormer?", chunks)
    user = messages[-1].content
    assert "[1] doc=restormer page=3 section=Method" in user
    assert "[2] doc=restormer page=8 section=Conclusion" in user
    assert "Question: What is Restormer?" in user
    # System prompt asks for inline citations.
    assert "square brackets" in messages[0].content


def test_extract_citation_indices() -> None:
    assert extract_citation_indices("answer [1]") == [1]
    assert extract_citation_indices("answer [2,3]") == [2, 3]
    assert extract_citation_indices("answer [1][2]") == [1, 2]
    assert extract_citation_indices("answer [2][1][2]") == [1, 2]  # dedup + sort
    assert extract_citation_indices("no citations here") == []


def test_resolve_citations_maps_to_chunks_and_skips_out_of_range() -> None:
    chunks = [_chunk("body a"), _chunk("body b")]
    citations = resolve_citations("claims [1] and [3]", chunks)
    assert [c.index for c in citations] == [1]  # [3] is out of range, dropped


def test_validate_grounding_in_range_and_overlap() -> None:
    chunks = [_chunk("Restoration Transformer achieves state of the art results.")]
    answer = "Restoration Transformer achieves strong performance [1]."
    assert validate_grounding(answer, chunks) == [True]


def test_validate_grounding_out_of_range_is_false() -> None:
    chunks = [_chunk("body")]
    assert validate_grounding("the answer is [9]", chunks) == [False]


def test_validate_grounding_no_lexical_overlap_is_false() -> None:
    chunks = [_chunk("The method uses attention mechanisms for image restoration.")]
    answer = "This introduces a totally unrelated quantum computing idea [1]."
    assert validate_grounding(answer, chunks) == [False]


def test_generate_cited_answer_end_to_end() -> None:
    chunks = [_chunk("Restormer is a restoration transformer.")]
    llm = _FakeLLM("Restormer is a restoration transformer [1].")
    result = __import__("asyncio").run(generate_cited_answer("What is Restormer?", chunks, llm))
    assert result.answer == "Restormer is a restoration transformer [1]."
    assert [c.index for c in result.citations] == [1]
    assert result.grounded == [True]
