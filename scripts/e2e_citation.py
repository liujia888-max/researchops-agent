"""End-to-end cited generation demo (runs on the GPU host).

Full Phase 1 pipeline: hybrid retrieval -> cite-aware generation with a real LLM
-> print the answer, its citations, grounding verdicts, and provenance.

Requires DEEPSEEK_API_KEY (via a `.env` in the CWD or the environment).

Usage:
    python e2e_citation.py ["query"] [top_k]
"""

from __future__ import annotations

import asyncio
import sys

from researchops.config import get_settings
from researchops.llm.providers import build_llm
from researchops.rag.citation import generate_cited_answer
from researchops.rag.retriever import Retriever

DEFAULT_QUERY = (
    "What PSNR does Restormer achieve on CBSD68 for Gaussian color image "
    "denoising at sigma=25, and how does it compare to SwinIR?"
)


async def main() -> None:
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    rerank_top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    settings = get_settings()
    llm = build_llm(settings)

    retriever = Retriever(settings)
    try:
        results = await retriever.retrieve(query, top_k=20, rerank_top_k=rerank_top_k)
        chunks = [r.chunk for r in results]
    finally:
        await retriever.close()

    print(f"Q: {query}\n")
    print("=== retrieved context (top-k) ===")
    for i, r in enumerate(results, 1):
        c = r.chunk
        print(f"[{i}] {c.doc_id}:p{c.page} ({c.section or '-'}) score={r.score:.3f}")

    print("\n=== generated answer (with citations) ===")
    result = await generate_cited_answer(query, chunks, llm)
    print(result.answer)

    print("\n=== citations / grounding ===")
    for c, ok in zip(result.citations, result.grounded, strict=True):
        ch = c.chunk
        status = "grounded" if ok else "UNGROUNDED"
        print(f"[{c.index}] {status}  doc={ch.doc_id} page={ch.page} section={ch.section or '-'}")
        print(f"    {ch.text[:160].strip()}")
    if result.dangling_indices:
        print(f"dangling (hallucinated) citations: {result.dangling_indices}")


if __name__ == "__main__":
    asyncio.run(main())
