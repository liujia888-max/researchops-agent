"""RAGAS faithfulness + answer_relevancy evaluation (runs on the GPU host).

For every golden-set item this (1) retrieves top-k contexts (hybrid + rerank),
(2) generates a plain answer with the configured LLM, then (3) runs RAGAS's
LLM-judge metrics — Faithfulness (is each claim in the answer supported by the
retrieved contexts?) and AnswerRelevancy (does the answer actually address the
question?) — over the (question, answer, contexts, reference) tuples.

AnswerRelevancy needs an embedding model: we wrap the bge-m3 inference service
in a LangChain-``Embeddings``-shaped adapter so the same dense encoder used for
retrieval is reused for evaluation.

Note: ragas 0.4 has two parallel APIs. This script uses the legacy ``evaluate()``
entry point (``ragas.metrics.Faithfulness`` / ``AnswerRelevancy`` are the ``Metric``
subclasses ``evaluate()`` accepts); the newer ``ragas.metrics.collections`` metrics
belong to the ``@experiment`` decorator and are rejected by ``evaluate()``.

Requires (installed on the host, NOT part of the core package):
    pip install ragas langchain-openai langchain-core

Also needs DEEPSEEK_API_KEY (.env) and the bge-m3 service at inference_base_url.

Usage:
    python eval_ragas.py [path/to/golden_set.json] [top_k]
"""

from __future__ import annotations

import asyncio
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import httpx

from researchops.config import Settings, get_settings
from researchops.llm.providers import ChatMessage, build_llm
from researchops.rag.models import Chunk
from researchops.rag.retriever import Retriever

FUSE_POOL = 20  # candidate pool size before rerank


def load_golden_set(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


async def _generate_plain_answer(query: str, chunks: list[Chunk], llm: Any) -> str:
    """Answer without citation markers, so RAGAS judges the content, not the `[n]`s."""
    context = "\n\n".join(f"[{i}] {c.text.strip()}" for i, c in enumerate(chunks, 1))
    system = (
        "You are a research assistant answering questions about scientific papers. "
        "Answer using ONLY the provided passages. Be concise and factual; do not "
        "invent numbers that are not in the passages."
    )
    user = f"Passages:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"
    resp = await llm.chat(
        [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)],
        temperature=0.0,
        max_tokens=512,
    )
    return resp.content


async def build_samples(data: dict[str, Any], k: int) -> list[dict[str, Any]]:
    """Retrieve + generate one sample per golden-set item."""
    settings = get_settings()
    llm = build_llm(settings)
    retriever = Retriever(settings)
    samples: list[dict[str, Any]] = []
    try:
        for item in data["items"]:
            query = item["query"]
            results = await retriever.retrieve(query, top_k=FUSE_POOL, rerank_top_k=k)
            chunks = [r.chunk for r in results]
            response = await _generate_plain_answer(query, chunks, llm)
            samples.append(
                {
                    "user_input": query,
                    "response": response,
                    "reference": item["answer"],
                    "retrieved_contexts": [c.text for c in chunks],
                }
            )
            print(f"[{item['id']}] {query}")
            print(f"    -> {response[:140]!r}")
    finally:
        await retriever.close()
    return samples


class BGEHttpEmbeddings:
    """LangChain-``Embeddings``-shaped adapter over the bge-m3 HTTP service.

    Exposes the same dense encoder the retriever uses, so AnswerRelevancy's
    similarity lives in the same vector space as retrieval. Only the sync
    ``embed_documents`` / ``embed_query`` surface is implemented; ragas's
    LangchainEmbeddingsWrapper runs those through an executor.
    """

    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")

    def _dense(self, texts: list[str]) -> list[list[float]]:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(f"{self._base}/v1/embeddings", json={"inputs": texts})
        resp.raise_for_status()
        return resp.json()["dense"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._dense(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._dense([text])[0]


def run_ragas(samples: list[dict[str, Any]], settings: Settings) -> None:
    # Heavy, eval-only deps are imported lazily so the retrieval/generation half
    # of this script still runs without them. ragas 0.4's legacy `evaluate()`
    # path emits DeprecationWarnings; silence them for readable output.
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, Faithfulness
    from ragas.run_config import RunConfig

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            base_url=settings.deepseek_base_url,
            api_key=settings.deepseek_api_key,
            model=settings.deepseek_model,
            temperature=0.0,
        ),
        # DeepSeek rejects n>1; ragas defaults statement/question generation to
        # n=3, so bypass it and let the client use n=1.
        bypass_n=True,
    )
    embeddings = LangchainEmbeddingsWrapper(BGEHttpEmbeddings(settings.inference_base_url))

    dataset = EvaluationDataset(samples=[SingleTurnSample(**s) for s in samples])
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=judge_llm),
            AnswerRelevancy(llm=judge_llm, embeddings=embeddings),
        ],
        # DeepSeek is latency/rate-limit sensitive; faithfulness does many
        # sequential NLI calls per item, so give it headroom and keep
        # concurrency modest to avoid 429s.
        run_config=RunConfig(timeout=600, max_workers=8),
        show_progress=False,
    )

    df = result.to_pandas()
    cols = [c for c in ("user_input", "faithfulness", "answer_relevancy") if c in df.columns]
    print("\n=== per-item ===")
    print(df[cols].to_string(index=False))
    print("\n=== summary ===")
    for name in ("faithfulness", "answer_relevancy"):
        if name in df.columns:
            print(f"  {name}: mean={df[name].mean():.4f}  min={df[name].min():.4f}")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/golden_set/restormer.json"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    data = load_golden_set(path)
    samples = asyncio.run(build_samples(data, k))
    run_ragas(samples, get_settings())


if __name__ == "__main__":
    main()
