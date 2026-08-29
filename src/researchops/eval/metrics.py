"""Retrieval evaluation metrics.

Pure, dependency-free functions over ``(gold pages, retrieved pages)`` so they
run anywhere (CI, local, remote) and are unit-testable without a vector store
or GPU. The runner in ``scripts/eval_retrieval.py`` supplies real retrievals.

Standard IR definitions:

* **Recall@k** = |gold ∩ top-k| / |gold| — the fraction of relevant pages that
  surfaced within the first ``k`` results.
* **Hit@k** = 1 if *any* gold page appears in the top-k, else 0.
* **MRR@k** = mean over queries of ``1 / rank`` of the first gold page in the
  top-k (0 when absent).

The "rerank delta" is measured by running the same metric against the RRF-fused
ordering (``fused_pages``) and the cross-encoder-reranked ordering
(``reranked_pages``) and taking the difference.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QueryResult:
    """One golden query and both retrieval orderings for it."""

    query: str
    gold_pages: list[int]
    reranked_pages: list[int]  # final ordering (after rerank)
    fused_pages: list[int] | None = None  # RRF ordering (before rerank), if captured


def recall_at_k(gold: list[int], retrieved: list[int], k: int) -> float:
    """Fraction of gold pages found in the top-k of ``retrieved``."""
    if not gold:
        return 0.0
    return len(set(gold) & set(retrieved[:k])) / len(gold)


def hit_at_k(gold: list[int], retrieved: list[int], k: int) -> bool:
    """True if any gold page is in the top-k."""
    return bool(set(gold) & set(retrieved[:k]))


def reciprocal_rank(gold: list[int], retrieved: list[int]) -> float:
    """1 / rank of the first gold page (1-based), or 0.0 if none found."""
    gold_set = set(gold)
    for rank, page in enumerate(retrieved, 1):
        if page in gold_set:
            return 1.0 / rank
    return 0.0


def mean_recall(results: list[QueryResult], k: int) -> float:
    if not results:
        return 0.0
    return sum(recall_at_k(r.gold_pages, r.reranked_pages, k) for r in results) / len(results)


def mean_hit(results: list[QueryResult], k: int) -> float:
    if not results:
        return 0.0
    return sum(hit_at_k(r.gold_pages, r.reranked_pages, k) for r in results) / len(results)


def mean_reciprocal_rank(results: list[QueryResult], k: int) -> float:
    if not results:
        return 0.0
    return sum(reciprocal_rank(r.gold_pages, r.reranked_pages[:k]) for r in results) / len(
        results
    )


def rerank_delta(results: list[QueryResult], k: int) -> float:
    """Recall@k improvement from reranking, averaged over queries with fused data.

    Positive means the cross-encoder moved relevant pages into the top-k.
    """
    usable = [r for r in results if r.fused_pages is not None]
    if not usable:
        return 0.0
    before = sum(recall_at_k(r.gold_pages, r.fused_pages or [], k) for r in usable) / len(usable)
    after = sum(recall_at_k(r.gold_pages, r.reranked_pages, k) for r in usable) / len(usable)
    return after - before


def summarize(results: list[QueryResult], *, k: int = 5) -> dict[str, float]:
    """One-stop summary of the headline retrieval metrics at a given ``k``."""
    return {
        f"recall@{k}": mean_recall(results, k),
        f"hit@{k}": mean_hit(results, k),
        f"mrr@{k}": mean_reciprocal_rank(results, k),
        f"rerank_delta@{k}": rerank_delta(results, k),
    }
