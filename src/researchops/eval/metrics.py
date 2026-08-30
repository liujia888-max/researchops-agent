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

from dataclasses import dataclass, field


@dataclass
class QueryResult:
    """One golden query and both retrieval orderings for it."""

    query: str
    gold_pages: list[int]
    reranked_pages: list[int]  # final ordering (after rerank)
    fused_pages: list[int] | None = None  # RRF ordering (before rerank), if captured

    # Table-row recall support (optional; only populated by the table-aware
    # runner). ``reranked_texts``/``reranked_types`` are the top-k chunk texts
    # and their ``chunk_type`` in rank order.
    gold_table_method: str | None = None
    gold_table_dataset: str | None = None
    reranked_texts: list[str] = field(default_factory=list)
    reranked_types: list[str] = field(default_factory=list)


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


def table_row_recall(results: list[QueryResult], k: int) -> float:
    """Fraction of table-centric queries whose top-k contains the gold table row.

    A query is "table-centric" when ``gold_table_method`` is set. It is counted
    as recalled when some top-k chunk is a ``table_row`` whose text carries both
    the gold method and dataset labels — i.e. the *structured* row surfaced, not
    just the prose number soup. Returns 0.0 when no query is table-centric.
    """
    total = 0
    hit = 0
    for r in results:
        if not r.gold_table_method:
            continue
        total += 1
        for text, ctype in zip(r.reranked_texts[:k], r.reranked_types[:k], strict=False):
            # Match the dataset as its own "on <dataset>:" token so e.g. "BSD68"
            # does not spuriously match the color "CBSD68" row.
            if (
                ctype == "table_row"
                and r.gold_table_method in text
                and f"on {r.gold_table_dataset}:" in text
            ):
                hit += 1
                break
    return hit / total if total else 0.0


def summarize(results: list[QueryResult], *, k: int = 5) -> dict[str, float]:
    """One-stop summary of the headline retrieval metrics at a given ``k``."""
    return {
        f"recall@{k}": mean_recall(results, k),
        f"hit@{k}": mean_hit(results, k),
        f"mrr@{k}": mean_reciprocal_rank(results, k),
        f"rerank_delta@{k}": rerank_delta(results, k),
    }
