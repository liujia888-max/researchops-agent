from __future__ import annotations

from researchops.eval.metrics import (
    QueryResult,
    hit_at_k,
    mean_hit,
    mean_recall,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
    rerank_delta,
)


def test_recall_at_k() -> None:
    assert recall_at_k([3], [1, 3, 5], k=5) == 1.0
    assert recall_at_k([3, 7], [1, 3, 5], k=5) == 0.5
    assert recall_at_k([7], [1, 3, 5], k=5) == 0.0
    assert recall_at_k([], [1, 2], k=5) == 0.0


def test_hit_at_k() -> None:
    assert hit_at_k([3], [1, 3, 5], k=2) is True
    assert hit_at_k([3], [1, 2, 5], k=2) is False


def test_reciprocal_rank() -> None:
    assert reciprocal_rank([3], [3, 1, 5]) == 1.0
    assert reciprocal_rank([3], [1, 3, 5]) == 0.5
    assert reciprocal_rank([3], [1, 2, 5]) == 0.0


def test_mean_recall() -> None:
    results = [
        QueryResult(query="q1", gold_pages=[3], reranked_pages=[3, 1]),
        QueryResult(query="q2", gold_pages=[3], reranked_pages=[1, 2]),
    ]
    assert mean_recall(results, k=2) == 0.5


def test_mean_hit() -> None:
    results = [
        QueryResult(query="q1", gold_pages=[3], reranked_pages=[3, 1]),
        QueryResult(query="q2", gold_pages=[3], reranked_pages=[1, 2]),
    ]
    assert mean_hit(results, k=2) == 0.5


def test_mean_reciprocal_rank() -> None:
    results = [
        QueryResult(query="q1", gold_pages=[3], reranked_pages=[3, 1]),
        QueryResult(query="q2", gold_pages=[3], reranked_pages=[1, 3]),
    ]
    assert mean_reciprocal_rank(results, k=2) == (1.0 + 0.5) / 2


def test_rerank_delta_positive_when_rerank_helps() -> None:
    # RRF put the relevant page 4th; rerank moved it to 1st -> Recall@3 improves.
    results = [
        QueryResult(
            query="q",
            gold_pages=[3],
            fused_pages=[1, 2, 4, 3],
            reranked_pages=[3, 1, 2, 4],
        )
    ]
    assert rerank_delta(results, k=3) == 1.0


def test_rerank_delta_ignores_results_without_fused_data() -> None:
    results = [QueryResult(query="q", gold_pages=[3], reranked_pages=[3])]
    assert rerank_delta(results, k=3) == 0.0
