"""Evaluation harness for ResearchOps Agent.

Phase 1 ships retrieval metrics (Recall@k / Hit@k / MRR@k / rerank delta) over
a hand-labeled golden set. The faithfulness / answer-relevancy scores (RAGAS)
land here in the same phase once the LLM-judge harness is wired up.
"""

from researchops.eval.metrics import (
    QueryResult,
    hit_at_k,
    mean_hit,
    mean_recall,
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
    rerank_delta,
    summarize,
)

__all__ = [
    "QueryResult",
    "hit_at_k",
    "mean_hit",
    "mean_recall",
    "mean_reciprocal_rank",
    "recall_at_k",
    "reciprocal_rank",
    "rerank_delta",
    "summarize",
]
