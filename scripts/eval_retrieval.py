"""Retrieval evaluation runner (runs on the GPU host).

Loads a hand-labeled golden set, runs each query through the two-stage hybrid
retrieval (RRF fusion, then cross-encoder rerank), and reports Recall@k / Hit@k
/ MRR@k plus the rerank delta.

Usage:
    python eval_retrieval.py [path/to/golden_set.json] [k]

Defaults to /root/autodl-tmp/golden_set/restormer.json and k=5.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from researchops.eval.metrics import QueryResult, summarize, table_row_recall
from researchops.rag.retriever import Retriever

FUSE_POOL = 20  # candidate pool size before rerank


def load_golden_set(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _fused_pages(chunks: list) -> list[int]:
    return [c.page for c in chunks]


def _reranked_pages(results: list) -> list[int]:
    return [r.chunk.page for r in results]


async def main(data: dict[str, Any], k: int) -> None:
    items = data["items"]
    print(f"golden set: {data['paper']} ({len(items)} items), k={k}")

    retriever = Retriever()
    results: list[QueryResult] = []
    try:
        for item in items:
            query = item["query"]
            gold = item["gold_pages"]
            fused = await retriever.fuse(query, top_k=FUSE_POOL)
            reranked = await retriever.rerank(query, fused, top_k=k)
            gold_table = item.get("gold_table") or {}
            results.append(
                QueryResult(
                    query=query,
                    gold_pages=gold,
                    reranked_pages=_reranked_pages(reranked),
                    fused_pages=_fused_pages(fused),
                    gold_table_method=gold_table.get("method"),
                    gold_table_dataset=gold_table.get("dataset"),
                    reranked_texts=[r.chunk.text for r in reranked],
                    reranked_types=[r.chunk.chunk_type for r in reranked],
                )
            )
    finally:
        await retriever.close()

    print("\n=== per-item ===")
    for r in results:
        print(f"\nQ: {r.query}")
        print(f"  gold={r.gold_pages}")
        print(f"  fused(top20)={r.fused_pages[:10]}")
        print(f"  reranked(top{k})={r.reranked_pages}")
        if r.gold_table_method:
            print(f"  gold_table=({r.gold_table_method}, {r.gold_table_dataset})")

    print("\n=== summary ===")
    for name, val in summarize(results, k=k).items():
        print(f"  {name}: {val:.4f}")
    tr = table_row_recall(results, k)
    print(f"  table_row_recall@{k}: {tr:.4f}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/golden_set/restormer.json"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    asyncio.run(main(load_golden_set(path), k))
