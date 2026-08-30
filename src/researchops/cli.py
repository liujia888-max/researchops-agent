"""Command-line entrypoint: `researchops <command>`."""

from __future__ import annotations

import argparse
import asyncio

from researchops.llm import build_llm
from researchops.llm.providers import ChatMessage


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="researchops", description="ResearchOps Agent CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ping", help="test the configured LLM provider")
    p.add_argument("--provider", choices=["deepseek", "qwen", "vllm"], default=None)
    p.add_argument("--message", default="Reply with exactly: pong")

    p = sub.add_parser("ingest", help="ingest a paper PDF into the RAG index")
    p.add_argument("pdf", help="path to the PDF to ingest")

    p = sub.add_parser("search", help="hybrid-retrieve against the RAG index")
    p.add_argument("query", help="search query")
    p.add_argument("--top-k", type=int, default=None)

    sub.add_parser("mcp", help="run the labops MCP server over stdio")
    return parser


async def _ping(provider: str | None, message: str) -> None:
    llm = build_llm(provider=provider)
    resp = await llm.chat([ChatMessage(role="user", content=message)])
    print(f"[{resp.model}] {resp.content}")
    print(f"tokens: in={resp.input_tokens} out={resp.output_tokens}")


async def _ingest(pdf: str) -> None:
    from researchops.rag.ingest import ingest_pdf

    n = await ingest_pdf(pdf)
    print(f"ingested {n} chunks")


async def _search(query: str, top_k: int | None) -> None:
    from researchops.rag.retriever import Retriever

    retriever = Retriever()
    try:
        results = await retriever.retrieve(query, top_k=top_k)
    finally:
        await retriever.close()

    for i, r in enumerate(results, 1):
        c = r.chunk
        print(f"[{i}] score={r.score:.4f}  {c.doc_id}:p{c.page}  ({c.section})")
        print(f"    {c.text[:160].strip()}")
        print()


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "ping":
        asyncio.run(_ping(args.provider, args.message))
    elif args.command == "ingest":
        asyncio.run(_ingest(args.pdf))
    elif args.command == "search":
        asyncio.run(_search(args.query, args.top_k))
    elif args.command == "mcp":
        from researchops.mcp.server import mcp

        mcp.run()
    else:
        raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
