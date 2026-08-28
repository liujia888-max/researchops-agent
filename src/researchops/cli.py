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
    return parser


async def _ping(provider: str | None, message: str) -> None:
    llm = build_llm(provider=provider)
    resp = await llm.chat([ChatMessage(role="user", content=message)])
    print(f"[{resp.model}] {resp.content}")
    print(f"tokens: in={resp.input_tokens} out={resp.output_tokens}")


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "ping":
        asyncio.run(_ping(args.provider, args.message))
    else:
        raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
