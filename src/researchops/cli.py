"""Command-line entrypoint: `researchops <command>`."""

from __future__ import annotations

import argparse
import asyncio
from typing import Any

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

    sub.add_parser("doctor", help="check external dependencies (LLM/Qdrant/embedding/GPU)")

    p = sub.add_parser("agent", help="run the agent end-to-end on a research task")
    p.add_argument("task", help="the research task, e.g. 'reproduce Restormer on CBSD68'")
    p.add_argument("--max-iterations", type=int, default=10)
    p.add_argument(
        "--interactive-approval",
        action="store_true",
        help="prompt for human approval before destructive labops actions (submit_job/cancel_job)",
    )
    p.add_argument(
        "--multi",
        action="store_true",
        help="run the supervisor + specialists multi-agent team instead of the single agent",
    )
    p.add_argument(
        "--trace",
        action="store_true",
        help="print a token/cost/latency trace summary after the run",
    )
    p.add_argument(
        "--langfuse",
        action="store_true",
        help="export the run to Langfuse (cloud) and print the trace URL; implies --trace",
    )
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


def _interactive_approver(tool_name: str, arguments: dict[str, Any]) -> bool:
    """Prompt the human before a destructive action (used with --interactive-approval)."""
    print(f"\n[approval] destructive action requested: {tool_name}({arguments})")
    answer = input("Approve? [y/N] ").strip().lower()
    return answer in {"y", "yes"}


async def _agent(
    task: str,
    max_iterations: int,
    interactive_approval: bool = False,
    trace: bool = False,
    langfuse: bool = False,
    multi: bool = False,
) -> None:
    from pathlib import Path

    from researchops.agent.runner import run_agent, run_multi_agent
    from researchops.agent.tools import build_default_tools
    from researchops.db.store import ExperimentStore
    from researchops.labops import LabClient, SshConnection
    from researchops.memory import SqliteMemoryStore
    from researchops.rag.retriever import Retriever

    llm = build_llm()
    retriever = Retriever()
    lab_client = LabClient(SshConnection())
    store = ExperimentStore()
    memory = SqliteMemoryStore()
    approver = _interactive_approver if interactive_approval else None
    registry = await build_default_tools(
        retriever, lab_client, approver=approver, store=store, memory=memory
    )
    run_trace = None
    final_report = ""
    try:
        Path(".researchops").mkdir(exist_ok=True)  # noqa: ASYNC240  # one-time startup, not hot-path I/O
        await store.init()
        if multi:
            if trace or langfuse:
                from researchops.observability.trace import traced_run_multi_agent

                multi_state, run_trace = await traced_run_multi_agent(
                    task, llm=llm, registry=registry, max_iterations=max_iterations
                )
            else:
                multi_state = await run_multi_agent(
                    task, llm=llm, registry=registry, max_iterations=max_iterations
                )
            final_report = multi_state.final_report
        elif trace or langfuse:
            from researchops.observability.trace import traced_run_agent

            traced_state, run_trace = await traced_run_agent(
                task, llm=llm, registry=registry, max_iterations=max_iterations
            )
            final_report = traced_state.final_report
        else:
            single_state = await run_agent(
                task, llm=llm, registry=registry, max_iterations=max_iterations
            )
            final_report = single_state.final_report
    finally:
        await retriever.close()
        await lab_client.close()
        await store.close()
        await memory.close()
    print(final_report)
    if trace and run_trace is not None:
        import json

        print("\n[trace]")
        print(json.dumps(run_trace.summary(), ensure_ascii=False, indent=2, default=str))
    if langfuse and run_trace is not None:
        from researchops.config import get_settings
        from researchops.observability.langfuse import build_client, export_trace, is_configured

        settings = get_settings()
        if not is_configured(settings):
            print(
                "\n[langfuse] skipped: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY "
                "not set in .env"
            )
            return
        lf = build_client(settings)
        try:
            url = export_trace(run_trace, lf=lf)
            lf.flush()
            print(f"\n[langfuse] {url}")
        finally:
            lf.shutdown()


async def _doctor() -> None:
    from researchops.doctor import format_report, run_doctor

    report = await run_doctor()
    print(format_report(report))


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "ping":
        asyncio.run(_ping(args.provider, args.message))
    elif args.command == "ingest":
        asyncio.run(_ingest(args.pdf))
    elif args.command == "search":
        asyncio.run(_search(args.query, args.top_k))
    elif args.command == "doctor":
        asyncio.run(_doctor())
    elif args.command == "mcp":
        from researchops.mcp.server import mcp

        mcp.run()
    elif args.command == "agent":
        asyncio.run(
            _agent(
                args.task,
                args.max_iterations,
                args.interactive_approval,
                args.trace,
                args.langfuse,
                args.multi,
            )
        )
    else:
        raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
