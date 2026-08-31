"""Streaming wrapper over the agent graph.

``run_agent`` blocks until the whole run finishes and returns the final state.
For a web UI we want incremental progress, so ``stream_agent`` drives the same
compiled graph with ``astream(stream_mode="updates")`` and turns each node
transition into a small, JSON-friendly event:

* ``plan``        — the planner's ordered steps.
* ``tool_call``   — the executor requested one tool call.
* ``tool_result`` — that tool finished and produced evidence.
* ``report``      — the reporter's final, citation-bearing answer.

Tracing is deliberately *not* this module's job: the caller wraps the LLM and
tool registry in ``TracedLLM`` / ``TracedToolRegistry`` beforehand if it wants a
``Trace`` (tokens/cost/latency) alongside the stream. That keeps this generator
pure and unit-testable with a scripted LLM and no observability dependency.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from researchops.agent.graph import build_agent
from researchops.agent.state import AgentState
from researchops.agent.tools import ToolRegistry
from researchops.llm.providers import BaseLLM


async def stream_agent(
    task: str,
    *,
    llm: BaseLLM,
    registry: ToolRegistry,
    max_iterations: int = 10,
    max_retries: int = 2,
    retry_backoff_s: float = 1.0,
) -> AsyncIterator[dict[str, Any]]:
    """Run the agent, yielding one event per graph node transition."""
    app = build_agent(
        llm,
        registry,
        max_iterations=max_iterations,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )
    initial = AgentState(task=task, max_iterations=max_iterations)

    async for chunk in app.astream(initial, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "planner":
                yield {"event": "plan", "plan": list(update.get("plan") or [])}
            elif node == "executor":
                pending = update.get("pending_tool")
                if pending:
                    yield {
                        "event": "tool_call",
                        "name": pending["name"],
                        "arguments": pending["arguments"],
                    }
            elif node == "tools":
                results = update.get("tool_results") or []
                if results:
                    last = results[-1]
                    yield {
                        "event": "tool_result",
                        "name": last.tool,
                        "arguments": last.arguments,
                        "output": last.output,
                    }
            elif node == "reporter":
                yield {"event": "report", "report": update.get("final_report", "")}
