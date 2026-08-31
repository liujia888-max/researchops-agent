"""Supervisor + specialist multi-agent graph.

The single-agent graph runs one ReAct loop over *all* tools. This module splits the
work across a small team so each specialist sees only the tools relevant to its role
and the supervisor routes the task:

* ``supervisor`` — one LLM call: pick the specialists and write a one-line brief each.
* ``workers``    — run the selected specialists (researcher / labops) concurrently,
                   each a bounded ReAct loop over its own tool subset.
* ``reporter``   — synthesize the specialists' findings into one citation-bearing report.

Each specialist loop reuses the same guardrails as the single agent (repetition guard +
failure retry), so a specialist that gets stuck cannot spin forever.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from researchops.agent.state import ToolResult
from researchops.agent.tools import ToolRegistry
from researchops.llm.providers import BaseLLM, ChatMessage

# Tool surface each specialist is allowed to touch. ``run_specialist`` filters these
# against the actually-registered tools, so a specialist silently skips any tool that
# wasn't wired up (e.g. ``run_experiment`` without a store, ``memory_search`` without
# a memory backend).
RESEARCHER_TOOLS = ["rag_search", "memory_search"]
LABOPS_TOOLS = [
    "gpu_info",
    "list_experiments",
    "submit_job",
    "job_status",
    "tail_log",
    "cancel_job",
    "fetch_metrics",
    "run_experiment",
]

SUPERVISOR_PROMPT = """You are the supervisor of a small multi-agent research team. Given the task, decide which specialists to run and write a one-line brief for each.

Specialists:
- researcher: searches the paper library (rag_search) and long-term memory (memory_search); produces cited findings about methods and results.
- labops: inspects the remote GPU lab (gpu_info, list_experiments, job_status, tail_log, fetch_metrics); only runs jobs (submit_job/cancel_job/run_experiment) when the task explicitly asks.

Return ONLY a JSON object with keys "specialists" (array, a subset of ["researcher","labops"]) and "briefs" (object mapping specialist name to a one-line brief). No other text."""

RESEARCHER_PROMPT = """You are a research librarian. Gather evidence with rag_search (paper library) and memory_search (past experiments/notes), then stop once you have enough and summarize the key facts, citing chunks by their [n] number exactly as returned."""

LABOPS_PROMPT = """You are a lab operations specialist. Inspect the remote GPU host with read-only tools (gpu_info, list_experiments, job_status, tail_log, fetch_metrics). Only call submit_job/cancel_job/run_experiment when the task explicitly asks to run a job. Summarize the host and job state you observed."""

MULTI_REPORTER_PROMPT = """You are writing a research report from a team's findings. Synthesize the specialists' evidence into a concise, well-structured report that directly answers the task. Cite evidence with [1][2]... matching the numbering in rag_search results. Do not invent numbers absent from the evidence. Use markdown headings and finish with a short "Conclusion"."""

_ERROR_PREFIX = "error in tool "
_NO_RETRY_TOOLS = {"run_experiment"}


def _signature(name: str, arguments: dict[str, Any]) -> str:
    return f"{name}:{json.dumps(arguments, sort_keys=True, default=str)}"


def _extract_json(content: str) -> dict[str, Any] | None:
    """Parse a JSON object out of the supervisor's reply, tolerating prose/backticks."""
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match is None:
            return None
        try:
            data = json.loads(match.group())
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


def _parse_supervisor(content: str) -> tuple[list[str], dict[str, str]]:
    """Extract ``(roles, briefs)``, defaulting to both specialists on any parse error."""
    data = _extract_json(content)
    if data is None:
        return ["researcher", "labops"], {}
    roles = [r for r in data.get("specialists", []) if r in {"researcher", "labops"}]
    briefs_raw = data.get("briefs", {})
    briefs = (
        {r: str(briefs_raw.get(r, "")) for r in roles}
        if isinstance(briefs_raw, dict)
        else {}
    )
    return roles or ["researcher", "labops"], briefs


@dataclass(frozen=True)
class SpecialistResult:
    """A specialist's finished work: the tools it called and its written summary."""

    role: str
    findings: list[ToolResult]
    summary: str


class MultiAgentState(BaseModel):
    """State threaded through the multi-agent graph."""

    task: str
    plan: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    briefs: dict[str, str] = field(default_factory=dict)
    findings: dict[str, list[ToolResult]] = field(default_factory=dict)
    summaries: dict[str, str] = field(default_factory=dict)
    final_report: str = ""
    max_iterations: int = 10


async def run_specialist(
    llm: BaseLLM,
    registry: ToolRegistry,
    *,
    role: str,
    system_prompt: str,
    task: str,
    brief: str,
    tool_names: list[str],
    max_iterations: int,
    max_retries: int = 2,
    retry_backoff_s: float = 1.0,
) -> SpecialistResult:
    """Run one specialist: a bounded ReAct loop over its own tool subset.

    The loop stops when the specialist emits no tool call (its text is the summary), when
    it repeats an already-executed call (repetition guard), or when the step budget runs
    out. Tool failures are retried with backoff, mirroring the single-agent executor.
    """
    available = set(registry.names())
    tool_names = [n for n in tool_names if n in available]
    if not tool_names:
        return SpecialistResult(role=role, findings=[], summary="(no tools assigned)")

    subset = registry.subset(tool_names)
    transcript: list[ChatMessage] = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=f"Task: {task}\n\nYour brief: {brief or task}"),
    ]
    findings: list[ToolResult] = []
    past: set[str] = set()

    for _ in range(max_iterations):
        resp = await llm.chat(
            transcript, tools=subset.schemas(), temperature=0.0, max_tokens=1024
        )
        if not resp.tool_calls:
            return SpecialistResult(
                role=role, findings=findings, summary=resp.content or "(no summary)"
            )

        call = resp.tool_calls[0]
        sig = _signature(call.name, call.arguments)
        if sig in past:
            return SpecialistResult(role=role, findings=findings, summary="(repetition guard)")
        past.add(sig)

        transcript.append(
            ChatMessage(role="assistant", content=resp.content, tool_calls=resp.raw_tool_calls[:1])
        )

        output = await subset.execute(call.name, call.arguments)
        if call.name not in _NO_RETRY_TOOLS:
            for _ in range(max_retries):
                if not output.startswith(_ERROR_PREFIX):
                    break
                await asyncio.sleep(retry_backoff_s)
                output = await subset.execute(call.name, call.arguments)

        findings.append(ToolResult(tool=call.name, arguments=call.arguments, output=output))
        transcript.append(
            ChatMessage(role="tool", content=output, name=call.name, tool_call_id=call.id)
        )

    return SpecialistResult(role=role, findings=findings, summary="(step budget exhausted)")


def build_multi_agent(
    llm: BaseLLM,
    registry: ToolRegistry,
    *,
    max_iterations: int = 10,
    max_retries: int = 2,
    retry_backoff_s: float = 1.0,
) -> Any:
    """Compile the supervisor -> (researcher | labops) -> reporter graph."""

    specialist_configs = {
        "researcher": (RESEARCHER_PROMPT, RESEARCHER_TOOLS),
        "labops": (LABOPS_PROMPT, LABOPS_TOOLS),
    }

    async def supervisor(state: MultiAgentState) -> dict[str, Any]:
        resp = await llm.chat(
            [
                ChatMessage(role="system", content=SUPERVISOR_PROMPT),
                ChatMessage(role="user", content=state.task),
            ],
            temperature=0.0,
            max_tokens=512,
        )
        roles, briefs = _parse_supervisor(resp.content)
        plan = [f"{r}: {briefs.get(r, '')}".strip() for r in roles]
        return {"roles": roles, "briefs": briefs, "plan": plan}

    async def workers(state: MultiAgentState) -> dict[str, Any]:
        async def run(role: str) -> SpecialistResult:
            system_prompt, tool_names = specialist_configs[role]
            return await run_specialist(
                llm,
                registry,
                role=role,
                system_prompt=system_prompt,
                task=state.task,
                brief=state.briefs.get(role, ""),
                tool_names=tool_names,
                max_iterations=state.max_iterations,
                max_retries=max_retries,
                retry_backoff_s=retry_backoff_s,
            )

        results = await asyncio.gather(*(run(role) for role in state.roles))
        return {
            "findings": {r.role: r.findings for r in results},
            "summaries": {r.role: r.summary for r in results},
        }

    async def reporter(state: MultiAgentState) -> dict[str, Any]:
        sections: list[str] = []
        for role in state.roles:
            sections.append(f"## {role}\n{state.summaries.get(role, '')}")
            for fr in state.findings.get(role, []):
                sections.append(f"### {fr.tool}({fr.arguments})\n{fr.output}")
        evidence = "\n\n".join(sections) or "(no evidence was gathered)"
        resp = await llm.chat(
            [
                ChatMessage(role="system", content=MULTI_REPORTER_PROMPT),
                ChatMessage(role="user", content=f"Task: {state.task}\n\nEvidence:\n{evidence}"),
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        return {"final_report": resp.content}

    graph = StateGraph(MultiAgentState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("workers", workers)
    graph.add_node("reporter", reporter)
    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "workers")
    graph.add_edge("workers", "reporter")
    graph.add_edge("reporter", END)
    return graph.compile()
