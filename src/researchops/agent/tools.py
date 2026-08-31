"""Tool abstraction + the concrete tools the agent can call (rag_search + labops).

A ``Tool`` is a name, a JSON-Schema argument contract, and an async handler returning
text. The LLM sees the schemas (OpenAI function format); the registry dispatches the
parsed arguments and turns every outcome — success or error — into feedback text so a
ReAct loop can recover instead of crashing.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from researchops.db.store import ExperimentStore
from researchops.experiment import run_and_collect
from researchops.labops import LabClient
from researchops.rag.retriever import Retriever

ToolHandler = Callable[..., Awaitable[str]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema object for the arguments
    handler: ToolHandler


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool {name!r}; available: {sorted(self._tools)}") from exc

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """OpenAI function-calling schemas for every registered tool."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool and return its output (or an error) as text."""
        tool = self.get(name)
        try:
            return await tool.handler(**arguments)
        except Exception as exc:  # noqa: BLE001 — tool failures become ReAct feedback
            return f"error in tool {name}({arguments}): {exc}"


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


# A destructive action (submit_job / cancel_job) is gated by an approver that decides,
# from (tool_name, arguments), whether to allow it. ``None`` means deny — the fail-safe
# default — so an unguarded agent can only *propose* a dangerous call, never execute it.
# An approver may return a plain bool (e.g. the CLI's interactive prompt) or an awaitable
# of one (e.g. the web server's human-in-the-loop that waits on an HTTP decision).
Approver = Callable[[str, dict[str, Any]], bool | Awaitable[bool]]


async def _require_approval(
    tool_name: str, approver: Approver | None, arguments: dict[str, Any]
) -> str | None:
    """Return a rejection message if the action is not approved, else ``None``."""
    decision: bool = False
    if approver is not None:
        result = approver(tool_name, arguments)
        decision = await result if inspect.isawaitable(result) else bool(result)
    if not decision:
        return (
            f"REJECTED: {tool_name} is a destructive action and needs human approval, "
            f"which was not granted, so it was NOT executed. arguments={arguments}. "
            "Use read-only tools (rag_search, list_experiments, gpu_info, job_status, "
            "tail_log, fetch_metrics) to answer instead."
        )
    return None


def make_rag_search_tool(retriever: Retriever) -> Tool:
    """Retrieve cited chunks from the paper library."""

    async def handler(query: str, top_k: int = 5) -> str:
        results = await retriever.retrieve(query, rerank_top_k=top_k)
        if not results:
            return "No matching chunks found."
        lines = [
            f"[{i}] ({r.chunk.doc_id} p{r.chunk.page}) {r.chunk.text}"
            for i, r in enumerate(results, 1)
        ]
        return "\n\n".join(lines)

    return Tool(
        name="rag_search",
        description=(
            "Search the paper library for evidence (methods, datasets, PSNR/SSIM "
            "results). Returns top-k chunks numbered [1]..[k]; cite them by that number."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "search query"},
                "top_k": {"type": "integer", "description": "number of chunks", "default": 5},
            },
            "required": ["query"],
        },
        handler=handler,
    )


def make_labops_tools(
    client: LabClient,
    *,
    approver: Approver | None = None,
    store: ExperimentStore | None = None,
) -> list[Tool]:
    """The remote-lab tools, each a thin text-serializing wrapper over LabClient.

    ``submit_job``/``cancel_job`` are destructive, so they sit behind an ``approver``
    gate. With no approver (the default) they are deny-by-default.

    When a ``store`` is supplied, an eighth ``run_experiment`` tool is registered that
    wraps the deterministic submit->poll->parse->persist pipeline: the agent names the
    command, and the tool runs it to completion (no LLM in the polling loop), then
    returns the parsed metrics. It is likewise gated by the ``approver``.
    """

    async def gpu_info() -> str:
        return _dump([g.model_dump() for g in await client.gpu_info()])

    async def list_experiments() -> str:
        return _dump([e.model_dump() for e in await client.list_experiments()])

    async def submit_job(job_id: str, command: str) -> str:
        rejection = await _require_approval(
            "submit_job", approver, {"job_id": job_id, "command": command}
        )
        if rejection is not None:
            return rejection
        return _dump((await client.submit_job(job_id, command)).model_dump())

    async def job_status(job_id: str) -> str:
        return _dump((await client.job_status(job_id)).model_dump())

    async def tail_log(job_id: str, lines: int = 50) -> str:
        return await client.tail_log(job_id, lines)

    async def cancel_job(job_id: str) -> str:
        rejection = await _require_approval("cancel_job", approver, {"job_id": job_id})
        if rejection is not None:
            return rejection
        return _dump((await client.cancel_job(job_id)).model_dump())

    async def fetch_metrics(job_id: str) -> str:
        return _dump((await client.fetch_metrics(job_id)).model_dump())

    tools = [
        Tool(
            name="gpu_info",
            description="Read current GPU state (name, memory MB, utilization %, temperature C).",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=gpu_info,
        ),
        Tool(
            name="list_experiments",
            description="List top-level dirs/files in the remote working directory.",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=list_experiments,
        ),
        Tool(
            name="submit_job",
            description=(
                "Launch a command on the remote GPU host as a detached screen session. "
                "job_id must match [A-Za-z0-9_-]{1,64}. Destructive and requires human "
                "approval — only call when the task explicitly asks to run a job."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "command": {"type": "string"},
                },
                "required": ["job_id", "command"],
            },
            handler=submit_job,
        ),
        Tool(
            name="job_status",
            description="Report whether a job's screen session is live and its log exists.",
            parameters={
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
            handler=job_status,
        ),
        Tool(
            name="tail_log",
            description="Return the last N lines of a job's log.",
            parameters={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "lines": {"type": "integer", "default": 50},
                },
                "required": ["job_id"],
            },
            handler=tail_log,
        ),
        Tool(
            name="cancel_job",
            description="Terminate a job's screen session (idempotent). Destructive; requires human approval.",
            parameters={
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
            handler=cancel_job,
        ),
        Tool(
            name="fetch_metrics",
            description="Return a job's latest metrics from its .metrics.json file.",
            parameters={
                "type": "object",
                "properties": {"job_id": {"type": "string"}},
                "required": ["job_id"],
            },
            handler=fetch_metrics,
        ),
    ]

    if store is not None:
        tools.append(make_run_experiment_tool(client, store, approver=approver))

    return tools


def make_run_experiment_tool(
    client: LabClient, store: ExperimentStore, *, approver: Approver | None = None
) -> Tool:
    """One-shot tool wrapping the deterministic submit->poll->parse->persist pipeline.

    The agent names the command; this tool runs it to completion (no LLM in the polling
    loop), parses PSNR/SSIM from the log, persists Experiment/JobRun/Metric rows, and
    returns the parsed metrics. Gated by ``approver`` like the other destructive tools.
    """

    async def run_experiment(
        experiment_name: str, job_id: str, command: str, task: str = ""
    ) -> str:
        rejection = await _require_approval(
            "run_experiment",
            approver,
            {"experiment_name": experiment_name, "job_id": job_id, "command": command},
        )
        if rejection is not None:
            return rejection
        outcome = await run_and_collect(
            client,
            store,
            experiment_name=experiment_name,
            task=task or f"run {job_id}",
            job_id=job_id,
            command=command,
        )
        metrics = [
            {"name": m.name, "value": m.value, "dataset": m.dataset, "sigma": m.sigma}
            for m in outcome.metrics
        ]
        return _dump(
            {
                "status": outcome.status,
                "job_id": outcome.job_id,
                "elapsed_s": round(outcome.elapsed_s, 1),
                "metrics": metrics,
            }
        )

    return Tool(
        name="run_experiment",
        description=(
            "Run a command on the remote GPU host to completion and persist the result: "
            "submit a detached job, poll until it finishes or times out, parse PSNR/SSIM "
            "from the log, and write Experiment/JobRun/Metric rows to the database. Returns "
            "the parsed metrics. Prefer this over submit_job + job_status + tail_log when the "
            "task asks to reproduce or evaluate a model. Destructive (executes a command), so "
            "it requires human approval."
        ),
        parameters={
            "type": "object",
            "properties": {
                "experiment_name": {
                    "type": "string",
                    "description": "unique name for the persisted experiment record",
                },
                "job_id": {
                    "type": "string",
                    "description": "screen session name, [A-Za-z0-9_-]{1,64}",
                },
                "command": {
                    "type": "string",
                    "description": "exact shell command to run (already cd'ed into the working dir)",
                },
                "task": {
                    "type": "string",
                    "description": "human-readable description of what this job does",
                    "default": "",
                },
            },
            "required": ["experiment_name", "job_id", "command"],
        },
        handler=run_experiment,
    )


def build_default_tools(
    retriever: Retriever,
    lab_client: LabClient,
    *,
    approver: Approver | None = None,
    store: ExperimentStore | None = None,
) -> ToolRegistry:
    """Wire the full toolset: paper retrieval + remote-lab orchestration."""
    registry = ToolRegistry()
    registry.register(make_rag_search_tool(retriever))
    for tool in make_labops_tools(lab_client, approver=approver, store=store):
        registry.register(tool)
    return registry
