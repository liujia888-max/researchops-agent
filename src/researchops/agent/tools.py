"""Tool abstraction + the concrete tools the agent can call (rag_search + labops).

A ``Tool`` is a name, a JSON-Schema argument contract, and an async handler returning
text. The LLM sees the schemas (OpenAI function format); the registry dispatches the
parsed arguments and turns every outcome — success or error — into feedback text so a
ReAct loop can recover instead of crashing.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

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


def make_labops_tools(client: LabClient) -> list[Tool]:
    """The seven remote-lab tools, each a thin text-serializing wrapper over LabClient."""

    async def gpu_info() -> str:
        return _dump([g.model_dump() for g in await client.gpu_info()])

    async def list_experiments() -> str:
        return _dump([e.model_dump() for e in await client.list_experiments()])

    async def submit_job(job_id: str, command: str) -> str:
        return _dump((await client.submit_job(job_id, command)).model_dump())

    async def job_status(job_id: str) -> str:
        return _dump((await client.job_status(job_id)).model_dump())

    async def tail_log(job_id: str, lines: int = 50) -> str:
        return await client.tail_log(job_id, lines)

    async def cancel_job(job_id: str) -> str:
        return _dump((await client.cancel_job(job_id)).model_dump())

    async def fetch_metrics(job_id: str) -> str:
        return _dump((await client.fetch_metrics(job_id)).model_dump())

    return [
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
                "job_id must match [A-Za-z0-9_-]{1,64}. Destructive — only call when "
                "the task asks to run a job."
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
            description="Terminate a job's screen session (idempotent).",
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


def build_default_tools(retriever: Retriever, lab_client: LabClient) -> ToolRegistry:
    """Wire the full toolset: paper retrieval + remote-lab orchestration."""
    registry = ToolRegistry()
    registry.register(make_rag_search_tool(retriever))
    for tool in make_labops_tools(lab_client):
        registry.register(tool)
    return registry
