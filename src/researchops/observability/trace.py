"""A small, dependency-free tracing layer for agent runs.

One ``Trace`` collects two kinds of spans:

* ``LlmSpan`` — one LLM ``chat`` call, attributed to a graph node
  (``planner`` / ``executor`` / ``reporter``), with its token usage and latency.
* ``ToolSpan`` — one executed tool call, with its latency.

``Trace.summary`` reduces the spans to the headline numbers the portfolio needs
(total tokens, estimated cost, wall time, per-node breakdown). The span model maps
1:1 onto OpenTelemetry/Langfuse (a trace of named, timestamped spans with attributes),
so swapping in a Langfuse exporter later is an adapter, not a rewrite.

Node attribution is done by matching the system prompt of each request against the
graph's prompt constants — exact matches, so no string heuristics. The wrapper is a
drop-in ``BaseLLM``: it forwards every call untouched and only observes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from researchops.agent.graph import PLANNER_PROMPT, REPORTER_PROMPT, SYSTEM_PROMPT
from researchops.agent.runner import run_agent
from researchops.agent.state import AgentState
from researchops.agent.tools import Tool, ToolRegistry
from researchops.llm.providers import BaseLLM, ChatMessage, ChatResponse

# DeepSeek-chat list prices (USD per 1M tokens). Verify against the provider before
# quoting exact dollar figures — the structure (tokens x price) is what matters here.
_DEFAULT_INPUT_PER_1M = 0.27
_DEFAULT_OUTPUT_PER_1M = 1.10


@dataclass(frozen=True)
class LlmSpan:
    """One LLM call, attributed to a graph node."""

    node: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float


@dataclass(frozen=True)
class ToolSpan:
    """One executed tool call."""

    name: str
    latency_s: float


@dataclass
class Trace:
    """Accumulates the spans of one agent run and reduces them to headline numbers."""

    task: str = ""
    llm_spans: list[LlmSpan] = field(default_factory=list)
    tool_spans: list[ToolSpan] = field(default_factory=list)
    _started: float = field(default_factory=time.perf_counter, repr=False)

    def record_llm(self, span: LlmSpan) -> None:
        self.llm_spans.append(span)

    def record_tool(self, span: ToolSpan) -> None:
        self.tool_spans.append(span)

    @property
    def wall_s(self) -> float:
        return time.perf_counter() - self._started

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.llm_spans)

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.llm_spans)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def estimated_cost_usd(
        self,
        *,
        input_per_1m: float = _DEFAULT_INPUT_PER_1M,
        output_per_1m: float = _DEFAULT_OUTPUT_PER_1M,
    ) -> float:
        """USD estimate from token counts; prices are per 1M tokens."""
        return (self.input_tokens * input_per_1m + self.output_tokens * output_per_1m) / 1_000_000

    def summary(self) -> dict[str, Any]:
        """Reduce to a JSON-friendly summary (headline numbers + per-node breakdown)."""
        per_node: dict[str, dict[str, float]] = {}
        for s in self.llm_spans:
            bucket = per_node.get(s.node)
            if bucket is None:
                bucket = {
                    "calls": 0.0,
                    "input_tokens": 0.0,
                    "output_tokens": 0.0,
                    "latency_s": 0.0,
                }
                per_node[s.node] = bucket
            bucket["calls"] += 1.0
            bucket["input_tokens"] += s.input_tokens
            bucket["output_tokens"] += s.output_tokens
            bucket["latency_s"] += s.latency_s
        return {
            "wall_s": round(self.wall_s, 3),
            "llm_calls": len(self.llm_spans),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd(), 6),
            "tool_calls": len(self.tool_spans),
            "per_node": per_node,
        }


def _infer_node(messages: list[ChatMessage]) -> str:
    """Attribute a request to a graph node by its leading system prompt."""
    first = messages[0].content if messages else ""
    if first == SYSTEM_PROMPT:
        return "executor"
    if first == PLANNER_PROMPT:
        return "planner"
    if first == REPORTER_PROMPT:
        return "reporter"
    return "llm"


class TracedLLM(BaseLLM):
    """Wraps a ``BaseLLM`` and records each ``chat`` call as an ``LlmSpan``."""

    def __init__(self, inner: BaseLLM, trace: Trace) -> None:
        super().__init__(inner.base_url, inner.api_key, inner.model)
        self._inner = inner
        self._trace = trace

    @property
    def name(self) -> str:
        return self._inner.name

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> ChatResponse:
        node = _infer_node(messages)
        start = time.perf_counter()
        resp = await self._inner.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
        )
        self._trace.record_llm(
            LlmSpan(
                node=node,
                model=resp.model,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                latency_s=time.perf_counter() - start,
            )
        )
        return resp


class TracedToolRegistry(ToolRegistry):
    """Wraps a ``ToolRegistry`` and records each ``execute`` as a ``ToolSpan``."""

    def __init__(self, inner: ToolRegistry, trace: Trace) -> None:
        super().__init__()
        self._inner = inner
        self._trace = trace

    def register(self, tool: Tool) -> None:
        self._inner.register(tool)

    def get(self, name: str) -> Tool:
        return self._inner.get(name)

    def names(self) -> list[str]:
        return self._inner.names()

    def schemas(self) -> list[dict[str, Any]]:
        return self._inner.schemas()

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        start = time.perf_counter()
        try:
            return await self._inner.execute(name, arguments)
        finally:
            self._trace.record_tool(ToolSpan(name=name, latency_s=time.perf_counter() - start))


async def traced_run_agent(
    task: str,
    *,
    llm: BaseLLM,
    registry: ToolRegistry,
    max_iterations: int = 10,
) -> tuple[AgentState, Trace]:
    """Run the agent with full tracing; return the final state and its trace."""
    trace = Trace(task=task)
    traced_llm: BaseLLM = TracedLLM(llm, trace)
    traced_registry: ToolRegistry = TracedToolRegistry(registry, trace)
    state = await run_agent(
        task, llm=traced_llm, registry=traced_registry, max_iterations=max_iterations
    )
    return state, trace
