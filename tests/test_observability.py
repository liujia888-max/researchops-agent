"""Tests for the observability layer (Trace / TracedLLM / TracedToolRegistry)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from researchops.agent.graph import PLANNER_PROMPT, REPORTER_PROMPT, SYSTEM_PROMPT
from researchops.agent.multi import (
    LABOPS_PROMPT,
    MULTI_REPORTER_PROMPT,
    RESEARCHER_PROMPT,
    SUPERVISOR_PROMPT,
)
from researchops.agent.tools import Tool, ToolRegistry
from researchops.config import get_settings
from researchops.llm.providers import BaseLLM, ChatMessage, ChatResponse, ToolCall
from researchops.observability.trace import (
    BudgetExceededError,
    LlmSpan,
    Trace,
    TracedLLM,
    TracedToolRegistry,
    traced_run_agent,
)


class _FakeBaseLLM(BaseLLM):
    """Pops one scripted ``ChatResponse`` per ``chat`` call (a real ``BaseLLM``)."""

    name = "fake"

    def __init__(self, responses: list[ChatResponse]) -> None:
        super().__init__("http://x", "k", "m")
        self._responses = list(responses)

    async def chat(self, messages: list[ChatMessage], **kw: Any) -> ChatResponse:
        if self._responses:
            return self._responses.pop(0)
        return ChatResponse(content="", model="fake")


def _resp(content: str, *, i: int = 0, o: int = 0) -> ChatResponse:
    return ChatResponse(content=content, model="fake", input_tokens=i, output_tokens=o)


def _tool_call(name: str, arguments: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        content="",
        model="fake",
        input_tokens=50,
        output_tokens=10,
        tool_calls=[ToolCall(id="c1", name=name, arguments=arguments)],
        raw_tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    )


def test_trace_summary_totals_and_cost() -> None:
    trace = Trace(task="t")
    trace.record_llm(LlmSpan(node="planner", model="m", input_tokens=100, output_tokens=10, latency_s=0.1))
    trace.record_llm(LlmSpan(node="executor", model="m", input_tokens=200, output_tokens=30, latency_s=0.2))
    trace.record_llm(LlmSpan(node="reporter", model="m", input_tokens=300, output_tokens=60, latency_s=0.3))

    s = trace.summary()
    assert s["input_tokens"] == 600
    assert s["output_tokens"] == 100
    assert s["total_tokens"] == 700
    assert s["llm_calls"] == 3
    # Explicit prices are respected (cross-provider comparison).
    assert trace.estimated_cost_usd(input_per_1m=0.27, output_per_1m=1.10) == pytest.approx(
        (600 * 0.27 + 100 * 1.10) / 1_000_000
    )
    # Defaults come from the (configurable) settings, not a hardcoded constant.
    cfg = get_settings()
    assert trace.estimated_cost_usd() == pytest.approx(
        (600 * cfg.llm_input_price_per_1m + 100 * cfg.llm_output_price_per_1m) / 1_000_000
    )
    assert set(s["per_node"]) == {"planner", "executor", "reporter"}


async def test_traced_llm_records_node_from_system_prompt() -> None:
    trace = Trace(task="t")
    inner = _FakeBaseLLM(
        [_resp("ok", i=5, o=2), _resp("ok", i=5, o=2), _resp("ok", i=5, o=2)]
    )
    llm = TracedLLM(inner, trace)

    await llm.chat([ChatMessage(role="system", content=PLANNER_PROMPT), ChatMessage(role="user", content="t")])
    await llm.chat([ChatMessage(role="system", content=SYSTEM_PROMPT), ChatMessage(role="user", content="t")])
    await llm.chat([ChatMessage(role="system", content=REPORTER_PROMPT), ChatMessage(role="user", content="e")])

    assert [s.node for s in trace.llm_spans] == ["planner", "executor", "reporter"]
    assert trace.total_tokens == 21  # (5 + 2) * 3


async def test_traced_run_agent_wires_llm_and_tool_spans() -> None:
    async def rag_handler(query: str, top_k: int = 5) -> str:
        return "[1] evidence"

    registry = ToolRegistry(
        [
            Tool(
                name="rag_search",
                description="d",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=rag_handler,
            )
        ]
    )
    responses = [
        _resp("- search\n", i=10, o=5),  # planner
        _tool_call("rag_search", {"query": "x"}),  # executor -> tool
        _resp("done", i=20, o=5),  # executor -> finish
        _resp("# Report", i=30, o=10),  # reporter
    ]
    llm = _FakeBaseLLM(responses)

    state, trace = await traced_run_agent("task", llm=llm, registry=registry, max_iterations=5)

    assert state.finished is True
    assert len(trace.llm_spans) == 4
    assert len(trace.tool_spans) == 1
    assert trace.tool_spans[0].name == "rag_search"
    assert {s.node for s in trace.llm_spans} == {"planner", "executor", "reporter"}
    assert trace.total_tokens == 15 + 60 + 25 + 40


async def test_traced_llm_attributes_multi_agent_nodes() -> None:
    """Supervisor/specialist prompts are attributed, not collapsed to "llm"."""
    trace = Trace(task="t")
    inner = _FakeBaseLLM([_resp("ok", i=5, o=2)] * 4)
    llm = TracedLLM(inner, trace)

    for prompt in (SUPERVISOR_PROMPT, RESEARCHER_PROMPT, LABOPS_PROMPT, MULTI_REPORTER_PROMPT):
        await llm.chat([ChatMessage(role="system", content=prompt), ChatMessage(role="user", content="t")])

    assert [s.node for s in trace.llm_spans] == ["supervisor", "researcher", "labops", "reporter"]


async def test_traced_registry_subset_forwards_and_traces() -> None:
    """``subset`` returns a traced registry (multi-agent specialists stay observable)."""
    async def rag_handler(query: str, top_k: int = 5) -> str:
        return "[1] evidence"

    registry = ToolRegistry(
        [
            Tool(
                name="rag_search",
                description="d",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=rag_handler,
            )
        ]
    )
    trace = Trace(task="t")
    traced = TracedToolRegistry(registry, trace)

    subset = traced.subset(["rag_search"])
    out = await subset.execute("rag_search", {"query": "x"})

    assert out == "[1] evidence"
    assert len(trace.tool_spans) == 1
    assert trace.tool_spans[0].name == "rag_search"


def test_trace_budget_cap_raises_when_exceeded() -> None:
    trace = Trace(task="t", budget_usd=1e-12)
    trace.record_llm(LlmSpan(node="planner", model="m", input_tokens=1, output_tokens=0, latency_s=0.1))
    with pytest.raises(BudgetExceededError):
        trace.check_budget()


def test_trace_budget_zero_is_unlimited() -> None:
    trace = Trace(task="t", budget_usd=0.0)
    trace.record_llm(LlmSpan(node="planner", model="m", input_tokens=1_000_000, output_tokens=0, latency_s=0.1))
    trace.check_budget()  # must not raise
