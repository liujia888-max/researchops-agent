"""Tests for the observability layer (Trace / TracedLLM / TracedToolRegistry)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from researchops.agent.graph import PLANNER_PROMPT, REPORTER_PROMPT, SYSTEM_PROMPT
from researchops.agent.tools import Tool, ToolRegistry
from researchops.llm.providers import BaseLLM, ChatMessage, ChatResponse, ToolCall
from researchops.observability.trace import LlmSpan, Trace, TracedLLM, traced_run_agent


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
    assert trace.estimated_cost_usd() == pytest.approx((600 * 0.27 + 100 * 1.10) / 1_000_000)
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
