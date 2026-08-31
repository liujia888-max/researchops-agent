"""Tests for the streaming agent wrapper (``agent.stream.stream_agent``)."""

from __future__ import annotations

import json
from typing import Any

from researchops.agent.stream import stream_agent
from researchops.agent.tools import Tool, ToolRegistry
from researchops.llm.providers import BaseLLM, ChatResponse, ToolCall
from researchops.observability.trace import Trace, TracedLLM, TracedToolRegistry


class FakeLLM(BaseLLM):
    """Pops one scripted response per ``chat`` call, in order."""

    name = "fake"

    def __init__(self, responses: list[ChatResponse]) -> None:
        super().__init__("http://x", "k", "m")
        self._responses = list(responses)

    async def chat(self, messages: list[Any], **kwargs: Any) -> ChatResponse:
        if self._responses:
            return self._responses.pop(0)
        return ChatResponse(content="", model="fake")


def _tool_call(name: str, arguments: dict[str, Any]) -> ChatResponse:
    return ChatResponse(
        content="",
        model="fake",
        tool_calls=[ToolCall(id="c1", name=name, arguments=arguments)],
        raw_tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    )


def _rag_registry() -> ToolRegistry:
    async def handler(query: str, top_k: int = 5) -> str:
        return f"[1] evidence for {query}"

    return ToolRegistry(
        [
            Tool(
                name="rag_search",
                description="search papers",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                handler=handler,
            )
        ]
    )


async def test_stream_agent_emits_plan_tool_and_report() -> None:
    llm = FakeLLM(
        [
            ChatResponse(content="- step one\n- step two\n", model="fake"),  # planner
            _tool_call("rag_search", {"query": "Restormer"}),  # executor -> tool
            ChatResponse(content="found", model="fake"),  # executor -> finish
            ChatResponse(content="# Report\nDone [1]", model="fake"),  # reporter
        ]
    )

    events = [
        e async for e in stream_agent("task", llm=llm, registry=_rag_registry(), max_iterations=5)
    ]

    assert [e["event"] for e in events] == ["plan", "tool_call", "tool_result", "report"]
    assert events[0]["plan"] == ["step one", "step two"]
    assert events[1]["name"] == "rag_search"
    assert events[1]["arguments"] == {"query": "Restormer"}
    assert events[2]["name"] == "rag_search"
    assert "evidence for Restormer" in events[2]["output"]
    assert events[3]["report"] == "# Report\nDone [1]"


async def test_stream_agent_with_tracing_populates_trace() -> None:
    llm = FakeLLM(
        [
            ChatResponse(content="- search\n", model="fake", input_tokens=10, output_tokens=5),
            _tool_call("rag_search", {"query": "q"}),
            ChatResponse(content="ok", model="fake", input_tokens=20, output_tokens=5),
            ChatResponse(content="report", model="fake", input_tokens=30, output_tokens=10),
        ]
    )
    trace = Trace(task="task")
    traced_llm = TracedLLM(llm, trace)
    traced_registry = TracedToolRegistry(_rag_registry(), trace)

    events = [
        e async for e in stream_agent("task", llm=traced_llm, registry=traced_registry, max_iterations=5)
    ]

    assert [e["event"] for e in events] == ["plan", "tool_call", "tool_result", "report"]
    assert len(trace.llm_spans) == 4  # planner + executor x2 + reporter
    assert len(trace.tool_spans) == 1
    assert trace.tool_spans[0].name == "rag_search"
    assert trace.total_tokens == 15 + 25 + 40
