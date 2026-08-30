"""Tests for the agent runtime against a scripted FakeLLM + fake tools (no live
LLM / Qdrant / GPU host needed)."""

from __future__ import annotations

from typing import Any

import pytest

from researchops.agent.graph import _parse_plan
from researchops.agent.runner import run_agent
from researchops.agent.tools import Tool, ToolRegistry
from researchops.llm.providers import ChatResponse, ToolCall


class FakeLLM:
    """Pops one scripted response per ``chat`` call, in order."""

    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def chat(self, messages: list[Any], **kwargs: Any) -> ChatResponse:
        self.calls += 1
        if self._responses:
            return self._responses.pop(0)
        return ChatResponse(content="", model="fake")


def _make_tool(name: str, output: str, *, calls: list[dict[str, Any]] | None = None) -> Tool:
    async def handler(**kwargs: Any) -> str:
        if calls is not None:
            calls.append(kwargs)
        return output

    return Tool(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )


def _tool_call(name: str, arguments: dict[str, Any], call_id: str = "c1") -> ChatResponse:
    import json

    return ChatResponse(
        content="",
        model="fake",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        raw_tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    )


# --------------------------------------------------------------------------- #
# _parse_plan
# --------------------------------------------------------------------------- #
def test_parse_plan_strips_bullets() -> None:
    assert _parse_plan("- search Restormer\n1. run job\n* report\n") == [
        "search Restormer",
        "run job",
        "report",
    ]


def test_parse_plan_empty() -> None:
    assert _parse_plan("") == []


# --------------------------------------------------------------------------- #
# ToolRegistry
# --------------------------------------------------------------------------- #
async def test_registry_schemas_and_execute() -> None:
    registry = ToolRegistry([_make_tool("t", "ok")])
    assert registry.names() == ["t"]
    assert registry.schemas() == [
        {
            "type": "function",
            "function": {
                "name": "t",
                "description": "t tool",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]
    assert await registry.execute("t", {}) == "ok"


async def test_registry_unknown_tool_raises() -> None:
    registry = ToolRegistry()
    with pytest.raises(KeyError):
        await registry.execute("nope", {})


async def test_registry_swallows_handler_error() -> None:
    async def boom(**kwargs: Any) -> str:
        raise RuntimeError("boom")

    registry = ToolRegistry(
        [Tool(name="t", description="d", parameters={"type": "object", "properties": {}, "required": []}, handler=boom)]
    )
    out = await registry.execute("t", {})
    assert out.startswith("error in tool t")


# --------------------------------------------------------------------------- #
# End-to-end agent run
# --------------------------------------------------------------------------- #
async def test_run_agent_single_tool_call_end_to_end() -> None:
    tool_calls: list[dict[str, Any]] = []
    async def rag_handler(query: str, top_k: int = 5) -> str:
        tool_calls.append({"query": query, "top_k": top_k})
        return "[1] (restormer p7) Restormer on CBSD68: σ=25 PSNR 31.79"

    registry = ToolRegistry(
        [
            Tool(
                name="rag_search",
                description="search papers",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
                handler=rag_handler,
            )
        ]
    )

    llm = FakeLLM(
        [
            ChatResponse(content="- search for Restormer CBSD68\n- report PSNR\n", model="fake"),
            _tool_call("rag_search", {"query": "Restormer CBSD68"}),
            ChatResponse(content="Found 31.79 dB", model="fake"),
            ChatResponse(content="# Report\n\nRestormer: 31.79 dB [1].\n\n## Conclusion\nDone.", model="fake"),
        ]
    )

    state = await run_agent("What is Restormer's PSNR on CBSD68?", llm=llm, registry=registry)

    assert state.final_report == "# Report\n\nRestormer: 31.79 dB [1].\n\n## Conclusion\nDone."
    assert len(state.tool_results) == 1
    assert state.tool_results[0].tool == "rag_search"
    assert tool_calls == [{"query": "Restormer CBSD68", "top_k": 5}]


async def test_run_agent_stops_at_max_iterations() -> None:
    registry = ToolRegistry([_make_tool("t", "evidence")])
    tool_call = _tool_call("t", {})
    llm = FakeLLM(
        [
            ChatResponse(content="- step one\n", model="fake"),  # planner
            tool_call,  # executor iter 0
            tool_call,  # executor iter 1
            ChatResponse(content="report body", model="fake"),  # reporter
        ]
    )

    state = await run_agent("task", llm=llm, registry=registry, max_iterations=2)

    assert state.iteration == 2
    assert len(state.tool_results) == 2
    assert state.finished is True
    assert state.final_report == "report body"
