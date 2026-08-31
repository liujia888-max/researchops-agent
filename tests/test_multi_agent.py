"""Tests for the supervisor + specialists multi-agent graph."""

from __future__ import annotations

import json
from typing import Any

from researchops.agent.multi import MultiAgentState, build_multi_agent, run_specialist
from researchops.agent.tools import Tool, ToolRegistry
from researchops.llm.providers import BaseLLM, ChatMessage, ChatResponse, ToolCall


class _FakeLLM(BaseLLM):
    name = "fake"

    def __init__(self, responses: list[ChatResponse]) -> None:
        super().__init__("http://x", "k", "m")
        self._responses = list(responses)

    async def chat(self, messages: list[ChatMessage], **kw: Any) -> ChatResponse:
        if self._responses:
            return self._responses.pop(0)
        return ChatResponse(content="", model="fake")


def _resp(content: str) -> ChatResponse:
    return ChatResponse(content=content, model="fake")


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


def _registry() -> ToolRegistry:
    async def rag_handler(query: str, top_k: int = 5) -> str:
        return "[1] PSNR 31.79"

    async def gpu_handler() -> str:
        return "RTX 5090"

    return ToolRegistry(
        [
            Tool(
                name="rag_search",
                description="d",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=rag_handler,
            ),
            Tool(
                name="gpu_info",
                description="d",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=gpu_handler,
            ),
        ]
    )


async def test_run_specialist_restricts_to_subset() -> None:
    registry = _registry()
    llm = _FakeLLM(
        [
            _tool_call("rag_search", {"query": "method"}),
            _resp("self-augmented noisy image network"),
        ]
    )
    result = await run_specialist(
        llm,
        registry,
        role="researcher",
        system_prompt="sys",
        task="what method?",
        brief="find the method",
        tool_names=["rag_search"],
        max_iterations=5,
    )
    assert result.summary == "self-augmented noisy image network"
    assert [f.tool for f in result.findings] == ["rag_search"]
    assert result.findings[0].output == "[1] PSNR 31.79"


async def test_run_specialist_skips_unregistered_tools() -> None:
    registry = _registry()
    llm = _FakeLLM([_tool_call("rag_search", {"query": "m"}), _resp("done")])
    # "memory_search" is not in the registry; the specialist must not crash on it.
    result = await run_specialist(
        llm,
        registry,
        role="researcher",
        system_prompt="sys",
        task="t",
        brief="b",
        tool_names=["rag_search", "memory_search"],
        max_iterations=5,
    )
    assert [f.tool for f in result.findings] == ["rag_search"]


async def test_build_multi_agent_end_to_end() -> None:
    registry = _registry()
    llm = _FakeLLM(
        [
            _resp(
                json.dumps(
                    {
                        "specialists": ["researcher"],
                        "briefs": {"researcher": "find the method name"},
                    }
                )
            ),
            _tool_call("rag_search", {"query": "method"}),
            _resp("self-augmented noisy image network"),
            _resp("# Report\nself-augmented method"),
        ]
    )
    app = build_multi_agent(llm, registry, max_iterations=5)
    state = await app.ainvoke(MultiAgentState(task="what method?", max_iterations=5))
    assert state["roles"] == ["researcher"]
    assert state["final_report"] == "# Report\nself-augmented method"
    assert len(state["findings"]["researcher"]) == 1


async def test_build_multi_agent_defaults_to_both_on_bad_json() -> None:
    registry = _registry()
    llm = _FakeLLM(
        [
            _resp("I have no idea what to route"),  # unparseable supervisor output
            _tool_call("rag_search", {"query": "m"}),  # researcher step 1
            _resp("researcher summary"),  # researcher done
            _tool_call("gpu_info", {}),  # labops step 1
            _resp("labops summary"),  # labops done
            _resp("# Report\ncombined"),
        ]
    )
    app = build_multi_agent(llm, registry, max_iterations=5)
    state = await app.ainvoke(MultiAgentState(task="do everything", max_iterations=5))
    assert sorted(state["roles"]) == ["labops", "researcher"]
    assert state["final_report"] == "# Report\ncombined"
