"""Tests for the agent trajectory eval (pure metrics + scripted harness)."""

from __future__ import annotations

import json
from typing import Any

from researchops.agent.tools import Tool, ToolRegistry
from researchops.eval.agent_eval import (
    GoldenTask,
    TaskOutcome,
    compute_report,
    load_tasks,
    run_eval,
    task_passed,
    tool_precision,
    tool_recall,
)
from researchops.llm.providers import BaseLLM, ChatMessage, ChatResponse, ToolCall


class _FakeBaseLLM(BaseLLM):
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


def _outcome(**kw: Any) -> TaskOutcome:
    return TaskOutcome(
        task_id=kw.get("task_id", "t"),
        finished=kw.get("finished", True),
        final_report=kw.get("final_report", ""),
        called_tools=kw.get("called_tools", []),
        steps=kw.get("steps", 0),
        input_tokens=kw.get("input_tokens", 0),
        output_tokens=kw.get("output_tokens", 0),
        cost_usd=kw.get("cost_usd", 0.0),
        wall_s=kw.get("wall_s", 0.0),
    )


def test_tool_recall_and_precision() -> None:
    assert tool_recall(["a", "b"], ["a", "c"]) == 0.5
    assert tool_recall([], ["a"]) == 1.0
    assert tool_precision(["a", "b"], ["a", "c"]) == 0.5
    assert tool_precision(["a"], []) == 0.0


def test_task_passed_requires_facts() -> None:
    task = GoldenTask(id="t", task="q", expected_facts=["31.79"])
    assert task_passed(task, _outcome(finished=True, final_report="PSNR 31.79 dB")) is True
    assert task_passed(task, _outcome(finished=True, final_report="PSNR 30.00 dB")) is False
    assert task_passed(task, _outcome(finished=False, final_report="PSNR 31.79 dB")) is False


def test_compute_report_aggregates() -> None:
    tasks = [GoldenTask(id="a", task="q", expected_tools=["rag"], expected_facts=["31.79"])]
    outcomes = [
        _outcome(
            finished=True,
            final_report="31.79",
            called_tools=["rag"],
            steps=2,
            input_tokens=100,
            output_tokens=50,
        )
    ]
    r = compute_report(tasks, outcomes)
    assert r.completion_rate == 1.0
    assert r.answer_accuracy == 1.0
    assert r.tool_recall == 1.0
    assert r.tool_precision == 1.0
    assert r.avg_steps == 2.0
    assert r.avg_tokens == 150.0


async def test_run_eval_scripted_single_task() -> None:
    async def rag_handler(query: str, top_k: int = 5) -> str:
        return "[1] 31.79"

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
        _resp("found 31.79", i=20, o=5),  # executor -> finish
        _resp("# Report\nPSNR 31.79 dB", i=30, o=10),  # reporter
    ]
    tasks = [GoldenTask(id="t1", task="q", expected_tools=["rag_search"], expected_facts=["31.79"])]

    report = await run_eval(tasks, llm=_FakeBaseLLM(responses), registry=registry, max_iterations=5)

    assert report.completion_rate == 1.0
    assert report.answer_accuracy == 1.0
    assert report.tool_recall == 1.0
    assert report.tool_precision == 1.0
    assert report.avg_steps == 1.0


def test_load_tasks_from_json(tmp_path: Any) -> None:
    p = tmp_path / "tasks.json"
    p.write_text(
        json.dumps(
            [
                {"id": "a", "task": "q1", "expected_tools": ["rag_search"], "expected_facts": ["31.79"]},
                {"id": "b", "task": "q2"},
            ]
        ),
        encoding="utf-8",
    )
    tasks = load_tasks(str(p))
    assert len(tasks) == 2
    assert tasks[0].expected_facts == ["31.79"]
    assert tasks[1].expected_tools == []
