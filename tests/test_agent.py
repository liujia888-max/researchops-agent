"""Tests for the agent runtime against a scripted FakeLLM + fake tools (no live
LLM / Qdrant / GPU host needed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from researchops.agent.graph import _parse_plan
from researchops.agent.runner import run_agent
from researchops.agent.tools import Tool, ToolRegistry, make_labops_tools
from researchops.db.store import ExperimentStore
from researchops.labops.schemas import JobHandle, JobStatus
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
    llm = FakeLLM(
        [
            ChatResponse(content="- step one\n", model="fake"),  # planner
            _tool_call("t", {"n": 1}),  # executor iter 0
            _tool_call("t", {"n": 2}),  # executor iter 1
            ChatResponse(content="report body", model="fake"),  # reporter
        ]
    )

    state = await run_agent("task", llm=llm, registry=registry, max_iterations=2)

    assert state.iteration == 2
    assert len(state.tool_results) == 2
    assert state.finished is True
    assert state.final_report == "report body"


async def test_run_agent_repetition_guard_stops_repeated_call() -> None:
    """A second identical (tool, arguments) call is refused instead of looping."""
    calls: list[dict[str, Any]] = []

    async def handler(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "evidence"

    registry = ToolRegistry(
        [
            Tool(
                name="t",
                description="d",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=handler,
            )
        ]
    )
    repeated = _tool_call("t", {"q": "same"})
    llm = FakeLLM(
        [
            ChatResponse(content="- step\n", model="fake"),  # planner
            repeated,  # executor: first call
            repeated,  # executor: proposes the same call again -> guard stops
            ChatResponse(content="final", model="fake"),  # reporter
        ]
    )

    state = await run_agent("task", llm=llm, registry=registry, max_iterations=10)

    assert len(calls) == 1  # the handler ran exactly once
    assert len(state.tool_results) == 1
    assert state.finished is True
    assert state.final_report == "final"
    assert any("repetition guard" in m.content for m in state.messages)


async def test_run_agent_retries_failing_tool() -> None:
    """A primitive tool that raises a couple of times is retried before giving up."""
    attempts: list[int] = []

    async def flaky(**kwargs: Any) -> str:
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("connection reset")
        return "recovered"

    registry = ToolRegistry(
        [
            Tool(
                name="t",
                description="d",
                parameters={"type": "object", "properties": {}, "required": []},
                handler=flaky,
            )
        ]
    )
    llm = FakeLLM(
        [
            ChatResponse(content="- step\n", model="fake"),  # planner
            _tool_call("t", {}),  # executor
            ChatResponse(content="done", model="fake"),  # executor: finish
            ChatResponse(content="final", model="fake"),  # reporter
        ]
    )

    state = await run_agent(
        "task", llm=llm, registry=registry, max_iterations=5, retry_backoff_s=0.0
    )

    # max_retries defaults to 2 -> 1 initial + 2 retries = 3 attempts, success on the 3rd.
    assert len(attempts) == 3
    assert state.tool_results[-1].output == "recovered"


async def test_run_agent_echoes_single_tool_call_for_parallel_calls() -> None:
    """When the LLM emits several parallel tool calls, only the first is executed and
    the assistant turn echoes exactly one tool_call id — so a real OpenAI-compatible
    API sees a 1:1 assistant-tool_call <-> tool-message pairing (no 'insufficient tool
    messages' 400)."""
    registry = ToolRegistry([_make_tool("t1", "one"), _make_tool("t2", "two")])
    parallel = ChatResponse(
        content="",
        model="fake",
        tool_calls=[
            ToolCall(id="c1", name="t1", arguments={}),
            ToolCall(id="c2", name="t2", arguments={}),
        ],
        raw_tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "t1", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "t2", "arguments": "{}"}},
        ],
    )
    llm = FakeLLM(
        [
            ChatResponse(content="- step\n", model="fake"),  # planner
            parallel,  # executor: 2 parallel tool calls
            ChatResponse(content="done", model="fake"),  # executor: finish
            ChatResponse(content="final", model="fake"),  # reporter
        ]
    )

    state = await run_agent("task", llm=llm, registry=registry, max_iterations=5)

    assert len(state.tool_results) == 1
    assert state.tool_results[0].tool == "t1"
    assistants = [m for m in state.messages if m.role == "assistant" and m.tool_calls]
    assert len(assistants) == 1
    assert len(assistants[0].tool_calls) == 1


# --------------------------------------------------------------------------- #
# Destructive-tool approval gate (human-in-the-loop)
# --------------------------------------------------------------------------- #
class _FakeLabClient:
    """Records submit/cancel calls; returns the same models the real client would."""

    def __init__(self) -> None:
        self.submitted: list[tuple[str, str]] = []
        self.cancelled: list[str] = []

    async def submit_job(self, job_id: str, command: str) -> JobHandle:
        self.submitted.append((job_id, command))
        return JobHandle(job_id=job_id, running=True)

    async def cancel_job(self, job_id: str) -> JobStatus:
        self.cancelled.append(job_id)
        return JobStatus(job_id=job_id, running=False, log_path="", log_exists=False)


def _tools_by_name(client: _FakeLabClient, **kwargs: Any) -> dict[str, Tool]:
    return {t.name: t for t in make_labops_tools(client, **kwargs)}  # type: ignore[arg-type]


async def test_destructive_tools_deny_without_approver() -> None:
    client = _FakeLabClient()
    tools = _tools_by_name(client)

    out = await tools["submit_job"].handler(job_id="x", command="python train.py")
    assert out.startswith("REJECTED")
    assert client.submitted == []

    out = await tools["cancel_job"].handler(job_id="x")
    assert out.startswith("REJECTED")
    assert client.cancelled == []


async def test_destructive_tools_run_with_approver() -> None:
    client = _FakeLabClient()
    tools = _tools_by_name(client, approver=lambda name, args: True)

    out = await tools["submit_job"].handler(job_id="x", command="echo hi")
    assert '"job_id": "x"' in out
    assert client.submitted == [("x", "echo hi")]


async def test_approver_can_selectively_deny() -> None:
    client = _FakeLabClient()
    tools = _tools_by_name(client, approver=lambda name, args: args.get("job_id") == "safe")

    assert (await tools["submit_job"].handler(job_id="bad", command="rm -rf /")).startswith(
        "REJECTED"
    )
    await tools["submit_job"].handler(job_id="safe", command="echo ok")
    assert client.submitted == [("safe", "echo ok")]


async def test_async_approver_is_awaited() -> None:
    """An async approver (like the web server's) is awaited, not called and discarded."""
    client = _FakeLabClient()
    seen: list[str] = []

    async def approver(name: str, args: dict[str, Any]) -> bool:
        seen.append(name)
        return True

    tools = _tools_by_name(client, approver=approver)
    out = await tools["submit_job"].handler(job_id="x", command="echo hi")
    assert '"job_id": "x"' in out
    assert seen == ["submit_job"]
    assert client.submitted == [("x", "echo hi")]


async def test_submit_job_rejects_dangerous_command_even_with_approver() -> None:
    """The command policy blocks a dangerous command before approval is even asked."""
    client = _FakeLabClient()
    tools = _tools_by_name(client, approver=lambda name, args: True)

    out = await tools["submit_job"].handler(job_id="x", command="rm -rf /")

    assert out.startswith("REJECTED")
    assert "policy" in out
    assert client.submitted == []


# --------------------------------------------------------------------------- #
# run_experiment (deterministic submit->poll->persist tool)
# --------------------------------------------------------------------------- #
class _RunFakeClient:
    """Implements just the three methods run_and_collect calls on a LabClient."""

    def __init__(self) -> None:
        self.submitted: list[tuple[str, str]] = []

    async def submit_job(self, job_id: str, command: str) -> JobHandle:
        self.submitted.append((job_id, command))
        return JobHandle(job_id=job_id, running=True)

    async def job_status(self, job_id: str) -> JobStatus:
        # Already finished: the poll loop breaks immediately.
        return JobStatus(job_id=job_id, running=False, log_path="", log_exists=True)

    async def tail_log(self, job_id: str, lines: int = 50) -> str:
        return "{'Sigma': 25, 'Denoise_PSNR': 29.96, 'Denoise_SSIM': 0.86}\n"


def _run_experiment_tools(
    client: object, store: ExperimentStore, **kwargs: Any
) -> dict[str, Tool]:
    return {t.name: t for t in make_labops_tools(client, store=store, **kwargs)}  # type: ignore[arg-type]


async def test_run_experiment_denies_without_approver() -> None:
    client = _RunFakeClient()
    store = ExperimentStore(url="sqlite+aiosqlite:///:memory:")
    tools = _run_experiment_tools(client, store)

    out = await tools["run_experiment"].handler(experiment_name="e", job_id="j", command="x")
    assert out.startswith("REJECTED")
    assert client.submitted == []
    await store.close()


async def test_run_experiment_registered_only_with_store() -> None:
    client = _RunFakeClient()
    without = {t.name for t in make_labops_tools(client)}  # type: ignore[arg-type]
    assert "run_experiment" not in without

    store = ExperimentStore(url="sqlite+aiosqlite:///:memory:")
    with_store = {t.name for t in make_labops_tools(client, store=store)}  # type: ignore[arg-type]
    assert "run_experiment" in with_store
    await store.close()


async def test_run_experiment_runs_and_persists(tmp_path: Path) -> None:
    url = f"sqlite+aiosqlite:///{(tmp_path / 'e.db').as_posix()}"
    store = ExperimentStore(url=url)
    await store.init()

    client = _RunFakeClient()
    tools = _run_experiment_tools(client, store, approver=lambda name, args: True)

    out = await tools["run_experiment"].handler(
        experiment_name="exp1", job_id="j1", command="echo hi", task="demo"
    )

    assert '"status": "completed"' in out
    assert '"metrics"' in out
    assert client.submitted == [("j1", "echo hi")]

    exp = await store.get_experiment("exp1")
    assert exp is not None
    runs = await store.list_runs(exp.id)
    assert len(runs) == 1
    assert runs[0].status == "completed"
    metrics = await store.list_metrics(runs[0].id)
    assert [(m.name, m.value, m.sigma) for m in metrics] == [
        ("psnr", 29.96, 25),
        ("ssim", 0.86, 25),
    ]
    await store.close()


# --------------------------------------------------------------------------- #
# Reflection (optional post-report critique pass)
# --------------------------------------------------------------------------- #
async def test_run_agent_with_reflection_revises_report() -> None:
    """With reflect=True, a reflector node runs after the reporter and its output wins."""
    registry = ToolRegistry([_make_tool("t", "evidence")])
    llm = FakeLLM(
        [
            ChatResponse(content="- step\n", model="fake"),  # planner
            _tool_call("t", {}),  # executor -> tool
            ChatResponse(content="done", model="fake"),  # executor -> finish
            ChatResponse(content="draft report", model="fake"),  # reporter
            ChatResponse(content="revised final report", model="fake"),  # reflector
        ]
    )

    state = await run_agent("task", llm=llm, registry=registry, max_iterations=5, reflect=True)

    assert state.final_report == "revised final report"


async def test_run_agent_without_reflection_skips_reflector() -> None:
    """Default reflect=False: the reporter's output is final (no extra LLM call)."""
    registry = ToolRegistry([_make_tool("t", "evidence")])
    llm = FakeLLM(
        [
            ChatResponse(content="- step\n", model="fake"),  # planner
            _tool_call("t", {}),  # executor -> tool
            ChatResponse(content="done", model="fake"),  # executor -> finish
            ChatResponse(content="draft report", model="fake"),  # reporter
        ]
    )

    state = await run_agent("task", llm=llm, registry=registry, max_iterations=5)

    assert state.final_report == "draft report"
    assert llm.calls == 4  # planner + executor x2 + reporter (no reflector)
