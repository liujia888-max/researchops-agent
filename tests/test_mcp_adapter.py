"""Tests for the MCP client adapter's tool-building path (no live server needed).

``LabopsMCPClient`` itself spawns a ``python -m researchops.mcp`` subprocess over
stdio, which is awkward to unit-test in isolation; these tests exercise the layer that
wraps the server's tools as agent ``Tool`` objects (``make_mcp_labops_tools`` /
``build_default_tools(via_mcp=True)``) against a duck-typed fake.
"""

from __future__ import annotations

from typing import Any

import pytest

from researchops.agent.tools import build_default_tools, make_mcp_labops_tools
from researchops.db.store import ExperimentStore
from researchops.mcp.client import LabopsMCPClient, RemoteTool


class _FakeMCP(LabopsMCPClient):
    """A LabopsMCPClient-shaped stub: no subprocess, just list_tools + call_tool."""

    def __init__(self, tools: list[RemoteTool]) -> None:
        self._tools = tools
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[RemoteTool]:
        return self._tools

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        self.calls.append((name, arguments or {}))
        return f"ok:{name}"


class _FakeRetriever:
    async def retrieve(self, query: str, rerank_top_k: int = 5) -> list[Any]:
        return []


_GPU = RemoteTool(
    name="gpu_info",
    description="Read GPU state.",
    input_schema={"type": "object", "properties": {}, "required": []},
)
_SUBMIT = RemoteTool(
    name="submit_job",
    description="Launch a job.",
    input_schema={
        "type": "object",
        "properties": {"job_id": {"type": "string"}, "command": {"type": "string"}},
        "required": ["job_id", "command"],
    },
)


async def test_make_mcp_labops_tools_wraps_each_tool() -> None:
    mcp = _FakeMCP([_GPU, _SUBMIT])
    tools = {t.name: t for t in await make_mcp_labops_tools(mcp)}

    assert set(tools) == {"gpu_info", "submit_job"}
    assert await tools["gpu_info"].handler() == "ok:gpu_info"
    assert mcp.calls == [("gpu_info", {})]


async def test_mcp_destructive_tool_denied_without_approver() -> None:
    mcp = _FakeMCP([_SUBMIT])
    tools = {t.name: t for t in await make_mcp_labops_tools(mcp)}

    out = await tools["submit_job"].handler(job_id="x", command="python train.py")
    assert out.startswith("REJECTED")
    assert mcp.calls == []  # never reached the MCP server


async def test_mcp_destructive_tool_runs_with_approver() -> None:
    mcp = _FakeMCP([_SUBMIT])
    tools = {
        t.name: t for t in await make_mcp_labops_tools(mcp, approver=lambda name, args: True)
    }

    out = await tools["submit_job"].handler(job_id="x", command="echo hi")
    assert out == "ok:submit_job"
    assert mcp.calls == [("submit_job", {"job_id": "x", "command": "echo hi"})]


async def test_mcp_labops_tools_adds_run_experiment_with_store() -> None:
    store = ExperimentStore(url="sqlite+aiosqlite:///:memory:")
    mcp = _FakeMCP([_GPU])
    tools = {t.name: t for t in await make_mcp_labops_tools(mcp, store=store)}

    assert "run_experiment" in tools
    await store.close()


async def test_build_default_tools_via_mcp() -> None:
    mcp = _FakeMCP([_GPU])
    registry = await build_default_tools(_FakeRetriever(), mcp, via_mcp=True)

    assert set(registry.names()) == {"rag_search", "gpu_info"}


async def test_build_default_tools_via_mcp_rejects_non_mcp() -> None:
    with pytest.raises(TypeError):
        await build_default_tools(_FakeRetriever(), object(), via_mcp=True)


async def test_build_default_tools_direct_rejects_non_labclient() -> None:
    with pytest.raises(TypeError):
        await build_default_tools(_FakeRetriever(), object(), via_mcp=False)
