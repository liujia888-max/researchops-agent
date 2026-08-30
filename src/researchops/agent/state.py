"""Agent state: the Pydantic state threaded through a LangGraph run."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    """One conversation turn. Tool-calling turns carry the extra fields the LLM
    layer echoes back (assistant ``tool_calls``, tool-result ``tool_call_id``/``name``).
    """

    role: str  # system | user | assistant | tool
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)


class ToolResult(BaseModel):
    """A single executed tool call and its text output (the evidence record)."""

    tool: str
    arguments: dict[str, Any]
    output: str


class AgentState(BaseModel):
    """Everything a run carries between nodes."""

    task: str
    plan: list[str] = Field(default_factory=list)
    messages: list[AgentMessage] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    iteration: int = 0
    max_iterations: int = 10
    # The tool call awaiting execution in the tools node: {"name", "arguments"}.
    pending_tool: dict[str, Any] | None = None
    finished: bool = False
    final_report: str = ""
