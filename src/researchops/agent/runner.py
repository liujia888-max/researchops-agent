"""Ergonomic entrypoint: run the agent end-to-end and return the final state."""

from __future__ import annotations

from researchops.agent.graph import build_agent
from researchops.agent.state import AgentState
from researchops.agent.tools import ToolRegistry
from researchops.llm.providers import BaseLLM


async def run_agent(
    task: str,
    *,
    llm: BaseLLM,
    registry: ToolRegistry,
    max_iterations: int = 10,
) -> AgentState:
    """Run the agent on a task and return the final state (incl. ``final_report``)."""
    app = build_agent(llm, registry, max_iterations=max_iterations)
    result = await app.ainvoke(AgentState(task=task, max_iterations=max_iterations))
    if isinstance(result, AgentState):
        return result
    return AgentState.model_validate(result)
