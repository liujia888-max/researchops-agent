"""Ergonomic entrypoint: run the agent end-to-end and return the final state."""

from __future__ import annotations

from researchops.agent.graph import build_agent
from researchops.agent.multi import MultiAgentState, build_multi_agent
from researchops.agent.state import AgentState
from researchops.agent.tools import ToolRegistry
from researchops.llm.providers import BaseLLM


async def run_agent(
    task: str,
    *,
    llm: BaseLLM,
    registry: ToolRegistry,
    max_iterations: int = 10,
    max_retries: int = 2,
    retry_backoff_s: float = 1.0,
) -> AgentState:
    """Run the agent on a task and return the final state (incl. ``final_report``)."""
    app = build_agent(
        llm,
        registry,
        max_iterations=max_iterations,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )
    result = await app.ainvoke(AgentState(task=task, max_iterations=max_iterations))
    if isinstance(result, AgentState):
        return result
    return AgentState.model_validate(result)


async def run_multi_agent(
    task: str,
    *,
    llm: BaseLLM,
    registry: ToolRegistry,
    max_iterations: int = 10,
    max_retries: int = 2,
    retry_backoff_s: float = 1.0,
) -> MultiAgentState:
    """Run the supervisor + specialists team and return the final state."""
    app = build_multi_agent(
        llm,
        registry,
        max_iterations=max_iterations,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
    )
    result = await app.ainvoke(MultiAgentState(task=task, max_iterations=max_iterations))
    if isinstance(result, MultiAgentState):
        return result
    return MultiAgentState.model_validate(result)
