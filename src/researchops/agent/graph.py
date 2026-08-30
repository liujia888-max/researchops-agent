"""LangGraph state machine: Planner -> Executor <-> Tools -> Reporter.

The plan's four roles map to nodes:
- ``planner``  — turns the task into an ordered list of steps (one-shot).
- ``executor`` — the ReAct loop: given the plan + history, asks the LLM for the next
  tool call or for the go-ahead to finish (this is where reflection lives — every
  iteration re-reads the accumulated evidence and re-decides).
- ``tools``    — executes the pending tool call and records the evidence.
- ``reporter`` — synthesizes the accumulated evidence into a citation-bearing report.

The graph is built by a factory so the LLM and tool registry are injected (and can be
fakes in tests). The compiled graph is returned; ``runner.run_agent`` is the ergonomic
entrypoint.
"""

from __future__ import annotations

import re
from typing import Any

from langgraph.graph import END, START, StateGraph

from researchops.agent.state import AgentMessage, AgentState, ToolResult
from researchops.agent.tools import ToolRegistry
from researchops.llm.providers import BaseLLM, ChatMessage

SYSTEM_PROMPT = """You are ResearchOps Agent, an autonomous deep-learning experiment orchestrator.

You are given a research task. Use the available tools to gather evidence, then answer.

Rules:
- Gather evidence with `rag_search` (paper library) and the labops tools (remote GPU lab).
- Never fabricate numbers: every numeric claim must come from a tool result.
- Cite retrieved chunks by their [n] number exactly as `rag_search` returned them.
- To reproduce or evaluate a model, call `run_experiment` once — it submits, polls to
  completion, parses metrics, and persists them, returning the numbers in a single step.
- `run_experiment`, `submit_job`, and `cancel_job` are destructive; only call them when
  the task explicitly asks to run a job. Use `job_status`/`tail_log` only to inspect.
- Stop calling tools once you have enough evidence to answer; then give a short summary."""

PLANNER_PROMPT = """Break the task into a short, ordered list of concrete steps, each doable
with a single tool call (rag_search or a labops tool). Return ONLY a bullet list, one
step per line, no preamble."""

REPORTER_PROMPT = """You are writing a research report. Synthesize the evidence below into a
concise, well-structured report that directly answers the task. Cite evidence with [1][2]...
matching the numbering in the rag_search results. Do not invent numbers absent from the
evidence. Use markdown headings and finish with a short "Conclusion"."""

_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s*")


def _parse_plan(content: str) -> list[str]:
    steps: list[str] = []
    for line in content.splitlines():
        step = _BULLET.sub("", line).strip()
        if step and len(steps) < 8:
            steps.append(step)
    return steps


def _to_chat_message(message: AgentMessage) -> ChatMessage:
    return ChatMessage(
        role=message.role,
        content=message.content,
        name=message.name,
        tool_call_id=message.tool_call_id,
        tool_calls=message.tool_calls,
    )


def build_agent(
    llm: BaseLLM,
    registry: ToolRegistry,
    *,
    max_iterations: int = 10,
) -> Any:
    """Compile the agent graph against an injected LLM and tool registry."""

    async def planner(state: AgentState) -> dict[str, Any]:
        resp = await llm.chat(
            [
                ChatMessage(role="system", content=PLANNER_PROMPT),
                ChatMessage(role="user", content=state.task),
            ],
            temperature=0.0,
            max_tokens=512,
        )
        plan = _parse_plan(resp.content)
        note = "Plan:\n" + "\n".join(f"- {s}" for s in plan) if plan else "(no plan)"
        return {
            "plan": plan,
            "messages": state.messages + [AgentMessage(role="assistant", content=note)],
        }

    async def executor(state: AgentState) -> dict[str, Any]:
        if state.iteration >= state.max_iterations:
            msg = AgentMessage(role="assistant", content="(iteration budget exhausted)")
            return {
                "messages": state.messages + [msg],
                "pending_tool": None,
                "finished": True,
            }

        messages = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
        messages.append(ChatMessage(role="user", content=state.task))
        messages.extend(_to_chat_message(m) for m in state.messages)

        resp = await llm.chat(
            messages,
            tools=registry.schemas(),
            temperature=0.0,
            max_tokens=1024,
        )

        if resp.tool_calls:
            call = resp.tool_calls[0]
            # Echo back exactly the one tool_call we are about to execute, so the next
            # turn pairs one assistant tool_call with one tool message. OpenAI-compatible
            # APIs reject an assistant tool_calls message that has more ids than following
            # tool replies; we run one tool per step and re-decide the rest next iteration.
            assistant = AgentMessage(
                role="assistant", content=resp.content, tool_calls=resp.raw_tool_calls[:1]
            )
            return {
                "messages": state.messages + [assistant],
                "pending_tool": {"name": call.name, "arguments": call.arguments},
            }

        assistant = AgentMessage(role="assistant", content=resp.content)
        return {
            "messages": state.messages + [assistant],
            "pending_tool": None,
            "finished": True,
        }

    async def tools(state: AgentState) -> dict[str, Any]:
        pending = state.pending_tool
        if pending is None:
            return {}
        name = pending["name"]
        arguments = pending["arguments"]
        output = await registry.execute(name, arguments)

        call_id = ""
        if state.messages and state.messages[-1].tool_calls:
            call_id = str(state.messages[-1].tool_calls[0].get("id", ""))

        tool_message = AgentMessage(
            role="tool", content=output, name=name, tool_call_id=call_id
        )
        return {
            "messages": state.messages + [tool_message],
            "tool_results": state.tool_results + [ToolResult(tool=name, arguments=arguments, output=output)],
            "iteration": state.iteration + 1,
            "pending_tool": None,
        }

    async def reporter(state: AgentState) -> dict[str, Any]:
        evidence = "\n\n".join(
            f"### {r.tool}({r.arguments})\n{r.output}" for r in state.tool_results
        )
        if not evidence:
            evidence = "(no tool evidence was gathered)"
        resp = await llm.chat(
            [
                ChatMessage(role="system", content=REPORTER_PROMPT),
                ChatMessage(role="user", content=f"Task: {state.task}\n\nEvidence:\n{evidence}"),
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        return {"final_report": resp.content}

    def route_executor(state: AgentState) -> str:
        return "tools" if state.pending_tool is not None else "reporter"

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("tools", tools)
    graph.add_node("reporter", reporter)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", route_executor, {"tools": "tools", "reporter": "reporter"})
    graph.add_edge("tools", "executor")
    graph.add_edge("reporter", END)

    return graph.compile()
