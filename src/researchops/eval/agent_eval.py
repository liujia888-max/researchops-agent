"""Agent trajectory evaluation.

Measures, over a golden set of tasks, the numbers the portfolio needs:

* ``completion_rate`` — fraction of runs that reached a terminal state.
* ``answer_accuracy`` — fraction whose final report is correct, judged either by a
  deterministic substring match against ``expected_facts`` (baseline) or by an LLM
  judge (``judge=True``) that checks correctness and faithfulness end-to-end.
* ``tool_recall`` / ``tool_precision`` — how well the agent picked the right tools.
* ``avg_steps`` / ``avg_tokens`` / ``avg_cost_usd`` / ``avg_wall_s`` — cost & latency.

The metric functions are pure (over ``GoldenTask`` + ``TaskOutcome``) and unit-tested
without any LLM/GPU. ``run_eval`` wires them to ``traced_run_agent`` so a real run
(DeepSeek + Qdrant + labops) produces the same numbers; a scripted ``BaseLLM`` makes
the harness deterministic in CI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

from researchops.agent.state import AgentState
from researchops.agent.tools import ToolRegistry
from researchops.llm.providers import BaseLLM, ChatMessage
from researchops.observability.trace import Trace, traced_run_agent


@dataclass(frozen=True)
class GoldenTask:
    """One golden case: the task, the tools it should use, and facts its answer must state.

    ``expected_facts`` entries are either a literal substring (must appear in the
    report) or a list of acceptable alternatives (any one must appear). The list form
    covers paper-specific noise — e.g. ``["31.78", "31.79"]`` for a PSNR that appears
    rounded differently across tables.
    """

    id: str
    task: str
    expected_tools: list[str] = field(default_factory=list)
    expected_facts: list[str | list[str]] = field(default_factory=list)


@dataclass(frozen=True)
class TaskOutcome:
    """The measured result of one golden task.

    ``passed`` is ``None`` until a judge (substring or LLM) sets it; ``compute_report``
    falls back to the substring matcher when it is unset.
    """

    task_id: str
    finished: bool
    final_report: str
    called_tools: list[str]
    steps: int
    input_tokens: int
    output_tokens: int
    cost_usd: float
    wall_s: float
    passed: bool | None = None


def outcome_from_run(task: GoldenTask, state: AgentState, trace: Trace) -> TaskOutcome:
    """Reduce a finished agent run (state + trace) to a ``TaskOutcome``."""
    return TaskOutcome(
        task_id=task.id,
        finished=state.finished,
        final_report=state.final_report,
        called_tools=[r.tool for r in state.tool_results],
        steps=state.iteration,
        input_tokens=trace.input_tokens,
        output_tokens=trace.output_tokens,
        cost_usd=trace.estimated_cost_usd(),
        wall_s=trace.wall_s,
    )


def tool_recall(expected: list[str], called: list[str]) -> float:
    """Fraction of expected tools that were actually called."""
    if not expected:
        return 1.0
    return len(set(expected) & set(called)) / len(expected)


def tool_precision(expected: list[str], called: list[str]) -> float:
    """Fraction of called tools that were expected (0 when nothing was called)."""
    if not called:
        return 0.0
    return len(set(expected) & set(called)) / len(called)


def _fact_matches(report: str, fact: str | list[str]) -> bool:
    """True when a fact (literal or list of alternatives) appears in the report.

    Case-insensitive so English acronyms/terms match regardless of how the model
    capitalized them in the report.
    """
    alts = fact if isinstance(fact, list) else [fact]
    low = report.lower()
    return any(alt.lower() in low for alt in alts)


def task_passed(task: GoldenTask, outcome: TaskOutcome) -> bool:
    """True when the run finished and its report contains every expected fact."""
    if not outcome.finished:
        return False
    if not task.expected_facts:
        return True
    return all(_fact_matches(outcome.final_report, fact) for fact in task.expected_facts)


_JUDGE_PROMPT = """You are grading whether an AI research agent answered a task correctly and faithfully.

Task: {task}

Expected facts (alternatives joined by " / " are interchangeable; each bullet is required):
{expected_facts}

Agent's answer:
---
{report}
---

Judge ONLY correctness and faithfulness: does the answer address the task, state facts consistent with the expected facts, and avoid inventing numbers or claims absent from the expected facts? Reply with exactly one word: PASS or FAIL."""


def _render_facts(task: GoldenTask) -> str:
    lines = []
    for fact in task.expected_facts:
        alts = fact if isinstance(fact, list) else [fact]
        lines.append("- " + " / ".join(alts))
    return "\n".join(lines) or "- (no specific facts required)"


async def judge_report(llm: BaseLLM, task: GoldenTask, report: str) -> bool:
    """LLM-judge one report: correct + faithful to the task/facts -> True."""
    prompt = _JUDGE_PROMPT.format(
        task=task.task,
        expected_facts=_render_facts(task),
        report=report or "(empty)",
    )
    resp = await llm.chat(
        [ChatMessage(role="user", content=prompt)], temperature=0.0, max_tokens=8
    )
    return resp.content.strip().upper().startswith("PASS")


@dataclass
class EvalReport:
    """Aggregate metrics over the golden set."""

    outcomes: list[TaskOutcome]
    completion_rate: float
    answer_accuracy: float
    tool_recall: float
    tool_precision: float
    avg_steps: float
    avg_tokens: float
    avg_cost_usd: float
    avg_wall_s: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_tasks": len(self.outcomes),
            "completion_rate": self.completion_rate,
            "answer_accuracy": self.answer_accuracy,
            "tool_recall": self.tool_recall,
            "tool_precision": self.tool_precision,
            "avg_steps": self.avg_steps,
            "avg_tokens": self.avg_tokens,
            "avg_cost_usd": self.avg_cost_usd,
            "avg_wall_s": self.avg_wall_s,
        }


def compute_report(tasks: list[GoldenTask], outcomes: list[TaskOutcome]) -> EvalReport:
    """Aggregate per-task outcomes (same order as ``tasks``) into headline metrics."""
    pairs = list(zip(tasks, outcomes, strict=True))
    n = len(pairs)
    if n == 0:
        return EvalReport(
            outcomes=[],
            completion_rate=0.0,
            answer_accuracy=0.0,
            tool_recall=0.0,
            tool_precision=0.0,
            avg_steps=0.0,
            avg_tokens=0.0,
            avg_cost_usd=0.0,
            avg_wall_s=0.0,
        )
    return EvalReport(
        outcomes=[o for _, o in pairs],
        completion_rate=sum(1 for _, o in pairs if o.finished) / n,
        answer_accuracy=sum(
            1
            for t, o in pairs
            if (o.passed if o.passed is not None else task_passed(t, o))
        )
        / n,
        tool_recall=sum(tool_recall(t.expected_tools, o.called_tools) for t, o in pairs) / n,
        tool_precision=sum(tool_precision(t.expected_tools, o.called_tools) for t, o in pairs) / n,
        avg_steps=sum(o.steps for _, o in pairs) / n,
        avg_tokens=sum(o.input_tokens + o.output_tokens for _, o in pairs) / n,
        avg_cost_usd=sum(o.cost_usd for _, o in pairs) / n,
        avg_wall_s=sum(o.wall_s for _, o in pairs) / n,
    )


async def run_eval(
    tasks: list[GoldenTask],
    *,
    llm: BaseLLM,
    registry: ToolRegistry,
    max_iterations: int = 10,
    judge: bool = False,
    judge_llm: BaseLLM | None = None,
) -> EvalReport:
    """Run every golden task through the agent and aggregate the metrics.

    With ``judge``, each report's correctness is decided by an LLM judge (``judge_llm``
    or ``llm``); otherwise the deterministic substring matcher is used.
    """
    evaluator = judge_llm or llm
    outcomes: list[TaskOutcome] = []
    for task in tasks:
        state, trace = await traced_run_agent(
            task.task, llm=llm, registry=registry, max_iterations=max_iterations
        )
        outcome = outcome_from_run(task, state, trace)
        if judge:
            outcome = replace(
                outcome, passed=await judge_report(evaluator, task, state.final_report)
            )
        outcomes.append(outcome)
    return compute_report(tasks, outcomes)


def load_tasks(path: str) -> list[GoldenTask]:
    """Load golden tasks from a JSON file (a list of task objects).

    ``expected_facts`` entries may be a string or a list of alternative strings.
    """
    with open(path, encoding="utf-8") as f:
        raw: list[dict[str, Any]] = json.load(f)
    return [
        GoldenTask(
            id=str(item["id"]),
            task=str(item["task"]),
            expected_tools=[str(t) for t in item.get("expected_tools", [])],
            expected_facts=[
                [str(a) for a in fact] if isinstance(fact, list) else str(fact)
                for fact in item.get("expected_facts", [])
            ],
        )
        for item in raw
    ]
