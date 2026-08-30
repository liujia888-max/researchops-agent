"""Run the agent trajectory eval over the golden set (real LLM + tools).

Usage:
    python scripts/eval_agent.py [--max-iterations N]

Reads ``golden_set/agent_tasks.json``. Needs ``.env`` (DeepSeek key); the Qdrant +
inference services and the GPU host are optional — absent hosts degrade to structured
tool errors, so the harness still runs and reports.
"""

from __future__ import annotations

import argparse
import asyncio
import json

from researchops.agent.tools import build_default_tools
from researchops.eval.agent_eval import load_tasks, run_eval
from researchops.labops import LabClient, SshConnection
from researchops.llm import build_llm
from researchops.rag.retriever import Retriever


async def main(max_iterations: int) -> None:
    tasks = load_tasks("golden_set/agent_tasks.json")
    llm = build_llm()
    retriever = Retriever()
    lab_client = LabClient(SshConnection())
    # No store, no approver -> submit_job/cancel_job are deny-by-default and
    # run_experiment is unregistered: the eval is read-only.
    registry = build_default_tools(retriever, lab_client)
    try:
        report = await run_eval(tasks, llm=llm, registry=registry, max_iterations=max_iterations)
    finally:
        await retriever.close()
        await lab_client.close()

    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, default=str))
    for o in report.outcomes:
        print(
            f"\n[{o.task_id}] finished={o.finished} steps={o.steps} "
            f"tools={o.called_tools} tokens={o.input_tokens + o.output_tokens}"
        )
        print(f"  report: {o.final_report[:160].strip()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(main(args.max_iterations))
