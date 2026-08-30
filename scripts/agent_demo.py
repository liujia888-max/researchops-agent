"""End-to-end agent demo: one sentence -> planner -> run_experiment x2 -> report.

Runs the *real* chain through the LangGraph executor: the agent plans, then calls the
deterministic ``run_experiment`` tool twice (model_v3_rgb and Restormer on CBSD68
sigma=15/25/50), which submits -> polls -> parses -> persists, then synthesizes a report.

The destructive ``run_experiment`` tool is deny-by-default. This demo installs an
auto-approver ONLY when ``RESEARCHOPS_DEMO_AUTO_APPROVE=1`` is set; without that flag it
stops at the approval gate instead of launching jobs.

Usage (project env active, GPU host online, Qdrant + .env ready):
    $env:RESEARCHOPS_DEMO_AUTO_APPROVE = '1'
    python scripts/agent_demo.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from researchops.agent.runner import run_agent
from researchops.agent.tools import build_default_tools
from researchops.db.store import ExperimentStore
from researchops.labops import LabClient, SshConnection
from researchops.llm import build_llm
from researchops.rag.retriever import Retriever

TASK = (
    "复现并对比两个模型在 CBSD68（彩色去噪，σ=15/25/50）的 PSNR/SSIM。已确认命令：\n"
    "1) model_v3_rgb：`cd pythonProject4 && python3 test_rgb.py`\n"
    "2) Restormer Gaussian Color Blind："
    "`cd pythonProject4/Restormer/Denoising && PYTHONPATH=/root/autodl-tmp/pythonProject4/Restormer "
    "python3 test_gaussian_color_denoising.py --model_type blind --sigmas 15,25,50 && "
    "PYTHONPATH=/root/autodl-tmp/pythonProject4/Restormer "
    "python3 evaluate_restormer_cbsd68_psnr_ssim.py --model_type blind --sigmas 15,25,50`\n"
    "请用 run_experiment 工具分别跑这两个命令（各起一个 experiment_name 和 job_id），"
    "拿到指标后对比 σ=25 的 PSNR，并写一份带结论的报告。"
)


async def main() -> None:
    auto_approve = os.environ.get("RESEARCHOPS_DEMO_AUTO_APPROVE") == "1"
    approver = (lambda name, args: True) if auto_approve else None  # noqa: E731

    llm = build_llm()
    retriever = Retriever()
    lab_client = LabClient(SshConnection())
    store = ExperimentStore()
    registry = build_default_tools(retriever, lab_client, approver=approver, store=store)

    try:
        Path(".researchops").mkdir(exist_ok=True)
        await store.init()
        state = await run_agent(TASK, llm=llm, registry=registry, max_iterations=12)
    finally:
        await retriever.close()
        await lab_client.close()
        await store.close()

    print("\n" + "=" * 72)
    print("PLAN:")
    for s in state.plan:
        print(f"  - {s}")
    print("\nTOOL CALLS:")
    for r in state.tool_results:
        print(f"  -> {r.tool}({r.arguments})")
        print(f"     {r.output[:200]}")
    print("=" * 72)
    print(state.final_report)


if __name__ == "__main__":
    asyncio.run(main())
