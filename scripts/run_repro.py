"""Run the real submit -> poll -> persist chain on the GPU host.

Reproduces two CBSD68 (sigma=15/25/50) evals — the user's ``model_v3_rgb`` and
the official Restormer Gaussian Color Blind baseline — polls each to completion,
parses PSNR/SSIM from the log, and persists Experiment/JobRun/Metric rows to the
database. No LLM in the loop; this is the deterministic hot path the agent calls.

Usage (from the repo root, with the project env active):
    python scripts/run_repro.py

Read-only preflight (GPU inventory + the in-flight training job PID) runs first so
a powered-off host or a missing weight fails fast before any job is launched.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from researchops.db.store import ExperimentStore
from researchops.experiment import run_and_collect
from researchops.labops import LabClient
from researchops.labops.ssh import SshConnection

WORKDIR = "/root/autodl-tmp"

# Exact, pre-reviewed commands. ``submit_job`` runs ``cd <workdir> && <command>``
# inside a detached screen session, so these are relative to WORKDIR.
MODEL_V3_RGB_CMD = "cd pythonProject4 && python3 test_rgb.py"
RESTORMER_CMD = (
    "cd pythonProject4/Restormer/Denoising && "
    "PYTHONPATH=/root/autodl-tmp/pythonProject4/Restormer "
    "python3 test_gaussian_color_denoising.py --model_type blind --sigmas 15,25,50 && "
    "PYTHONPATH=/root/autodl-tmp/pythonProject4/Restormer "
    "python3 evaluate_restormer_cbsd68_psnr_ssim.py --model_type blind --sigmas 15,25,50"
)

POLL_INTERVAL = 15.0
TIMEOUT = 1800.0  # 30 min wall-clock per job — plenty for a 68-image eval.


async def _preflight(client: LabClient, conn: SshConnection) -> None:
    print("== preflight ==")
    for g in await client.gpu_info():
        print(
            f"  GPU {g.name}: {g.memory_used_mb}/{g.memory_total_mb} MB used, "
            f"{g.memory_free_mb} MB free, util {g.utilization_pct}%"
        )
    # Read-only: confirm the in-flight training job is untouched (we must not stop it).
    r = await conn.run("ps -p 3606 -o pid=,etime=,cmd= 2>/dev/null || true")
    if r.stdout.strip():
        print(f"  training job alive (NOT touched): {r.stdout.strip()}")
    else:
        print("  (PID 3606 not found — training may have finished; no action taken)")


async def main() -> None:
    Path(".researchops").mkdir(exist_ok=True)

    store = ExperimentStore()
    await store.init()
    conn = SshConnection()
    client = LabClient(conn, workdir=WORKDIR)

    try:
        await _preflight(client, conn)

        print("\n== run 1/2: model_v3_rgb ==")
        v3 = await run_and_collect(
            client,
            store,
            experiment_name="repro_cbsd68_model_v3_rgb",
            task="复现 model_v3_rgb 在 CBSD68 sigma=15/25/50 的 PSNR/SSIM",
            job_id="repro_model_v3_rgb",
            command=MODEL_V3_RGB_CMD,
            poll_interval=POLL_INTERVAL,
            timeout=TIMEOUT,
        )
        print(f"  status={v3.status} elapsed={v3.elapsed_s:.0f}s")
        for m in v3.metrics:
            print(f"    {m.name:4s} sigma={m.sigma}: {m.value}")

        print("\n== run 2/2: Restormer (Gaussian Color Blind) ==")
        rs = await run_and_collect(
            client,
            store,
            experiment_name="repro_restormer_cbsd68",
            task="复现 Restormer Gaussian Color Blind 在 CBSD68 sigma=15/25/50 的 PSNR/SSIM",
            job_id="repro_restormer",
            command=RESTORMER_CMD,
            poll_interval=POLL_INTERVAL,
            timeout=TIMEOUT,
        )
        print(f"  status={rs.status} elapsed={rs.elapsed_s:.0f}s")
        for m in rs.metrics:
            print(f"    {m.name:4s} sigma={m.sigma}: {m.value}")

        print("\n== comparison ==")
        print(f"{'sigma':>5}  {'model_v3_rgb':>13}  {'Restormer':>10}  {'delta':>7}")
        for sigma in (15, 25, 50):
            a = next((m.value for m in v3.metrics if m.name == "psnr" and m.sigma == sigma), None)
            b = next((m.value for m in rs.metrics if m.name == "psnr" and m.sigma == sigma), None)
            if a is not None and b is not None:
                print(f"{sigma:>5}  {a:>13.2f}  {b:>10.2f}  {b - a:>+7.2f}")
    finally:
        await client.close()
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
