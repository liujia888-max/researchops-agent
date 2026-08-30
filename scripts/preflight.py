"""Quick read-only preflight: is the GPU host up, how much VRAM is free, is the
in-flight training job still alive, and do the eval scripts/weights exist?"""

from __future__ import annotations

import asyncio

from researchops.labops import LabClient
from researchops.labops.ssh import SshConnection


async def main() -> None:
    conn = SshConnection()
    client = LabClient(conn, workdir="/root/autodl-tmp")
    try:
        for g in await client.gpu_info():
            print(
                f"[gpu] {g.name} used={g.memory_used_mb}/{g.memory_total_mb}MB "
                f"free={g.memory_free_mb}MB util={g.utilization_pct}%"
            )

        r = await conn.run("ps -p 3606 -o pid=,etime=,cmd= 2>/dev/null || true")
        print("[pid3606]", r.stdout.strip() or "(not found)")

        files = [
            "/root/autodl-tmp/pythonProject4/test_rgb.py",
            "/root/autodl-tmp/pythonProject4/rgb_stage2.pth",
            "/root/autodl-tmp/pythonProject4/Restormer/Denoising/test_gaussian_color_denoising.py",
            "/root/autodl-tmp/pythonProject4/Restormer/Denoising/pretrained_models/gaussian_color_denoising_blind.pth",
        ]
        for f in files:
            rr = await conn.run(f"test -f {f}; echo $?")
            print(f"[file] {f} -> {'OK' if rr.stdout.strip() == '0' else 'MISSING'}")

        rr = await conn.run(
            "ls /root/autodl-tmp/pythonProject4/Restormer/Denoising/evaluate*.py 2>/dev/null || true"
        )
        print("[evaluate scripts]\n" + rr.stdout.strip())
    finally:
        await client.close()


asyncio.run(main())
