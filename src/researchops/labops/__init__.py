"""labops — remote GPU-lab orchestration over SSH (transport + operations layer).

MCP-free by design: ``LabClient`` runs over a ``CommandRunner`` protocol so it is
unit-testable without a live host; ``researchops.mcp`` is a thin adapter on top.
"""

from __future__ import annotations

from typing import Protocol

from researchops.labops.client import LabClient
from researchops.labops.schemas import Experiment, Gpu, JobHandle, JobStatus, Metrics
from researchops.labops.ssh import CommandResult, CommandRunner, SshConnection


class RemoteLab(Protocol):
    """The three remote-lab operations the deterministic pipeline needs.

    Both ``LabClient`` (direct SSH) and ``LabopsMCPClient`` (MCP stdio) satisfy it,
    so ``run_and_collect`` drives either without knowing which transport is underneath.
    """

    async def submit_job(self, job_id: str, command: str) -> JobHandle: ...
    async def job_status(self, job_id: str) -> JobStatus: ...
    async def tail_log(self, job_id: str, lines: int = 50) -> str: ...


__all__ = [
    "LabClient",
    "SshConnection",
    "CommandRunner",
    "CommandResult",
    "RemoteLab",
    "Gpu",
    "Experiment",
    "JobHandle",
    "JobStatus",
    "Metrics",
]
