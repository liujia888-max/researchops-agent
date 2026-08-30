"""labops — remote GPU-lab orchestration over SSH (transport + operations layer).

MCP-free by design: ``LabClient`` runs over a ``CommandRunner`` protocol so it is
unit-testable without a live host; ``researchops.mcp`` is a thin adapter on top.
"""

from __future__ import annotations

from researchops.labops.client import LabClient
from researchops.labops.schemas import Experiment, Gpu, JobHandle, JobStatus, Metrics
from researchops.labops.ssh import CommandResult, CommandRunner, SshConnection

__all__ = [
    "LabClient",
    "SshConnection",
    "CommandRunner",
    "CommandResult",
    "Gpu",
    "Experiment",
    "JobHandle",
    "JobStatus",
    "Metrics",
]
