"""MCPServer adapter: exposes ``LabClient`` as the ``labops`` MCP server over stdio.

Intentionally thin — all logic lives in ``researchops.labops``. Each tool opens a fresh
SSH connection, runs its operation, and closes it, so a rebooted AutoDL host surfaces as
a clean structured error instead of a hung pooled connection.

Uses the MCP 2.x API (``MCPServer``); FastMCP was renamed in mcp 2.0.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.mcpserver import MCPServer

from researchops.config import Settings
from researchops.labops import LabClient, SshConnection

mcp = MCPServer(
    "labops",
    instructions=(
        "Remote GPU-lab orchestration over SSH. Submit, poll, and cancel training/"
        "evaluation jobs on the AutoDL host, and read GPU state, experiments, logs and "
        "metrics. `submit_job` runs arbitrary commands on the host — call it only when "
        "authorized; the other tools are read-only."
    ),
)


@asynccontextmanager
async def _client() -> AsyncIterator[LabClient]:
    conn = SshConnection(Settings())
    try:
        await conn.connect()
        yield LabClient(conn)
    finally:
        await conn.close()


@mcp.tool()
async def gpu_info() -> list[dict[str, Any]]:
    """Return each GPU's name, memory (MB), utilization (%) and temperature (°C)."""
    async with _client() as client:
        return [g.model_dump() for g in await client.gpu_info()]


@mcp.tool()
async def list_experiments() -> list[dict[str, Any]]:
    """List top-level directories and files in the remote working directory."""
    async with _client() as client:
        return [e.model_dump() for e in await client.list_experiments()]


@mcp.tool()
async def submit_job(job_id: str, command: str) -> dict[str, Any]:
    """Launch `command` (run from the working directory) as a detached screen session.

    Idempotent: if `job_id` already has a live session, nothing is launched and
    running=False is returned. job_id must match [A-Za-z0-9_-]{1,64}.
    """
    async with _client() as client:
        return (await client.submit_job(job_id, command)).model_dump()


@mcp.tool()
async def job_status(job_id: str) -> dict[str, Any]:
    """Report whether a job's screen session is live and its log file exists."""
    async with _client() as client:
        return (await client.job_status(job_id)).model_dump()


@mcp.tool()
async def tail_log(job_id: str, lines: int = 50) -> str:
    """Return the last `lines` lines of a job's log (empty if none yet)."""
    async with _client() as client:
        return await client.tail_log(job_id, lines)


@mcp.tool()
async def cancel_job(job_id: str) -> dict[str, Any]:
    """Terminate a job's screen session (idempotent) and return the new status."""
    async with _client() as client:
        return (await client.cancel_job(job_id)).model_dump()


@mcp.tool()
async def fetch_metrics(job_id: str) -> dict[str, Any]:
    """Return a job's latest metrics from its ``.metrics.json`` file, if any."""
    async with _client() as client:
        return (await client.fetch_metrics(job_id)).model_dump()
