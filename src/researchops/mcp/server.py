"""MCPServer adapter: exposes ``LabClient`` as the ``labops`` MCP server over stdio.

Intentionally thin — all logic lives in ``researchops.labops``. Each tool opens a fresh
SSH connection, runs its operation, and closes it, so a rebooted AutoDL host surfaces as
a clean structured error instead of a hung pooled connection.

Labops failures (host down, command failed) are converted into a JSON-serializable
``{"error", "advice"}`` result instead of letting the MCP layer wrap them in a raw
traceback — the agent then gets an actionable message, not a stack dump.

Uses the MCP 2.x API (``MCPServer``); FastMCP was renamed in mcp 2.0.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.mcpserver import MCPServer

from researchops.config import Settings
from researchops.labops import LabClient, SshConnection
from researchops.labops.errors import HostUnreachableError, LabopsError

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


def _structured_error(exc: LabopsError) -> dict[str, str]:
    """Turn a labops failure into a readable result (the agent reads this JSON text)."""
    if isinstance(exc, HostUnreachableError):
        advice = (
            "GPU host unreachable. Check: (1) the AutoDL instance is powered on in the "
            "console; (2) LABOPS_HOST/LABOPS_PORT in .env match the instance's current "
            "SSH address — the port changes whenever the instance is re-provisioned."
        )
    else:
        advice = "The remote operation failed; see the error message."
    return {"error": str(exc), "advice": advice}


async def _run(operation: Callable[[LabClient], Awaitable[Any]]) -> Any:
    """Run one remote-lab operation, mapping labops failures to structured results."""
    try:
        async with _client() as client:
            return await operation(client)
    except LabopsError as exc:
        return _structured_error(exc)


@mcp.tool()
async def gpu_info() -> Any:
    """Return each GPU's name, memory (MB), utilization (%) and temperature (°C)."""

    async def op(client: LabClient) -> list[dict[str, Any]]:
        return [g.model_dump() for g in await client.gpu_info()]

    return await _run(op)


@mcp.tool()
async def list_experiments() -> Any:
    """List top-level directories and files in the remote working directory."""

    async def op(client: LabClient) -> list[dict[str, Any]]:
        return [e.model_dump() for e in await client.list_experiments()]

    return await _run(op)


@mcp.tool()
async def submit_job(job_id: str, command: str) -> Any:
    """Launch `command` (run from the working directory) as a detached screen session.

    Idempotent: if `job_id` already has a live session, nothing is launched and
    running=False is returned. job_id must match [A-Za-z0-9_-]{1,64}.
    """

    async def op(client: LabClient) -> dict[str, Any]:
        return (await client.submit_job(job_id, command)).model_dump()

    return await _run(op)


@mcp.tool()
async def job_status(job_id: str) -> Any:
    """Report whether a job's screen session is live and its log file exists."""

    async def op(client: LabClient) -> dict[str, Any]:
        return (await client.job_status(job_id)).model_dump()

    return await _run(op)


@mcp.tool()
async def tail_log(job_id: str, lines: int = 50) -> Any:
    """Return the last `lines` lines of a job's log (empty if none yet)."""

    async def op(client: LabClient) -> str:
        return await client.tail_log(job_id, lines)

    return await _run(op)


@mcp.tool()
async def cancel_job(job_id: str) -> Any:
    """Terminate a job's screen session (idempotent) and return the new status."""

    async def op(client: LabClient) -> dict[str, Any]:
        return (await client.cancel_job(job_id)).model_dump()

    return await _run(op)


@mcp.tool()
async def fetch_metrics(job_id: str) -> Any:
    """Return a job's latest metrics from its ``.metrics.json`` file, if any."""

    async def op(client: LabClient) -> dict[str, Any]:
        return (await client.fetch_metrics(job_id)).model_dump()

    return await _run(op)
