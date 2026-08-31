"""MCP client adapter: drive the `labops` MCP server as a stdio subprocess.

The server runs as a child process (``python -m researchops.mcp``); this client
enumerates its tools and calls them, so the agent consumes remote-lab capabilities
over standard MCP instead of a direct ``LabClient`` import. It also implements the
three methods of the ``RemoteLab`` protocol so the deterministic
``submit -> poll -> parse`` pipeline (``run_and_collect``) can drive it unchanged.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from researchops.config import Settings
from researchops.labops.schemas import JobHandle, JobStatus


@dataclass(frozen=True)
class RemoteTool:
    """A tool advertised by the labops MCP server, in a transport-agnostic shape."""

    name: str
    description: str
    input_schema: dict[str, Any]


def _content_text(result: Any) -> str:
    """Join the text blocks of a ``CallToolResult`` (ignores images/audio)."""
    parts: list[str] = []
    for item in result.content or []:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


class LabopsMCPClient:
    """An MCP stdio client for the labops server with an explicit lifecycle.

    ``start``/``close`` bound the connection so the host keeps the subprocess alive
    for a whole agent run and tears it down afterwards. ``list_tools`` snapshots the
    server's tool surface; ``call_tool`` invokes one and returns its text.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or Settings()
        self._ctx: Any = None
        self._session: ClientSession | None = None

    async def start(self) -> None:
        """Launch the server subprocess and open the session."""
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "researchops.mcp"],
            env=os.environ.copy(),
        )
        self._ctx = stdio_client(params)
        read, write = await self._ctx.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()

    async def close(self) -> None:
        """Tear down the session and terminate the subprocess (idempotent)."""
        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._ctx is not None:
            await self._ctx.__aexit__(None, None, None)
            self._ctx = None

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("LabopsMCPClient is not started")
        return self._session

    async def list_tools(self) -> list[RemoteTool]:
        result = await self._require_session().list_tools()
        return [
            RemoteTool(
                name=t.name,
                description=t.description or "",
                input_schema=t.input_schema or {"type": "object", "properties": {}},
            )
            for t in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        result = await self._require_session().call_tool(name, arguments or {})
        text = _content_text(result)
        if result.is_error:
            raise RuntimeError(text or f"labops MCP tool {name!r} failed")
        return text

    # -- RemoteLab protocol ------------------------------------------------ #
    async def submit_job(self, job_id: str, command: str) -> JobHandle:
        text = await self.call_tool("submit_job", {"job_id": job_id, "command": command})
        return JobHandle.model_validate(json.loads(text))

    async def job_status(self, job_id: str) -> JobStatus:
        text = await self.call_tool("job_status", {"job_id": job_id})
        return JobStatus.model_validate(json.loads(text))

    async def tail_log(self, job_id: str, lines: int = 50) -> str:
        return await self.call_tool("tail_log", {"job_id": job_id, "lines": lines})
