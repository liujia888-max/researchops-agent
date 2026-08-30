"""Async SSH transport to the GPU host, built on ``asyncssh``.

The labops layer is deliberately free of any MCP dependency. It exposes a plain
``CommandRunner`` protocol; ``SshConnection`` is the real implementation and a
``FakeRunner`` in the test suite stands in for it, so all higher-level logic is
unit-testable without a live host.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import asyncssh

from researchops.config import Settings
from researchops.labops.errors import CommandFailedError, HostUnreachableError


@dataclass(frozen=True)
class CommandResult:
    """The outcome of one remote command."""

    exit_status: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_status == 0


def _to_str(value: str | bytes) -> str:
    """Coerce asyncssh's ``str | bytes`` output to ``str``."""
    return value.decode() if isinstance(value, bytes) else value


class CommandRunner(Protocol):
    """Minimal async channel for running one command and (optionally) closing."""

    async def run(self, command: str, *, timeout: float | None = None) -> CommandResult: ...  # noqa: ASYNC109

    async def close(self) -> None: ...


class SshConnection:
    """A reusable asyncssh connection with per-command timeout.

    AutoDL re-provisions hosts on restart, so their host keys rotate. We accept
    the presented key rather than pinning it to ``known_hosts`` — the accepted
    tradeoff for a lab (see the plan's threat model). A production fleet would
    pin keys instead.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        s = settings or Settings()
        self._host = s.labops_host
        self._port = s.labops_port
        self._user = s.labops_user
        self._key_path = str(Path(s.labops_key_path).expanduser())
        self._timeout = s.labops_command_timeout
        self._conn: asyncssh.SSHClientConnection | None = None

    async def connect(self) -> None:
        if self._conn is not None:
            return
        try:
            self._conn = await asyncssh.connect(
                self._host,
                port=self._port,
                username=self._user,
                client_keys=[self._key_path],
                known_hosts=None,
                connect_timeout=self._timeout,
            )
        except (asyncssh.Error, OSError) as exc:
            raise HostUnreachableError(
                f"cannot reach {self._user}@{self._host}:{self._port} — {exc}. "
                "Is the GPU host powered on and is the key valid?"
            ) from exc

    async def run(self, command: str, *, timeout: float | None = None) -> CommandResult:  # noqa: ASYNC109
        """Run one command and return its exit status + captured stdout/stderr."""
        await self.connect()
        assert self._conn is not None
        try:
            result = await self._conn.run(command, check=False, timeout=timeout or self._timeout)
        except asyncssh.ProcessError as exc:
            raise CommandFailedError(
                f"command exited {exc.exit_status}: {_to_str(exc.stderr or '')[:300]}"
            ) from exc
        except asyncssh.TimeoutError as exc:
            raise CommandFailedError(
                f"command timed out after {timeout or self._timeout}s: {command[:120]}"
            ) from exc
        except (asyncssh.Error, OSError) as exc:
            raise CommandFailedError(f"command failed: {exc}") from exc
        return CommandResult(
            exit_status=result.exit_status or 0,
            stdout=_to_str(result.stdout or ""),
            stderr=_to_str(result.stderr or ""),
        )

    async def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            await self._conn.wait_closed()
            self._conn = None
