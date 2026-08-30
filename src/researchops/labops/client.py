"""The labops operations layer: seven remote-lab capabilities over a CommandRunner.

Everything here is pure async and MCP-free. Parsers are plain functions (unit-testable
against captured ``nvidia-smi`` / ``screen -ls`` / ``find`` output); ``LabClient`` binds
them to a working directory and a transport. The MCP server is a thin adapter on top.

Security model
--------------
- ``job_id`` is the only user-controlled value that reaches a path or a screen session
  name. It is validated against ``^[A-Za-z0-9_-]{1,64}$`` before any use, so ``../``
  and shell metacharacters are impossible.
- ``submit_job`` *is* arbitrary remote execution by design (that is what a training job
  is). It is meant to sit behind human-in-the-loop at the *agent* layer, not here — this
  layer only narrows the blast radius: every job runs ``cd <workdir>`` first, stdout/stderr
  are captured to a log, and the whole thing is a detached ``screen`` session the agent
  can poll and cancel.
- Read-only tools (``gpu_info``, ``list_experiments``, ``job_status``, ``tail_log``,
  ``fetch_metrics``) never execute user strings.
"""

from __future__ import annotations

import csv
import io
import json
import re
import shlex
from dataclasses import dataclass

from researchops.config import Settings
from researchops.labops.errors import CommandFailedError, InvalidJobIdError
from researchops.labops.schemas import Experiment, Gpu, JobHandle, JobStatus, Metrics
from researchops.labops.ssh import CommandResult, CommandRunner

_JOB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# nvidia-smi CSV columns (order matters; see _query_gpu_csv in gpu_info).
_NVIDIA_CSV_FIELDS = (
    "index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu"
)


def validate_job_id(job_id: str) -> str:
    """Return ``job_id`` unchanged, or raise ``InvalidJobIdError``."""
    if not _JOB_ID_RE.match(job_id):
        raise InvalidJobIdError(
            f"invalid job_id {job_id!r}: must match ^[A-Za-z0-9_-]{{1,64}}$ "
            "(no slashes, spaces, or shell metacharacters)."
        )
    return job_id


# --------------------------------------------------------------------------- #
# Pure parsers (no I/O) — the unit-testable heart of the layer.
# --------------------------------------------------------------------------- #
def parse_nvidia_smi(csv_text: str) -> list[Gpu]:
    """Parse ``nvidia-smi --query-gpu=... --format=csv,noheader`` output."""
    rows = list(csv.reader(io.StringIO(csv_text)))
    gpus: list[Gpu] = []
    for row in rows:
        if not row or not row[0].strip():
            continue
        gpus.append(
            Gpu(
                index=int(row[0]),
                name=row[1].strip(),
                memory_total_mb=int(row[2]),
                memory_used_mb=int(row[3]),
                memory_free_mb=int(row[4]),
                utilization_pct=int(row[5]),
                temperature_c=int(row[6]),
            )
        )
    return gpus


# A live session line may be ``\t12345.job_name\t(Detached)`` (older screen) or,
# on newer screen/AutoDL hosts, ``\t12345.job_name\t(MM/DD/YY HH:MM:SS)\t(Detached)``.
# ``.*`` swallows the optional creation-time field; ``(Detached|Attached)`` is the
# state, which must be the last token so ``(Dead ??)`` is still excluded.
_SESSION_RE = re.compile(r"^\s*(\d+)\.(\S+)\s+.*\((Detached|Attached)\)\s*$")


def parse_screen_sessions(screen_ls: str) -> set[str]:
    """Return the set of *live* session names from ``screen -ls`` output.

    A live session line looks like ``\t12345.job_name\t(Detached)`` (older hosts) or
    ``\t12345.job_name\t(08/30/26 21:44:41)\t(Detached)`` (newer hosts). ``(Dead ??)``
    sessions are excluded — they have no process behind them and must not read as
    "running".
    """
    live: set[str] = set()
    for line in screen_ls.splitlines():
        match = _SESSION_RE.match(line)
        if match:
            live.add(match.group(2))
    return live


def parse_find(output: str, workdir: str) -> list[Experiment]:
    """Parse ``find <workdir> -maxdepth 1 -printf '%y\t%f\t%T@\\n'`` output."""
    experiments: list[Experiment] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        kind, name, mtime = line.split("\t")
        experiments.append(
            Experiment(
                name=name,
                path=f"{workdir.rstrip('/')}/{name}",
                kind="dir" if kind == "d" else "file",
                mtime_epoch=float(mtime),
            )
        )
    return experiments


@dataclass(frozen=True)
class _Paths:
    """Derived paths for one job id (log + metrics)."""

    log: str
    metrics: str


# --------------------------------------------------------------------------- #
# LabClient
# --------------------------------------------------------------------------- #
class LabClient:
    """High-level operations on the remote lab, bound to a working directory."""

    def __init__(self, runner: CommandRunner, workdir: str | None = None) -> None:
        self._runner = runner
        self._workdir = (workdir or Settings().labops_workdir).rstrip("/")

    # -- internal helpers ---------------------------------------------------- #
    def _paths(self, job_id: str) -> _Paths:
        base = f"{self._workdir}/.labops/logs/{job_id}"
        return _Paths(log=f"{base}.log", metrics=f"{base}.metrics.json")

    async def _run(self, command: str, *, timeout: float | None = None) -> CommandResult:  # noqa: ASYNC109
        result = await self._runner.run(command, timeout=timeout)
        if not result.ok:
            raise CommandFailedError(
                f"command failed (exit {result.exit_status}): {result.stderr[:300] or result.stdout[:300]}"
            )
        return result

    async def close(self) -> None:
        """Close the underlying transport (idempotent)."""
        await self._runner.close()

    # -- tools --------------------------------------------------------------- #
    async def gpu_info(self) -> list[Gpu]:
        """Current GPU inventory (name, memory, util, temp) via ``nvidia-smi``."""
        result = await self._run(
            f"nvidia-smi --query-gpu={_NVIDIA_CSV_FIELDS} --format=csv,noheader,nounits"
        )
        return parse_nvidia_smi(result.stdout)

    async def list_experiments(self) -> list[Experiment]:
        """Top-level entries (dirs/files) under the working directory."""
        result = await self._run(
            f"find {shlex.quote(self._workdir)} -maxdepth 1 -mindepth 1 "
            "-printf '%y\\t%f\\t%T@\\n'"
        )
        return parse_find(result.stdout, self._workdir)

    async def submit_job(self, job_id: str, command: str) -> JobHandle:
        """Launch ``command`` as a detached ``screen`` session, idempotently.

        Returns ``running=False`` if the id was already taken (the existing session is
        left untouched), so a retry of the same logical job never double-launches.
        """
        validate_job_id(job_id)
        if not command.strip():
            raise CommandFailedError("submit_job: command is empty")

        live = parse_screen_sessions((await self._runner.run("screen -ls")).stdout)
        if job_id in live:
            return JobHandle(job_id=job_id, running=False)

        paths = self._paths(job_id)
        await self._run(f"mkdir -p {shlex.quote(f'{self._workdir}/.labops/logs')}")
        wrapped = shlex.quote(f"cd {self._workdir} && {command} >> {paths.log} 2>&1")
        await self._run(f"screen -dmS {shlex.quote(job_id)} bash -c {wrapped}")
        return JobHandle(job_id=job_id, running=True)

    async def job_status(self, job_id: str) -> JobStatus:
        """Whether a session is live and whether its log file exists."""
        validate_job_id(job_id)
        paths = self._paths(job_id)
        live = parse_screen_sessions((await self._runner.run("screen -ls")).stdout)
        log_exists = (await self._runner.run(f"test -f {shlex.quote(paths.log)}")).ok
        return JobStatus(
            job_id=job_id,
            running=job_id in live,
            log_path=paths.log,
            log_exists=log_exists,
        )

    async def tail_log(self, job_id: str, lines: int = 50) -> str:
        """Last ``lines`` lines of the job log (empty string if none yet)."""
        validate_job_id(job_id)
        paths = self._paths(job_id)
        result = await self._run(f"tail -n {int(lines)} {shlex.quote(paths.log)}")
        return result.stdout

    async def cancel_job(self, job_id: str) -> JobStatus:
        """Terminate the detached session (idempotent) and report the new state."""
        validate_job_id(job_id)
        # -X quit on a missing session is a no-op with non-zero exit; ignore it.
        await self._runner.run(f"screen -S {shlex.quote(job_id)} -X quit")
        return await self.job_status(job_id)

    async def fetch_metrics(self, job_id: str) -> Metrics:
        """The job's latest metrics from ``<job_id>.metrics.json``, if any."""
        validate_job_id(job_id)
        paths = self._paths(job_id)
        result = await self._runner.run(f"test -f {shlex.quote(paths.metrics)} && cat {shlex.quote(paths.metrics)}")
        if not result.ok or not result.stdout.strip():
            return Metrics(job_id=job_id, metrics={}, source="none")
        try:
            raw = json.loads(result.stdout)
            metrics = {k: float(v) for k, v in raw.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            # A malformed/partial metrics file must not kill the tool; report empty.
            return Metrics(job_id=job_id, metrics={}, source="none")
        return Metrics(job_id=job_id, metrics=metrics, source="metrics.json")
