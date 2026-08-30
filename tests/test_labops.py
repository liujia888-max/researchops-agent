"""Unit tests for the labops layer, against a fake in-process CommandRunner.

No live GPU host is required: the fake records every command and returns canned
output, so we assert on *command construction* (security-relevant: quoting, cd
sandbox, log redirect) and on the pure parsers.
"""

from __future__ import annotations

import pytest

from researchops.labops.client import (
    LabClient,
    parse_find,
    parse_nvidia_smi,
    parse_screen_sessions,
    validate_job_id,
)
from researchops.labops.errors import InvalidJobIdError
from researchops.labops.ssh import CommandResult


class FakeRunner:
    """Minimal CommandRunner: records commands, matches responses by substring."""

    def __init__(self, default: CommandResult | None = None) -> None:
        self.calls: list[str] = []
        self._rules: list[tuple[str, CommandResult]] = []
        self._default = default or CommandResult(0, "", "")

    def respond(self, substr: str, *, stdout: str = "", stderr: str = "", exit_status: int = 0) -> FakeRunner:
        self._rules.append((substr, CommandResult(exit_status, stdout, stderr)))
        return self

    async def run(self, command: str, *, timeout: float | None = None) -> CommandResult:  # noqa: ASYNC109
        self.calls.append(command)
        for substr, result in self._rules:
            if substr in command:
                return result
        return self._default

    async def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# Pure parsers
# --------------------------------------------------------------------------- #
def test_parse_nvidia_smi_single_gpu() -> None:
    gpus = parse_nvidia_smi("0, NVIDIA GeForce RTX 5090, 32768, 4200, 28568, 95, 71\n")
    assert len(gpus) == 1
    g = gpus[0]
    assert g.index == 0
    assert g.name == "NVIDIA GeForce RTX 5090"
    assert (g.memory_total_mb, g.memory_used_mb, g.memory_free_mb) == (32768, 4200, 28568)
    assert g.utilization_pct == 95
    assert g.temperature_c == 71


def test_parse_nvidia_smi_ignores_blank_lines() -> None:
    assert parse_nvidia_smi("\n\n") == []


def test_parse_screen_sessions_detached_only_excludes_dead() -> None:
    out = (
        "There is a screen on:\n"
        "\t1234.train_v6\t(Detached)\n"
        "\t5678.dead_job\t(Dead ???)\n"
        "1 Socket in /root/.screen.\n"
    )
    assert parse_screen_sessions(out) == {"train_v6"}


def test_parse_screen_sessions_no_sockets() -> None:
    assert parse_screen_sessions("No Sockets found in /root/.screen.\n") == set()


def test_parse_find_dir_and_file() -> None:
    out = "d\tpythonProject4\t1720000000.0\nf\tnotes.txt\t1720000100.5\n"
    exps = parse_find(out, "/root/autodl-tmp")
    assert [e.name for e in exps] == ["pythonProject4", "notes.txt"]
    assert [e.kind for e in exps] == ["dir", "file"]
    assert exps[0].path == "/root/autodl-tmp/pythonProject4"


# --------------------------------------------------------------------------- #
# job_id validation (path-traversal / injection guard)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["../evil", "a/b", "foo; rm -rf", "has space", "", "x" * 65])
def test_validate_job_id_rejects(bad: str) -> None:
    with pytest.raises(InvalidJobIdError):
        validate_job_id(bad)


@pytest.mark.parametrize("ok", ["train_v6", "eval-2026-08", "a", "A1_-b2"])
def test_validate_job_id_accepts(ok: str) -> None:
    assert validate_job_id(ok) == ok


# --------------------------------------------------------------------------- #
# LabClient operations via FakeRunner
# --------------------------------------------------------------------------- #
def test_gpu_info() -> None:
    runner = FakeRunner().respond(
        "nvidia-smi --query-gpu=",
        stdout="0, NVIDIA GeForce RTX 5090, 32768, 4200, 28568, 95, 71\n",
    )
    client = LabClient(runner, workdir="/root/autodl-tmp")

    gpus = _run(client.gpu_info())

    assert [g.index for g in gpus] == [0]
    assert gpus[0].name == "NVIDIA GeForce RTX 5090"
    assert "nvidia-smi --query-gpu=index,name" in runner.calls[0]


def test_submit_job_constructs_sandboxed_command() -> None:
    runner = FakeRunner().respond("screen -ls", stdout="No Sockets found\n")
    client = LabClient(runner, workdir="/root/autodl-tmp")

    handle = _run(client.submit_job("train_v6", "python train.py"))

    assert handle.running is True
    assert handle.job_id == "train_v6"
    assert runner.calls[0] == "screen -ls"
    assert runner.calls[1].startswith("mkdir -p ")
    launch = runner.calls[2]
    assert launch.startswith("screen -dmS train_v6 bash -c ")
    # the wrapped body is single-quoted by shlex.quote; it must cd into the sandbox
    # and append to the job log.
    assert "cd /root/autodl-tmp" in launch
    assert "python train.py" in launch
    assert ">> /root/autodl-tmp/.labops/logs/train_v6.log 2>&1" in launch


def test_submit_job_idempotent_when_already_live() -> None:
    runner = FakeRunner().respond(
        "screen -ls", stdout="\t1234.train_v6\t(Detached)\n"
    )
    client = LabClient(runner, workdir="/root/autodl-tmp")

    handle = _run(client.submit_job("train_v6", "python train.py"))

    assert handle.running is False
    # No mkdir / no second launch: only the status check ran.
    assert runner.calls == ["screen -ls"]


def test_submit_job_rejects_bad_id_before_any_io() -> None:
    runner = FakeRunner()
    client = LabClient(runner, workdir="/root/autodl-tmp")

    with pytest.raises(InvalidJobIdError):
        _run(client.submit_job("../evil", "python train.py"))

    assert runner.calls == []


def test_job_status_running_and_log_present() -> None:
    runner = (
        FakeRunner()
        .respond("screen -ls", stdout="\t1234.train_v6\t(Detached)\n")
        .respond("test -f", exit_status=0)
    )
    client = LabClient(runner, workdir="/root/autodl-tmp")

    status = _run(client.job_status("train_v6"))

    assert status.running is True
    assert status.log_exists is True
    assert status.log_path.endswith("/.labops/logs/train_v6.log")


def test_job_status_not_running_no_log() -> None:
    runner = (
        FakeRunner()
        .respond("screen -ls", stdout="No Sockets found\n")
        .respond("test -f", exit_status=1)
    )
    client = LabClient(runner, workdir="/root/autodl-tmp")

    status = _run(client.job_status("train_v6"))

    assert status.running is False
    assert status.log_exists is False


def test_tail_log_returns_stdout() -> None:
    runner = FakeRunner().respond("tail -n", stdout="epoch 3 loss=0.01\n")
    client = LabClient(runner, workdir="/root/autodl-tmp")

    assert _run(client.tail_log("train_v6", lines=10)) == "epoch 3 loss=0.01\n"
    assert runner.calls[0] == "tail -n 10 /root/autodl-tmp/.labops/logs/train_v6.log"


def test_cancel_job_sends_quit_and_reports_status() -> None:
    runner = (
        FakeRunner()
        .respond("screen -ls", stdout="No Sockets found\n")
        .respond("test -f", exit_status=0)
    )
    client = LabClient(runner, workdir="/root/autodl-tmp")

    status = _run(client.cancel_job("train_v6"))

    assert runner.calls[0].startswith("screen -S train_v6 -X quit")
    assert status.running is False


def test_fetch_metrics_parses_json() -> None:
    runner = FakeRunner().respond("cat ", stdout='{"psnr": 31.79, "ssim": 0.89}\n')
    client = LabClient(runner, workdir="/root/autodl-tmp")

    metrics = _run(client.fetch_metrics("train_v6"))

    assert metrics.metrics == {"psnr": 31.79, "ssim": 0.89}
    assert metrics.source == "metrics.json"


def test_fetch_metrics_missing_file_returns_none() -> None:
    runner = FakeRunner().respond("test -f", exit_status=1)
    client = LabClient(runner, workdir="/root/autodl-tmp")

    metrics = _run(client.fetch_metrics("train_v6"))

    assert metrics.metrics == {}
    assert metrics.source == "none"


def _run(awaitable):
    """Drive a coroutine synchronously (the package has no pytest-asyncio dep)."""
    import asyncio

    return asyncio.run(awaitable)
