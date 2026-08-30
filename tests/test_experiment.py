"""Tests for the submit->poll->persist pipeline and metric parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from researchops.db.store import ExperimentStore
from researchops.experiment import parse_metrics, run_and_collect
from researchops.labops import LabClient
from researchops.labops.errors import CommandFailedError
from researchops.labops.ssh import CommandResult


def _sqlite_url(tmp_path: Path, name: str = "e.db") -> str:
    return f"sqlite+aiosqlite:///{(tmp_path / name).as_posix()}"


# --------------------------------------------------------------------------- #
# Metric parsing
# --------------------------------------------------------------------------- #
def test_parse_metrics_rgb_row() -> None:
    log = (
        "{'Dataset': 'CBSD68', 'Images': 68, 'Sigma': 25, 'Noisy_PSNR': 20.53, "
        "'Denoise_PSNR': 29.96, 'PSNR_Gain': 9.43, 'Noisy_SSIM': 0.75, "
        "'Denoise_SSIM': 0.86, 'SSIM_Gain': 0.11}\n"
    )
    metrics = parse_metrics(log)
    assert [(m.name, m.value, m.sigma) for m in metrics] == [
        ("psnr", 29.96, 25),
        ("ssim", 0.86, 25),
    ]


def test_parse_metrics_restormer_row() -> None:
    log = "Restormer sigma=25: PSNR=31.7800, SSIM=0.9301\n"
    metrics = parse_metrics(log)
    assert [(m.name, m.value, m.sigma) for m in metrics] == [
        ("psnr", 31.78, 25),
        ("ssim", 0.9301, 25),
    ]


def test_parse_metrics_dedupes_summary_line() -> None:
    log = (
        "Restormer sigma=25: PSNR=31.7800, SSIM=0.9301\n"
        "Sigma=25 | PSNR=31.7800 | SSIM=0.9301\n"
    )
    metrics = parse_metrics(log)
    assert [(m.name, m.value, m.sigma) for m in metrics] == [
        ("psnr", 31.78, 25),
        ("ssim", 0.9301, 25),
    ]


def test_parse_metrics_no_match() -> None:
    assert parse_metrics("some unrelated log\n") == []


# --------------------------------------------------------------------------- #
# Pipeline (submit -> poll -> persist)
# --------------------------------------------------------------------------- #
class _ScriptedRunner:
    """Pops one canned ``screen -ls`` response per call; returns ``tail`` for logs."""

    def __init__(self, screen_ls: list[str], tail: str = "") -> None:
        self._screen = list(screen_ls)
        self._tail = tail
        self.calls: list[str] = []

    async def run(self, command: str, *, timeout: float | None = None) -> CommandResult:  # noqa: ASYNC109
        self.calls.append(command)
        if "screen -ls" in command:
            out = self._screen.pop(0) if self._screen else "No Sockets found\n"
            return CommandResult(0, out, "")
        if "tail -n" in command:
            return CommandResult(0, self._tail, "")
        return CommandResult(0, "", "")

    async def close(self) -> None:
        pass


async def test_run_and_collect_persists_metrics(tmp_path: Path) -> None:
    url = _sqlite_url(tmp_path)
    store = ExperimentStore(url=url)
    await store.init()

    runner = _ScriptedRunner(
        screen_ls=[
            "No Sockets found\n",  # submit_job idempotency check
            "\t1234.repro_x\t(Detached)\n",  # poll 1: still running
            "No Sockets found\n",  # poll 2: finished
        ],
        tail="{'Sigma': 25, 'Denoise_PSNR': 29.96, 'Denoise_SSIM': 0.86}\n",
    )
    client = LabClient(runner, workdir="/root/autodl-tmp")

    try:
        outcome = await run_and_collect(
            client,
            store,
            experiment_name="repro_x",
            task="reproduce model_v3_rgb on CBSD68",
            job_id="repro_x",
            command="python test_rgb.py",
            poll_interval=0.0,
            timeout=60.0,
        )
    finally:
        await store.close()

    assert outcome.status == "completed"
    assert [(m.name, m.value, m.sigma) for m in outcome.metrics] == [
        ("psnr", 29.96, 25),
        ("ssim", 0.86, 25),
    ]

    # Round-trip through a fresh store against the same file.
    store2 = ExperimentStore(url=url)
    await store2.init()
    try:
        exp = await store2.get_experiment("repro_x")
        assert exp is not None
        runs = await store2.list_runs(exp.id)
        assert len(runs) == 1
        assert runs[0].status == "completed"
        assert runs[0].command == "python test_rgb.py"
        metrics = await store2.list_metrics(runs[0].id)
        assert [(m.name, m.value, m.sigma) for m in metrics] == [
            ("psnr", 29.96, 25),
            ("ssim", 0.86, 25),
        ]
    finally:
        await store2.close()


async def test_run_and_collect_rejects_live_job_id(tmp_path: Path) -> None:
    url = _sqlite_url(tmp_path)
    store = ExperimentStore(url=url)
    await store.init()

    runner = _ScriptedRunner(screen_ls=["\t1234.repro_x\t(Detached)\n"])
    client = LabClient(runner, workdir="/root/autodl-tmp")

    try:
        with pytest.raises(CommandFailedError):
            await run_and_collect(
                client,
                store,
                experiment_name="repro_x",
                task="task",
                job_id="repro_x",
                command="python x.py",
                poll_interval=0.0,
            )
    finally:
        await store.close()
