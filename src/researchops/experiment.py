"""Deterministic ``submit -> poll -> persist`` pipeline.

The agent decides *what* to run; this layer runs it reliably. Given an exact,
reviewed command it (1) launches a detached job, (2) polls until it finishes or
times out, (3) extracts metrics from the log, and (4) persists an
Experiment/JobRun/Metric record. There is no LLM in the hot path — the command is
fully specified, so there is nothing left to hallucinate.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from researchops.db.store import ExperimentStore
from researchops.labops import LabClient
from researchops.labops.errors import CommandFailedError


@dataclass(frozen=True)
class MetricValue:
    """One parsed numeric result (e.g. PSNR at a given sigma)."""

    name: str
    value: float
    dataset: str | None = None
    sigma: int | None = None


MetricParser = Callable[[str], list[MetricValue]]


@dataclass
class JobOutcome:
    """The collected result of one job run."""

    job_id: str
    status: str  # "completed" | "timeout"
    log_tail: str
    metrics: list[MetricValue] = field(default_factory=list)
    elapsed_s: float = 0.0


# test_rgb.py prints one dict per sigma, e.g.
#   {'Dataset': 'CBSD68', ..., 'Sigma': 25, ..., 'Denoise_PSNR': 29.96, 'Denoise_SSIM': 0.8}
_RGB_ROW = re.compile(
    r"'Sigma':\s*(?P<sigma>\d+)[^\n]*?'Denoise_PSNR':\s*(?P<psnr>\d+\.?\d*)"
    r"(?:[^\n]*?'Denoise_SSIM':\s*(?P<ssim>\d+\.?\d*))?"
)

# evaluate_restormer_cbsd68_psnr_ssim.py prints per-sigma lines
#   Restormer sigma=25: PSNR=31.7800, SSIM=0.9...
# and a final summary   Sigma=25 | PSNR=31.7800 | SSIM=0.9...
_RESTORMER_ROW = re.compile(
    r"[Ss]igma\s*=\s*(?P<sigma>\d+)[^\d\n]*?PSNR\s*=\s*(?P<psnr>\d+\.?\d*)"
    r"(?:[^\d\n]*?SSIM\s*=\s*(?P<ssim>\d+\.?\d*))?"
)


def parse_metrics(log: str) -> list[MetricValue]:
    """Extract (sigma, PSNR, SSIM) triples from a denoising eval log.

    Handles both eval scripts' output formats; duplicates are collapsed by
    ``(name, sigma)``, keeping the first occurrence.
    """
    metrics: list[MetricValue] = []
    seen: set[tuple[str, int]] = set()
    for sigma_s, psnr_s, ssim_s in _RGB_ROW.findall(log) + _RESTORMER_ROW.findall(log):
        sigma = int(sigma_s)
        for name, value_s in (("psnr", psnr_s), ("ssim", ssim_s)):
            if not value_s or (name, sigma) in seen:
                continue
            seen.add((name, sigma))
            metrics.append(
                MetricValue(name=name, value=float(value_s), dataset="CBSD68", sigma=sigma)
            )
    return metrics


async def run_and_collect(
    client: LabClient,
    store: ExperimentStore,
    *,
    experiment_name: str,
    task: str,
    job_id: str,
    command: str,
    parser: MetricParser | None = parse_metrics,
    poll_interval: float = 5.0,
    timeout: float = 1800.0,  # noqa: ASYNC109  # wall-clock job timeout, not an I/O op
) -> JobOutcome:
    """Run one job to completion and persist an experiment/run/metric record.

    ``command`` is executed verbatim on the remote host as a detached screen
    session (so it must already be reviewed — this function deliberately runs it,
    with no approval gate of its own). Returns the outcome and metrics.
    """
    experiment = await store.get_or_create_experiment(experiment_name, task)
    run = await store.create_run(experiment.id, job_id, command)

    handle = await client.submit_job(job_id, command)
    if not handle.running:
        raise CommandFailedError(
            f"submit_job: job_id {job_id!r} is already in use by a live screen session"
        )

    start = time.monotonic()
    timed_out = False
    while True:
        status = await client.job_status(job_id)
        if not status.running:
            break
        if time.monotonic() - start > timeout:
            timed_out = True
            break
        await asyncio.sleep(poll_interval)

    final_status = "timeout" if timed_out else "completed"
    log_tail = await client.tail_log(job_id, lines=400)
    metrics = parser(log_tail) if parser is not None else []
    await store.finish_run(run.id, status=final_status, log_tail=log_tail)
    for metric in metrics:
        await store.add_metric(
            run.id,
            name=metric.name,
            value=metric.value,
            dataset=metric.dataset,
            sigma=metric.sigma,
        )
    return JobOutcome(
        job_id=job_id,
        status=final_status,
        log_tail=log_tail,
        metrics=metrics,
        elapsed_s=time.monotonic() - start,
    )
