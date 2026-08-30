"""Typed results returned by the labops tools (serialized over MCP as JSON)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Gpu(BaseModel):
    """One physical GPU as reported by ``nvidia-smi``."""

    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int
    memory_free_mb: int
    utilization_pct: int
    temperature_c: int


class Experiment(BaseModel):
    """One entry in the remote working directory."""

    name: str
    path: str
    kind: str = Field(description="'dir' or 'file'")
    mtime_epoch: float


class JobHandle(BaseModel):
    """Result of submitting a job."""

    job_id: str
    running: bool


class JobStatus(BaseModel):
    """Current state of a submitted job."""

    job_id: str
    running: bool  # a detached screen session exists
    log_path: str
    log_exists: bool


class Metrics(BaseModel):
    """A job's latest numeric metrics."""

    job_id: str
    metrics: dict[str, float]
    source: str = Field(description="'metrics.json' when present, else 'none'")
