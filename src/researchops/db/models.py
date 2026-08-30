"""SQLAlchemy ORM models for persisted experiment records.

Postgres-ready by construction: the models use nothing SQLite-specific, so
swapping the ``db_url`` setting to a Postgres DSN (e.g. ``postgresql+asyncpg://``)
needs no model changes.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    """Declarative base for all persisted models."""


class Experiment(Base):
    """A named research experiment — one logical task, one or more job runs."""

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    task: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class JobRun(Base):
    """One submitted job (a detached ``screen`` session on the GPU host)."""

    __tablename__ = "job_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), index=True)
    job_id: Mapped[str] = mapped_column(String(64), index=True)
    command: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), default="submitted")
    log_tail: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Metric(Base):
    """A single numeric result (PSNR, SSIM, ...) for one run, keyed by dataset/sigma."""

    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("job_runs.id"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    value: Mapped[float] = mapped_column()
    dataset: Mapped[str | None] = mapped_column(String(64), nullable=True)
    sigma: Mapped[int | None] = mapped_column(Integer, nullable=True)
