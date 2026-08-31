"""Async persistence layer over SQLAlchemy 2.0 (SQLite default, Postgres-ready)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    async_sessionmaker,
    create_async_engine,
)

from researchops.config import Settings
from researchops.db.models import Base, Experiment, JobRun, Metric


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ExperimentStore:
    """Create/read experiment records, runs, and metrics in a SQLAlchemy database."""

    def __init__(self, url: str | None = None) -> None:
        self._engine: AsyncEngine = create_async_engine(url or Settings().db_url)
        self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)

    async def init(self) -> None:
        """Create tables if they don't exist (idempotent)."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self._engine.dispose()

    async def get_or_create_experiment(self, name: str, task: str) -> Experiment:
        """Return the experiment named ``name``, creating it if absent."""
        async with self._sessions() as session:
            existing = cast(
                Experiment | None,
                await session.scalar(select(Experiment).where(Experiment.name == name)),
            )
            if existing is not None:
                return existing
            experiment = Experiment(name=name, task=task)
            session.add(experiment)
            await session.commit()
            await session.refresh(experiment)
            return experiment

    async def create_run(self, experiment_id: int, job_id: str, command: str) -> JobRun:
        """Record a new job run as 'submitted' and return it (with an assigned id)."""
        async with self._sessions() as session:
            run = JobRun(
                experiment_id=experiment_id,
                job_id=job_id,
                command=command,
                status="submitted",
                started_at=_utcnow(),
            )
            session.add(run)
            await session.commit()
            await session.refresh(run)
            return run

    async def finish_run(self, run_id: int, *, status: str, log_tail: str) -> None:
        """Mark a run finished with its final status and captured log tail."""
        async with self._sessions() as session:
            run = await session.get(JobRun, run_id)
            if run is None:
                return
            run.status = status
            run.log_tail = log_tail
            run.finished_at = _utcnow()
            await session.commit()

    async def add_metric(
        self,
        run_id: int,
        *,
        name: str,
        value: float,
        dataset: str | None = None,
        sigma: int | None = None,
    ) -> None:
        """Append one numeric metric to a run."""
        async with self._sessions() as session:
            session.add(Metric(run_id=run_id, name=name, value=value, dataset=dataset, sigma=sigma))
            await session.commit()

    async def get_experiment(self, name: str) -> Experiment | None:
        async with self._sessions() as session:
            return cast(
                Experiment | None,
                await session.scalar(select(Experiment).where(Experiment.name == name)),
            )

    async def list_experiments(self) -> list[Experiment]:
        """Return every experiment, oldest first."""
        async with self._sessions() as session:
            rows = await session.scalars(select(Experiment).order_by(Experiment.id))
            return list(rows)

    async def list_runs(self, experiment_id: int) -> list[JobRun]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(JobRun).where(JobRun.experiment_id == experiment_id).order_by(JobRun.id)
            )
            return list(rows)

    async def list_metrics(self, run_id: int) -> list[Metric]:
        async with self._sessions() as session:
            rows = await session.scalars(select(Metric).where(Metric.run_id == run_id).order_by(Metric.id))
            return list(rows)
