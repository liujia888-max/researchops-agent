from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from researchops.db.store import ExperimentStore
from researchops.server.main import app


def test_health() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_experiments_returns_persisted_records(tmp_path, monkeypatch) -> None:
    """The experiments endpoint reads the same store the agent writes to."""
    db = tmp_path / "e.db"
    url = f"sqlite+aiosqlite:///{db.as_posix()}"
    monkeypatch.setenv("DB_URL", url)

    async def _seed() -> None:
        store = ExperimentStore(url=url)
        await store.init()
        exp = await store.get_or_create_experiment("exp1", "demo task")
        run = await store.create_run(exp.id, "j1", "python eval.py")
        await store.add_metric(run.id, name="psnr", value=29.96, sigma=25)
        await store.finish_run(run.id, status="completed", log_tail="...")
        await store.close()

    asyncio.run(_seed())

    client = TestClient(app)
    resp = client.get("/experiments")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "exp1"
    assert data[0]["runs"][0]["job_id"] == "j1"
    assert data[0]["runs"][0]["status"] == "completed"
    assert data[0]["runs"][0]["metrics"] == [
        {"name": "psnr", "value": 29.96, "dataset": None, "sigma": 25}
    ]
