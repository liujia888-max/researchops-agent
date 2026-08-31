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


class _FakeEmbedder:
    async def embed_chunks(self, chunks) -> None:
        for c in chunks:
            c.dense = [0.0] * 1024
            c.sparse_indices = [0]
            c.sparse_values = [1.0]


class _FakeQdrantStore:
    def __init__(self, settings=None) -> None:
        pass

    async def upsert(self, chunks) -> None:
        return None

    async def close(self) -> None:
        return None


def test_upload_document_txt(tmp_path, monkeypatch) -> None:
    """Uploading a .txt parses, chunks, and ingests (with embed/store faked)."""
    import researchops.rag.ingest as ingest_mod
    import researchops.server.main as main_mod

    monkeypatch.setattr(ingest_mod, "Embedder", _FakeEmbedder)
    monkeypatch.setattr(ingest_mod, "QdrantStore", _FakeQdrantStore)
    monkeypatch.setattr(main_mod, "UPLOAD_DIR", tmp_path / "uploads")

    client = TestClient(app)
    resp = client.post(
        "/documents",
        files={"file": ("notes.txt", "Introduction\n\nWe propose a method.", "text/plain")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == "notes"
    assert body["chunks"] >= 1
    assert (tmp_path / "uploads" / "notes.txt").exists()


def test_upload_document_rejects_unsupported(tmp_path, monkeypatch) -> None:
    import researchops.server.main as main_mod

    monkeypatch.setattr(main_mod, "UPLOAD_DIR", tmp_path / "uploads")
    client = TestClient(app)
    resp = client.post(
        "/documents",
        files={"file": ("legacy.doc", b"not a docx", "application/msword")},
    )
    assert resp.status_code == 400
