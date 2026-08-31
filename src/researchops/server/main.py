"""FastAPI application.

Serves the REST/SSE API the web frontend (``web/``, a Next.js app) consumes:

* ``GET  /health``           — liveness probe.
* ``POST /chat/completions`` — thin passthrough over the provider abstraction.
* ``POST /agent/stream``     — run the agent end-to-end and stream progress as
  server-sent events (plan → tool calls → tool results → report → trace).
* ``GET  /experiments``      — persisted experiment/job/metric records.

The agent stream is the interesting one: it builds the full toolset (RAG +
labops + persistence), wraps the LLM and registry in the tracing layer, drives the
LangGraph run with ``astream``, and emits each node transition as one ``data:``
line. Tracing is always on; Langfuse export is opt-in per request.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from researchops.config import Settings, get_settings
from researchops.llm import BaseLLM, build_llm
from researchops.llm.providers import ChatMessage, LLMError
from researchops.rag.parser import SUPPORTED_EXTENSIONS


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.settings = get_settings()
    yield


app = FastAPI(title="ResearchOps Agent", version="0.1.0", lifespan=lifespan)

# Uploaded documents land here (git-ignored), then get ingested into Qdrant.
UPLOAD_DIR = Path(".researchops/uploads")
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _safe_filename(name: str) -> str:
    """Neutralize a user-supplied filename against path traversal / weird chars."""
    stem = Path(name).name
    sanitized = re.sub(r"[^\w.\- ]", "_", stem)
    return sanitized.strip("._ ") or "upload"

# The Next.js dev server runs on :3000 and calls this API cross-origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_llm() -> BaseLLM:
    settings: Settings = app.state.settings
    return build_llm(settings)


class Message(BaseModel):
    role: str = Field(default="user", pattern="^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)


class ChatResponse(BaseModel):
    content: str
    model: str
    input_tokens: int
    output_tokens: int


class AgentRunRequest(BaseModel):
    task: str = Field(min_length=1, max_length=2000)
    max_iterations: int = Field(default=10, ge=1, le=50)
    langfuse: bool = Field(default=False)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


@app.post("/chat/completions", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    llm: BaseLLM = Depends(get_llm),  # noqa: B008 — FastAPI dependency-injection idiom
) -> ChatResponse:
    messages = [ChatMessage(role=m.role, content=m.content) for m in req.messages]
    try:
        resp = await llm.chat(messages, temperature=req.temperature, max_tokens=req.max_tokens)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ChatResponse(
        content=resp.content,
        model=resp.model,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
    )


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


@app.post("/agent/stream")
async def agent_stream(req: AgentRunRequest) -> StreamingResponse:
    """Run the agent and stream progress as server-sent events."""
    settings: Settings = app.state.settings

    async def events() -> AsyncIterator[str]:
        from researchops.agent.stream import stream_agent
        from researchops.agent.tools import build_default_tools
        from researchops.db.store import ExperimentStore
        from researchops.labops import LabClient, SshConnection
        from researchops.observability.langfuse import build_client, export_trace, is_configured
        from researchops.observability.trace import Trace, TracedLLM, TracedToolRegistry
        from researchops.rag.retriever import Retriever

        llm = build_llm(settings)
        retriever = Retriever()
        lab_client = LabClient(SshConnection())
        store = ExperimentStore()
        registry = build_default_tools(retriever, lab_client, store=store)
        trace = Trace(task=req.task)
        traced_llm: BaseLLM = TracedLLM(llm, trace)
        traced_registry = TracedToolRegistry(registry, trace)

        yield _sse({"event": "start", "task": req.task})
        try:
            Path(".researchops").mkdir(exist_ok=True)  # noqa: ASYNC240  # one-time, not hot-path I/O
            await store.init()
            async for event in stream_agent(
                req.task,
                llm=traced_llm,
                registry=traced_registry,
                max_iterations=req.max_iterations,
            ):
                yield _sse(event)
            yield _sse({"event": "trace", "summary": trace.summary()})
            if req.langfuse and is_configured(settings):
                lf = build_client(settings)
                try:
                    url = export_trace(trace, lf=lf)
                    lf.flush()
                    yield _sse({"event": "langfuse", "url": url})
                finally:
                    lf.shutdown()
            yield _sse({"event": "done"})
        finally:
            await retriever.close()
            await lab_client.close()
            await store.close()

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/experiments")
async def experiments() -> list[dict[str, Any]]:
    """List persisted experiments with their runs and metrics (newest first)."""
    from researchops.db.store import ExperimentStore

    store = ExperimentStore()
    try:
        await store.init()
        result: list[dict[str, Any]] = []
        for exp in await store.list_experiments():
            runs: list[dict[str, Any]] = []
            for run in await store.list_runs(exp.id):
                metrics = await store.list_metrics(run.id)
                runs.append(
                    {
                        "id": run.id,
                        "job_id": run.job_id,
                        "command": run.command,
                        "status": run.status,
                        "started_at": run.started_at,
                        "finished_at": run.finished_at,
                        "metrics": [
                            {
                                "name": m.name,
                                "value": m.value,
                                "dataset": m.dataset,
                                "sigma": m.sigma,
                            }
                            for m in metrics
                        ],
                    }
                )
            result.append(
                {
                    "id": exp.id,
                    "name": exp.name,
                    "task": exp.task,
                    "created_at": exp.created_at,
                    "runs": runs,
                }
            )
        return list(reversed(result))
    finally:
        await store.close()


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:  # noqa: B008 — FastAPI upload idiom
    """Upload a document (pdf/docx/txt/md), ingest it into the RAG index.

    Saves the file under ``UPLOAD_DIR``, then runs parse -> chunk -> embed -> upsert.
    Returns the assigned ``doc_id`` and the number of chunks indexed.
    """
    from researchops.rag.ingest import doc_id_from_path, ingest_document
    from researchops.rag.parser import UnsupportedFormatError

    filename = _safe_filename(file.filename or "")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unsupported type {suffix or '(none)'!r}; "
                f"supported: pdf / docx / txt / md (please 'save as .docx' for Word)"
            ),
        )

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file too large (max 50MB)")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 — one-time save, not hot-path I/O
    dest = UPLOAD_DIR / filename
    i = 1
    while dest.exists():
        dest = UPLOAD_DIR / f"{Path(filename).stem}_{i}{suffix}"
        i += 1
    dest.write_bytes(data)

    try:
        chunks = await ingest_document(dest)
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — embedder/Qdrant outage surfaces as 500
        raise HTTPException(status_code=500, detail=f"ingest failed: {exc}") from exc

    return {"filename": dest.name, "doc_id": doc_id_from_path(dest), "chunks": chunks}


@app.get("/documents")
async def documents() -> list[dict[str, Any]]:
    """List the documents currently indexed in the RAG store."""
    from researchops.rag.qdrant_store import QdrantStore

    store = QdrantStore()
    try:
        return await store.list_documents()
    finally:
        await store.close()
