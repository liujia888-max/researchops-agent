"""FastAPI application skeleton.

Phase 0 ships only `/health` and a minimal `/chat/completions` passthrough that
proves the provider abstraction works end-to-end. The SSE streaming, RAG, and
agent routes land in Phases 1–3.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from researchops.config import Settings, get_settings
from researchops.llm import BaseLLM, build_llm
from researchops.llm.providers import ChatMessage, LLMError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.settings = get_settings()
    yield


app = FastAPI(title="ResearchOps Agent", version="0.1.0", lifespan=lifespan)


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
