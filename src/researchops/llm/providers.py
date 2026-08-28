"""Unified, provider-agnostic LLM client.

Design notes:
- Every provider speaks the OpenAI-compatible `/chat/completions` shape, so the
  differences (base URL, auth header, model name) are captured in a small config.
- `chat` returns a normalized dataclass; callers never see provider-specific JSON.
- `httpx.AsyncClient` is created per-request for now (simple, safe). A shared,
  pooled client with retry/backoff lands alongside LangGraph in Phase 2.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

import httpx

from researchops.config import Settings


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatResponse:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    raw: dict[str, object] = field(default_factory=dict, repr=False)


class LLMError(Exception):
    """Raised on any provider failure (network, auth, non-2xx)."""


class BaseLLM(abc.ABC):
    """Async chat-completions client over an OpenAI-compatible endpoint."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatResponse:
        """Send a chat request and return a normalized response."""
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name} request failed: {exc}") from exc

        if resp.status_code != 200:
            raise LLMError(f"{self.name} returned {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.name} malformed response: {data}") from exc

        usage = data.get("usage", {})
        return ChatResponse(
            content=content,
            model=self.model,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            raw=data,
        )


class DeepSeekLLM(BaseLLM):
    name = "deepseek"


class QwenLLM(BaseLLM):
    name = "qwen"


class VLLMLLM(BaseLLM):
    name = "vllm"


def build_llm(settings: Settings | None = None, *, provider: str | None = None) -> BaseLLM:
    """Build the configured LLM client.

    `provider` overrides `settings.llm_provider` (useful for tests / demos).
    """
    settings = settings or Settings()
    name = provider or settings.llm_provider

    if name == "deepseek":
        return DeepSeekLLM(settings.deepseek_base_url, settings.deepseek_api_key, settings.deepseek_model)
    if name == "qwen":
        return QwenLLM(settings.qwen_base_url, settings.qwen_api_key, settings.qwen_model)
    if name == "vllm":
        return VLLMLLM(settings.vllm_base_url, settings.vllm_api_key, settings.vllm_model)
    raise ValueError(f"unknown llm_provider: {name!r} (expected deepseek|qwen|vllm)")
