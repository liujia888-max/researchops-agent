"""Unified, provider-agnostic LLM client.

Design notes:
- Every provider speaks the OpenAI-compatible `/chat/completions` shape, so the
  differences (base URL, auth header, model name) are captured in a small config.
- `chat` returns a normalized dataclass; callers never see provider-specific JSON.
- Function calling is supported through the same `chat` method: pass OpenAI-format
  `tools` schemas and read `ChatResponse.tool_calls` / `raw_tool_calls` (the latter
  is echoed back verbatim on the next turn, as OpenAI-compatible APIs require).
- `httpx.AsyncClient` is created per-request for now (simple, safe). A shared,
  pooled client with retry/backoff lands alongside LangGraph in Phase 2.
"""

from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from researchops.config import Settings


@dataclass(frozen=True)
class ChatMessage:
    """One conversation turn. ``name``/``tool_call_id``/``tool_calls`` are only set
    for the tool-calling turns the agent layer emits (tool results + assistant calls).
    """

    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ToolCall:
    """A parsed function call the model requested."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatResponse:
    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw_tool_calls: list[dict[str, Any]] = field(default_factory=list, repr=False)
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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
    ) -> ChatResponse:
        """Send a chat request (optionally with tool schemas) and return a response."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [self._to_payload(m) for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice
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
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.name} malformed response: {data}") from exc

        raw_tool_calls: list[dict[str, Any]] = message.get("tool_calls") or []
        tool_calls = [self._parse_tool_call(tc) for tc in raw_tool_calls]

        usage = data.get("usage", {})
        return ChatResponse(
            content=content,
            model=self.model,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            tool_calls=tool_calls,
            raw_tool_calls=raw_tool_calls,
            raw=data,
        )

    @staticmethod
    def _to_payload(message: ChatMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.name is not None:
            payload["name"] = message.name
        if message.tool_call_id is not None:
            payload["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            payload["tool_calls"] = message.tool_calls
        return payload

    @staticmethod
    def _parse_tool_call(raw: dict[str, Any]) -> ToolCall:
        fn = raw.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        return ToolCall(
            id=str(raw.get("id") or ""),
            name=str(fn.get("name") or ""),
            arguments=args if isinstance(args, dict) else {},
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
