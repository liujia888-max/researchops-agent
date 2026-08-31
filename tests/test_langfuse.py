"""Tests for the Langfuse exporter (adapter over the ``Trace`` span model).

No network, no real SDK: the exporter is exercised against a fake Langfuse client
that records every ``start_as_current_observation`` call, and ``build_client`` is
checked with the ``Langfuse`` constructor monkeypatched.
"""

from __future__ import annotations

from typing import Any

import pytest

from researchops.config import Settings
from researchops.observability.langfuse import (
    build_client,
    export_trace,
    is_configured,
)
from researchops.observability.trace import LlmSpan, ToolSpan, Trace


class _FakeObservation:
    """Context manager that records the kwargs it was started with."""

    def __init__(self, parent: _FakeLangfuse, kwargs: dict[str, Any]) -> None:
        self._parent = parent
        self._kwargs = kwargs

    def __enter__(self) -> _FakeObservation:
        self._parent.started.append(self._kwargs)
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False


class _FakeLangfuse:
    def __init__(self, trace_id: str = "trace-abc") -> None:
        self.started: list[dict[str, Any]] = []
        self._trace_id = trace_id

    def start_as_current_observation(self, **kwargs: Any) -> _FakeObservation:
        return _FakeObservation(self, kwargs)

    def get_current_trace_id(self) -> str | None:
        return self._trace_id

    def get_trace_url(self, *, trace_id: str | None = None) -> str:
        return f"https://cloud.langfuse.com/trace/{trace_id}"

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


def _settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "langfuse_public_key": "pk-x",
        "langfuse_secret_key": "sk-x",
        "langfuse_base_url": "https://cloud.langfuse.com",
    }
    values.update(overrides)
    return Settings(**values)


def test_is_configured_requires_both_keys() -> None:
    assert is_configured(_settings()) is True
    assert is_configured(_settings(langfuse_public_key="")) is False
    assert is_configured(_settings(langfuse_secret_key="")) is False


def test_build_client_passes_keys_and_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeCtor:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("langfuse.Langfuse", _FakeCtor)
    build_client(_settings())

    assert captured == {
        "public_key": "pk-x",
        "secret_key": "sk-x",
        "base_url": "https://cloud.langfuse.com",
    }


def test_export_trace_builds_span_tree_and_returns_url() -> None:
    trace = Trace(task="reproduce Restormer on CBSD68")
    trace.record_llm(LlmSpan(node="planner", model="m", input_tokens=100, output_tokens=10, latency_s=0.1))
    trace.record_llm(LlmSpan(node="executor", model="m", input_tokens=200, output_tokens=30, latency_s=0.2))
    trace.record_tool(ToolSpan(name="rag_search", latency_s=0.05))
    trace.record_llm(LlmSpan(node="reporter", model="m", input_tokens=300, output_tokens=60, latency_s=0.3))

    lf = _FakeLangfuse()
    url = export_trace(trace, lf=lf)

    assert url == "https://cloud.langfuse.com/trace/trace-abc"
    # Root span, then the three LLM generations, then the one tool span.
    assert [s["name"] for s in lf.started] == [
        "researchops-agent",
        "planner",
        "executor",
        "reporter",
        "rag_search",
    ]
    assert [s["as_type"] for s in lf.started] == [
        "span",
        "generation",
        "generation",
        "generation",
        "tool",
    ]
    root = lf.started[0]
    assert root["input"] == "reproduce Restormer on CBSD68"
    assert root["metadata"]["llm_calls"] == 3
    assert root["metadata"]["tool_calls"] == 1

    planner = lf.started[1]
    assert planner["model"] == "m"
    assert planner["usage_details"] == {"input": 100, "output": 10, "total": 110}
    from researchops.config import get_settings
    cfg = get_settings()
    assert planner["cost_details"]["total"] == pytest.approx(
        (100 * cfg.llm_input_price_per_1m + 10 * cfg.llm_output_price_per_1m) / 1_000_000
    )

    rag = lf.started[4]
    assert rag["metadata"]["latency_s"] == 0.05
