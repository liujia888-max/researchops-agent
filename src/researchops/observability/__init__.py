"""Observability: dependency-free tracing of agent runs (tokens, cost, latency).

The data model mirrors OpenTelemetry/Langfuse — a ``Trace`` of named spans — so a
Langfuse exporter can be added later without touching the agent. See ``trace.py``.
"""

from researchops.observability.trace import (
    LlmSpan,
    ToolSpan,
    Trace,
    TracedLLM,
    TracedToolRegistry,
    traced_run_agent,
)

__all__ = [
    "LlmSpan",
    "ToolSpan",
    "Trace",
    "TracedLLM",
    "TracedToolRegistry",
    "traced_run_agent",
]
