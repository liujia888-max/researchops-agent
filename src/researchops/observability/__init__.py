"""Observability: dependency-free tracing of agent runs (tokens, cost, latency).

The data model mirrors OpenTelemetry/Langfuse — a ``Trace`` of named spans — so the
Langfuse exporter in ``langfuse.py`` is a thin adapter over the same spans. See
``trace.py`` for the local trace and ``langfuse.py`` for the cloud export.
"""

from researchops.observability.langfuse import build_client, export_trace, is_configured
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
    "build_client",
    "export_trace",
    "is_configured",
]
