"""Langfuse exporter for the dependency-free :class:`Trace` layer.

The observability data model in :mod:`researchops.observability.trace` maps 1:1
onto Langfuse's span tree, so exporting is a thin adapter: the root ``Trace``
becomes a root span and each ``LlmSpan`` / ``ToolSpan`` becomes a child observation
in chronological order.

Langfuse v4 is OpenTelemetry-native — a trace is created implicitly when the first
root observation starts, so there is no ``lf.trace(...)`` call. Each observation is
opened with ``start_as_current_observation`` and closed on context exit; the
``usage_details`` / ``cost_details`` keywords carry token usage and USD cost so the
Langfuse dashboard can render the cost panel, and ``metadata`` keeps the local
latency figures (Langfuse's own span wall-clock is ~0 for a post-hoc export, so the
truthful latency numbers travel as attributes rather than being dropped).

``langfuse`` is imported lazily: it pulls in the OpenTelemetry SDK, so it stays out
of the import path unless an export is actually requested.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from researchops.observability.trace import Trace, llm_prices

if TYPE_CHECKING:
    from langfuse import Langfuse

    from researchops.config import Settings

_ROOT_SPAN_NAME = "researchops-agent"


def is_configured(settings: Settings) -> bool:
    """True when both Langfuse keys are present (exporter is usable)."""
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def build_client(settings: Settings) -> Langfuse:
    """Construct a Langfuse client from settings (lazy-imports the SDK)."""
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        base_url=settings.langfuse_base_url,
    )


def _span_cost_usd(input_tokens: int, output_tokens: int) -> dict[str, float]:
    input_price, output_price = llm_prices()
    input_usd = input_tokens * input_price / 1_000_000
    output_usd = output_tokens * output_price / 1_000_000
    return {"input": input_usd, "output": output_usd, "total": input_usd + output_usd}


def export_trace(trace: Trace, *, lf: Langfuse) -> str | None:
    """Push ``trace`` to Langfuse as a trace of observations.

    Returns the Langfuse trace URL, or ``None`` when no trace id could be captured.
    The caller is responsible for ``lf.flush()`` / ``lf.shutdown()`` afterwards.
    """
    with lf.start_as_current_observation(
        name=_ROOT_SPAN_NAME,
        as_type="span",
        input=trace.task,
        metadata={
            "llm_calls": len(trace.llm_spans),
            "tool_calls": len(trace.tool_spans),
            "total_tokens": trace.total_tokens,
            "estimated_cost_usd": trace.estimated_cost_usd(),
        },
    ):
        for span in trace.llm_spans:
            with lf.start_as_current_observation(
                name=span.node,
                as_type="generation",
                model=span.model,
                usage_details={
                    "input": span.input_tokens,
                    "output": span.output_tokens,
                    "total": span.input_tokens + span.output_tokens,
                },
                cost_details=_span_cost_usd(span.input_tokens, span.output_tokens),
                metadata={"latency_s": span.latency_s},
            ):
                pass
        for tool in trace.tool_spans:
            with lf.start_as_current_observation(
                name=tool.name,
                as_type="tool",
                metadata={"latency_s": tool.latency_s},
            ):
                pass
        trace_id = lf.get_current_trace_id()

    if trace_id:
        return lf.get_trace_url(trace_id=trace_id)
    return None
