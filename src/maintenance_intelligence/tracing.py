"""OpenTelemetry tracing.

``configure_tracing`` installs a tracer provider according to ``TRACING_EXPORTER``:

* ``none``    – no-op (default; zero overhead, tests stay quiet)
* ``console`` – spans printed to stdout, handy while developing
* ``otlp``    – OTLP/HTTP to ``OTEL_EXPORTER_OTLP_ENDPOINT`` (Arize Phoenix in
  Compose, any OTLP collector, or an Azure Monitor OTLP ingestion pipeline)
* ``azure``   – Azure Monitor / Application Insights via the optional
  ``azure-monitor-opentelemetry-exporter`` package and
  ``APPLICATIONINSIGHTS_CONNECTION_STRING``

Application code only ever calls :func:`span`, a thin context manager, so the
retrieval, answering, agent, guardrail and LLM layers are instrumented once and
the backend is a deployment decision.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

logger = logging.getLogger("maintenance_intelligence.tracing")

SERVICE_NAME = "maintenance-intelligence-api"
_configured_exporter: str | None = None


def configure_tracing(settings: Any) -> str:
    """Install a global tracer provider once. Returns the exporter actually used."""
    global _configured_exporter
    exporter = (getattr(settings, "tracing_exporter", "none") or "none").lower()
    if _configured_exporter is not None:
        return _configured_exporter
    if exporter == "none":
        _configured_exporter = "none"
        return "none"

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    resource = Resource.create({"service.name": SERVICE_NAME, "service.version": getattr(settings, "version", "0.1.0")})
    provider = TracerProvider(resource=resource)

    if exporter == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    elif exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = getattr(settings, "otlp_endpoint", None)
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()))
    elif exporter == "azure":
        try:
            from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
        except ImportError:
            logger.warning("TRACING_EXPORTER=azure but azure-monitor-opentelemetry-exporter is not installed; tracing disabled")
            _configured_exporter = "none"
            return "none"
        connection = getattr(settings, "applicationinsights_connection_string", None)
        if not connection:
            logger.warning("TRACING_EXPORTER=azure but APPLICATIONINSIGHTS_CONNECTION_STRING is unset; tracing disabled")
            _configured_exporter = "none"
            return "none"
        provider.add_span_processor(BatchSpanProcessor(AzureMonitorTraceExporter(connection_string=connection)))
    else:
        logger.warning("Unknown TRACING_EXPORTER=%s; tracing disabled", exporter)
        _configured_exporter = "none"
        return "none"

    trace.set_tracer_provider(provider)
    _configured_exporter = exporter
    return exporter


def instrument_fastapi(app: Any) -> None:
    """Attach the FastAPI auto-instrumentation when a real exporter is active."""
    if _configured_exporter in (None, "none"):
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="health,ready,metrics")
    except Exception as exc:  # pragma: no cover - instrumentation must never break startup
        logger.warning("FastAPI instrumentation failed: %s", exc)


def tracer() -> trace.Tracer:
    return trace.get_tracer("maintenance_intelligence")


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """Start a span, set scalar attributes, and record exceptions."""
    with tracer().start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is None:
                continue
            if isinstance(value, (str, bool, int, float)):
                current.set_attribute(key, value)
            else:
                current.set_attribute(key, str(value)[:512])
        try:
            yield current
        except Exception as exc:
            current.record_exception(exc)
            current.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
            raise


def record_llm_usage(current: Span, result: Any) -> None:
    """Standard attribute names (OpenTelemetry GenAI semantic conventions)."""
    current.set_attribute("gen_ai.request.model", getattr(result, "model", ""))
    current.set_attribute("gen_ai.usage.input_tokens", int(getattr(result, "prompt_tokens", 0)))
    current.set_attribute("gen_ai.usage.output_tokens", int(getattr(result, "completion_tokens", 0)))
    current.set_attribute("llm.cost_usd", float(getattr(result, "cost_usd", 0.0)))
    current.set_attribute("llm.latency_ms", round(float(getattr(result, "latency_seconds", 0.0)) * 1000, 2))
    current.set_attribute("llm.cached", bool(getattr(result, "cached", False)))
