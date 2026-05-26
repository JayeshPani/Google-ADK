"""Phoenix + OpenInference tracing bootstrap."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from job_rejection_agent.config import Settings, get_settings


_TRACING_READY = False
LOGGER = logging.getLogger(__name__)


def _phoenix_working_dir(settings: Settings) -> Path:
    return settings.local_storage_path.parent / "phoenix"


def apply_phoenix_environment(settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    if not settings.phoenix_api_key:
        return False
    working_dir = _phoenix_working_dir(settings)
    working_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PHOENIX_WORKING_DIR", str(working_dir))
    os.environ.setdefault("PHOENIX_API_KEY", settings.phoenix_api_key)
    os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", settings.phoenix_collector_endpoint)
    return True


def format_span_id(span_id: int) -> str:
    return f"{span_id:016x}"


def format_trace_id(trace_id: int) -> str:
    return f"{trace_id:032x}"


def _instrument_openinference(tracer_provider: object) -> None:
    from openinference.instrumentation.google_adk import GoogleADKInstrumentor
    from openinference.instrumentation.mcp import MCPInstrumentor

    instrumentors = [
        GoogleADKInstrumentor(),
        MCPInstrumentor(),
    ]
    for instrumentor in instrumentors:
        if getattr(instrumentor, "is_instrumented_by_opentelemetry", False):
            continue
        instrumentor.instrument(tracer_provider=tracer_provider)


def configure_tracing(settings: Settings | None = None) -> bool:
    global _TRACING_READY
    if _TRACING_READY:
        return True
    settings = settings or get_settings()
    if not apply_phoenix_environment(settings):
        return False
    try:
        from phoenix.otel import register
    except ImportError:
        return False
    try:
        tracer_provider = register(
            project_name=settings.phoenix_project_name,
            auto_instrument=False,
        )
        _instrument_openinference(tracer_provider)
    except Exception:
        LOGGER.exception("Failed to configure Phoenix/OpenInference tracing.")
        return False
    _TRACING_READY = True
    return True
