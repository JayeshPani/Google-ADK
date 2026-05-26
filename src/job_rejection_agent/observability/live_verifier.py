"""Shared live verification helpers for ADK, Phoenix, and Phoenix MCP."""

from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any
import uuid

from job_rejection_agent.config import Settings, get_settings

from .phoenix_mcp import _mcp_command_available, query_trace_summary_by_ids, query_trace_summary_by_session_id


REQUIRED_ANNOTATION_NAMES = {
    "actionability",
    "evidence_grounding",
    "specificity",
    "non_hallucination",
}


def get_live_network_test_skip_reason(settings: Settings | None = None) -> str | None:
    settings = settings or get_settings()
    enabled = os.getenv("RUN_LIVE_NETWORK_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return "Set RUN_LIVE_NETWORK_TESTS=1 to run live Gemini/Phoenix/MCP integration tests."
    if not settings.google_genai_enabled:
        if settings.google_genai_use_vertexai:
            return "Configure Vertex AI with GOOGLE_GENAI_USE_VERTEXAI=true, GOOGLE_CLOUD_PROJECT, and GOOGLE_CLOUD_LOCATION."
        return "GOOGLE_API_KEY is required for live ADK integration tests."
    if not settings.phoenix_api_key:
        return "PHOENIX_API_KEY is required for live Phoenix integration tests."
    if not settings.phoenix_project_name:
        return "PHOENIX_PROJECT_NAME must be configured."
    if not settings.phoenix_mcp_enabled:
        return "PHOENIX_MCP_ENABLED must be true for live MCP integration tests."
    if not _mcp_command_available(settings.phoenix_mcp_command):
        return f"Phoenix MCP command '{settings.phoenix_mcp_command}' is unavailable."
    return None


def wait_for_trace_readback(
    *,
    session_id: str = "",
    trace_id: str = "",
    span_id: str = "",
    attempts: int = 6,
    poll_interval_seconds: float = 5.0,
) -> dict[str, Any] | None:
    summary = None
    for attempt in range(max(attempts, 1)):
        if trace_id and span_id:
            summary = query_trace_summary_by_ids(trace_id=trace_id, span_id=span_id)
        elif session_id:
            summary = query_trace_summary_by_session_id(session_id)
        else:
            return None
        if summary:
            return summary
        if attempt < attempts - 1:
            time.sleep(poll_interval_seconds)
    return summary


def verify_live_stack_run(
    *,
    resume_path: str | Path,
    jd_text: str,
    rejection_notes: str,
    user_id: str | None = None,
    settings: Settings | None = None,
    attempts: int = 6,
    poll_interval_seconds: float = 5.0,
) -> dict[str, Any]:
    from job_rejection_agent.agents.root_agent import AgentRuntime

    settings = settings or get_settings()
    runtime = AgentRuntime(settings=settings)
    run_result = runtime.run_diagnostic(
        resume_path=str(resume_path),
        jd_text=jd_text,
        rejection_notes=rejection_notes,
        user_id=user_id or f"live-integration-{uuid.uuid4().hex[:12]}",
    )

    summary = wait_for_trace_readback(
        session_id=run_result.get("session_id", ""),
        trace_id=run_result.get("trace_id", ""),
        span_id=run_result.get("root_span_id", ""),
        attempts=attempts,
        poll_interval_seconds=poll_interval_seconds,
    )
    annotation_names = {
        item.get("name")
        for item in (summary or {}).get("annotations", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    missing_annotation_names = sorted(REQUIRED_ANNOTATION_NAMES - annotation_names)
    return {
        "verification_status": "ok" if summary and not missing_annotation_names else "failed",
        "result": run_result,
        "readback_summary": summary,
        "query_source": (summary or {}).get("query_source"),
        "annotation_names": sorted(annotation_names),
        "missing_annotation_names": missing_annotation_names,
    }
