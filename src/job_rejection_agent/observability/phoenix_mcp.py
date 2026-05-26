"""Phoenix MCP and trace-query helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

from job_rejection_agent.config import Settings, get_settings
from .tracing import apply_phoenix_environment


LOGGER = logging.getLogger(__name__)


def _mcp_command_available(command: str) -> bool:
    candidate = Path(command)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate.exists()
    return shutil.which(command) is not None


def _phoenix_mcp_env(settings: Settings) -> dict[str, str]:
    return {
        "PHOENIX_API_KEY": settings.phoenix_api_key or "",
        "PHOENIX_HOST": settings.phoenix_query_base_url,
        "PHOENIX_PROJECT": settings.phoenix_project_name,
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
    }


def _phoenix_mcp_args(settings: Settings) -> list[str]:
    args = list(settings.phoenix_mcp_args)
    if "--baseUrl" in args:
        index = args.index("--baseUrl")
        if index + 1 < len(args):
            args[index + 1] = settings.phoenix_query_base_url
            return args
    return [*args, "--baseUrl", settings.phoenix_query_base_url]


def build_phoenix_mcp_toolset(settings: Settings | None = None):
    settings = settings or get_settings()
    if not settings.phoenix_mcp_enabled or not settings.phoenix_api_key:
        return None
    if not _mcp_command_available(settings.phoenix_mcp_command):
        LOGGER.warning("Phoenix MCP command '%s' is unavailable; ADK MCP tools disabled.", settings.phoenix_mcp_command)
        return None
    try:
        from google.adk.tools.mcp_tool import McpToolset
        from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
        from mcp import StdioServerParameters
    except ImportError:
        return None
    return McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=settings.phoenix_mcp_command,
                args=_phoenix_mcp_args(settings),
                env=_phoenix_mcp_env(settings),
            ),
            timeout=30.0,
        ),
    )


def _build_phoenix_client(settings: Settings) -> Any | None:
    if not apply_phoenix_environment(settings):
        return None
    try:
        from phoenix.client import Client
    except Exception:
        return None
    return Client(base_url=settings.phoenix_query_base_url, api_key=settings.phoenix_api_key)


def _annotation_score(annotation_name: str, result: dict[str, Any] | None) -> float | None:
    if not result:
        return None
    if result.get("score") is not None:
        try:
            return float(result["score"])
        except (TypeError, ValueError):
            return None
    label = result.get("label")
    if not isinstance(label, str):
        return None
    mappings = {
        "actionability": {"good": 1.0, "needs_improvement": 0.5, "bad": 0.0},
        "evidence_grounding": {"grounded": 1.0, "partially_grounded": 0.5, "weakly_grounded": 0.0},
        "specificity": {"yes": 1.0, "partial": 0.5, "no": 0.0},
        "non_hallucination": {"good": 1.0, "needs_review": 0.5, "bad": 0.0},
    }
    return mappings.get(annotation_name, {}).get(label)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return dumped
    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        dumped = as_dict()
        if isinstance(dumped, dict):
            return dumped
    value_dict = getattr(value, "__dict__", None)
    if isinstance(value_dict, dict):
        return dict(value_dict)
    return {}


def _extract_span_attributes(span: dict[str, Any]) -> dict[str, Any]:
    attributes = _as_dict(span).get("attributes", {})
    return attributes if isinstance(attributes, dict) else {}


def _extract_trace_spans(traces: Iterable[Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for trace in traces:
        trace_payload = _as_dict(trace)
        raw_spans = trace_payload.get("spans", [])
        if not isinstance(raw_spans, list):
            continue
        for span in raw_spans:
            span_payload = _as_dict(span)
            if span_payload:
                spans.append(span_payload)
    return spans


def _batched(items: Iterable[str], size: int) -> list[list[str]]:
    batch: list[str] = []
    batches: list[list[str]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches


def _extract_mcp_payload(result: Any) -> Any:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, (dict, list)):
        return structured
    for block in getattr(result, "content", []):
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", "")
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    continue
        if isinstance(block, dict) and block.get("type") == "text":
            try:
                return json.loads(block.get("text", ""))
            except json.JSONDecodeError:
                continue
    return {}


def _normalize_trace_summaries(
    spans: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    *,
    limit: int,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    annotations_by_span: dict[str, list[dict[str, Any]]] = {}
    for annotation in annotations:
        span_id = annotation.get("span_id")
        if not isinstance(span_id, str):
            continue
        annotations_by_span.setdefault(span_id, []).append(annotation)

    summaries: list[dict[str, Any]] = []
    for span in spans:
        span_payload = _as_dict(span)
        context = span_payload.get("context", {})
        span_id = context.get("span_id", "")
        span_annotations = annotations_by_span.get(span_id, [])
        if not span_annotations:
            continue
        normalized_annotations: list[dict[str, Any]] = []
        scores: list[float] = []
        failure_explanations: list[str] = []
        for annotation in span_annotations:
            annotation_name = annotation.get("name", "unknown")
            result = annotation.get("result", {})
            score = _annotation_score(annotation_name, result if isinstance(result, dict) else None)
            label = result.get("label") if isinstance(result, dict) else None
            explanation = result.get("explanation") if isinstance(result, dict) else None
            if score is not None:
                scores.append(score)
                if score < 0.75 and explanation:
                    failure_explanations.append(str(explanation))
            normalized_annotations.append(
                {
                    "name": annotation_name,
                    "label": label,
                    "score": score,
                    "explanation": explanation,
                }
            )
        if not scores:
            continue
        composite_score = round(sum(scores) / len(scores), 3)
        attributes = _extract_span_attributes(span_payload)
        current_session_id = attributes.get("job_rejection.session_id", "")
        if session_id and current_session_id != session_id:
            continue
        summaries.append(
            {
                "trace_id": context.get("trace_id", "unknown"),
                "span_id": span_id or "unknown",
                "session_id": current_session_id,
                "name": span_payload.get("name", "unknown"),
                "status": span_payload.get("status_code", "unknown"),
                "packet_id": attributes.get("job_rejection.packet_id", ""),
                "role_title": attributes.get("job_rejection.role_title", ""),
                "company_name": attributes.get("job_rejection.company_name", ""),
                "recommended_decision": attributes.get("job_rejection.recommended_decision", ""),
                "score_overall": attributes.get("job_rejection.score_overall"),
                "output_preview": attributes.get("job_rejection.output_preview", ""),
                "top_gap_titles": attributes.get("job_rejection.top_gap_titles", ""),
                "composite_score": composite_score,
                "annotations": normalized_annotations,
                "failure_explanations": failure_explanations,
            }
        )
    return sorted(summaries, key=lambda item: item.get("composite_score", 1.0))[:limit]


async def _query_trace_summaries_via_mcp_async(
    limit: int,
    settings: Settings,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    if not settings.phoenix_mcp_enabled:
        return []
    if not _mcp_command_available(settings.phoenix_mcp_command):
        return []
    try:
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError:
        return []

    server_params = StdioServerParameters(
        command=settings.phoenix_mcp_command,
        args=_phoenix_mcp_args(settings),
        env=_phoenix_mcp_env(settings),
    )
    eval_names = [
        "actionability",
        "evidence_grounding",
        "specificity",
        "non_hallucination",
    ]

    async with stdio_client(server_params) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = {tool.name for tool in tools.tools}
            if "get-spans" not in tool_names or "get-span-annotations" not in tool_names:
                return []

            target = max(limit * 8, 20, 100 if session_id else 20)
            session_spans: list[dict[str, Any]] = []
            if session_id and "list-traces" in tool_names:
                response = await session.call_tool(
                    "list-traces",
                    {
                        "project_identifier": settings.phoenix_project_name,
                        "limit": max(target, 20),
                        "include_annotations": False,
                    },
                )
                payload = _extract_mcp_payload(response)
                raw_traces = payload if isinstance(payload, list) else payload.get("traces", [])
                trace_spans = _extract_trace_spans(raw_traces)
                session_spans = [span for span in trace_spans if span.get("name") == "job_rejection_session"][:target]
            else:
                spans: list[dict[str, Any]] = []
                cursor = None
                while len(spans) < target:
                    args: dict[str, Any] = {
                        "projectName": settings.phoenix_project_name,
                        "limit": min(target, 1000),
                    }
                    if cursor:
                        args["cursor"] = cursor
                    response = await session.call_tool("get-spans", args)
                    payload = _extract_mcp_payload(response)
                    batch = payload.get("spans", [])
                    if isinstance(batch, list):
                        spans.extend(_as_dict(item) for item in batch if _as_dict(item))
                    cursor = payload.get("nextCursor")
                    if not cursor or not batch:
                        break
                session_spans = [span for span in spans if span.get("name") == "job_rejection_session"][:target]
            if not session_spans:
                return []
            span_ids = [
                span.get("context", {}).get("span_id", "")
                for span in session_spans
                if isinstance(span.get("context", {}), dict) and span.get("context", {}).get("span_id")
            ]
            annotations: list[dict[str, Any]] = []
            for batch_ids in _batched(span_ids, 100):
                cursor = None
                while True:
                    response = await session.call_tool(
                        "get-span-annotations",
                        {
                            "project_identifier": settings.phoenix_project_name,
                            "span_ids": batch_ids,
                            "include_annotation_names": eval_names,
                            "limit": 1000,
                        },
                    )
                    payload = _extract_mcp_payload(response)
                    batch = payload.get("annotations", [])
                    if isinstance(batch, list):
                        annotations.extend(item for item in batch if isinstance(item, dict))
                    cursor = payload.get("nextCursor")
                    if not cursor or not batch:
                        break
            return _normalize_trace_summaries(session_spans, annotations, limit=limit, session_id=session_id)


def _query_trace_summaries_via_mcp(
    limit: int,
    settings: Settings,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    try:
        return asyncio.run(_query_trace_summaries_via_mcp_async(limit, settings, session_id=session_id))
    except RuntimeError:
        return []


def _query_trace_summaries_via_client(
    limit: int,
    settings: Settings,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    client = _build_phoenix_client(settings)
    if client is None:
        return []
    eval_names = [
        "actionability",
        "evidence_grounding",
        "specificity",
        "non_hallucination",
    ]
    if session_id:
        try:
            traces = client.traces.get_traces(
                project_identifier=settings.phoenix_project_name,
                session_id=session_id,
                include_spans=True,
                limit=max(limit, 10),
            )
            session_spans = [
                span for span in _extract_trace_spans(traces) if span.get("name") == "job_rejection_session"
            ]
            if session_spans:
                annotations = client.spans.get_span_annotations(
                    span_ids=[span.get("context", {}).get("span_id", "") for span in session_spans if span.get("context", {}).get("span_id")],
                    project_identifier=settings.phoenix_project_name,
                    include_annotation_names=eval_names,
                )
                return _normalize_trace_summaries(session_spans, annotations, limit=limit, session_id=session_id)
        except Exception:
            pass
    try:
        span_filters: dict[str, Any] = {}
        if session_id:
            span_filters["job_rejection.session_id"] = session_id
        spans = client.spans.get_spans(
            project_identifier=settings.phoenix_project_name,
            name="job_rejection_session",
            attributes=span_filters or None,
            limit=max(limit * 8, 20, 100 if session_id else 20),
        )
    except Exception:
        return []
    if not spans:
        return []
    try:
        annotations = client.spans.get_span_annotations(
            spans=spans,
            project_identifier=settings.phoenix_project_name,
            include_annotation_names=eval_names,
        )
    except Exception:
        return []
    return _normalize_trace_summaries(spans, annotations, limit=limit, session_id=session_id)


def query_low_scoring_trace_summaries(
    limit: int = 5,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    settings = settings or get_settings()
    if not settings.phoenix_api_key:
        return []
    summaries = _query_trace_summaries_via_mcp(limit, settings)
    if summaries:
        return summaries
    return _query_trace_summaries_via_client(limit, settings)


def query_recent_trace_summaries(limit: int = 5, settings: Settings | None = None) -> list[dict[str, Any]]:
    return query_low_scoring_trace_summaries(limit=limit, settings=settings)


def query_trace_summary_by_session_id(
    session_id: str,
    *,
    settings: Settings | None = None,
) -> dict[str, Any] | None:
    settings = settings or get_settings()
    if not settings.phoenix_api_key or not session_id:
        return None
    summaries = _query_trace_summaries_via_mcp(limit=1, settings=settings, session_id=session_id)
    if not summaries:
        summaries = _query_trace_summaries_via_client(limit=1, settings=settings, session_id=session_id)
        if summaries:
            summaries[0]["query_source"] = "client"
    elif summaries:
        summaries[0]["query_source"] = "mcp"
    return summaries[0] if summaries else None


def query_trace_summary_by_ids(
    *,
    trace_id: str,
    span_id: str,
    settings: Settings | None = None,
) -> dict[str, Any] | None:
    settings = settings or get_settings()
    if not settings.phoenix_api_key or not trace_id or not span_id:
        return None

    mcp_summaries = _query_trace_summary_by_ids_via_mcp(trace_id=trace_id, span_id=span_id, settings=settings)
    if mcp_summaries:
        mcp_summaries["query_source"] = "mcp"
        return mcp_summaries

    client_summary = _query_trace_summary_by_ids_via_client(trace_id=trace_id, span_id=span_id, settings=settings)
    if client_summary:
        client_summary["query_source"] = "client"
    return client_summary


def _query_trace_summary_by_ids_via_mcp(
    *,
    trace_id: str,
    span_id: str,
    settings: Settings,
) -> dict[str, Any] | None:
    try:
        return asyncio.run(_query_trace_summary_by_ids_via_mcp_async(trace_id=trace_id, span_id=span_id, settings=settings))
    except RuntimeError:
        return None


async def _query_trace_summary_by_ids_via_mcp_async(
    *,
    trace_id: str,
    span_id: str,
    settings: Settings,
) -> dict[str, Any] | None:
    if not settings.phoenix_mcp_enabled:
        return None
    if not _mcp_command_available(settings.phoenix_mcp_command):
        return None
    try:
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError:
        return None

    server_params = StdioServerParameters(
        command=settings.phoenix_mcp_command,
        args=_phoenix_mcp_args(settings),
        env=_phoenix_mcp_env(settings),
    )

    async with stdio_client(server_params) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = {tool.name for tool in tools.tools}
            if "get-trace" not in tool_names or "get-span-annotations" not in tool_names:
                return None
            response = await session.call_tool(
                "get-trace",
                {
                    "project_identifier": settings.phoenix_project_name,
                    "trace_id": trace_id,
                    "include_annotations": False,
                },
            )
            payload = _extract_mcp_payload(response)
            if isinstance(payload, list):
                trace_payload = _as_dict(payload[0]) if payload else {}
            elif isinstance(payload, dict):
                trace_payload = _as_dict(payload.get("trace", payload))
            else:
                trace_payload = {}
            spans = [span for span in _extract_trace_spans([trace_payload]) if span.get("context", {}).get("span_id") == span_id]
            if not spans:
                return None
            response = await session.call_tool(
                "get-span-annotations",
                {
                    "project_identifier": settings.phoenix_project_name,
                    "span_ids": [span_id],
                    "include_annotation_names": [
                        "actionability",
                        "evidence_grounding",
                        "specificity",
                        "non_hallucination",
                    ],
                    "limit": 100,
                },
            )
            annotations = _extract_mcp_payload(response).get("annotations", [])
            summaries = _normalize_trace_summaries(spans, annotations, limit=1)
            return summaries[0] if summaries else None


def _query_trace_summary_by_ids_via_client(
    *,
    trace_id: str,
    span_id: str,
    settings: Settings,
) -> dict[str, Any] | None:
    client = _build_phoenix_client(settings)
    if client is None:
        return None
    try:
        spans = client.spans.get_spans(
            project_identifier=settings.phoenix_project_name,
            trace_ids=[trace_id],
            name="job_rejection_session",
            limit=10,
        )
        annotations = client.spans.get_span_annotations(
            span_ids=[span_id],
            project_identifier=settings.phoenix_project_name,
            include_annotation_names=[
                "actionability",
                "evidence_grounding",
                "specificity",
                "non_hallucination",
            ],
            limit=100,
        )
    except Exception:
        return None
    filtered_spans = [span for span in spans if _as_dict(span).get("context", {}).get("span_id") == span_id]
    summaries = _normalize_trace_summaries(filtered_spans, annotations, limit=1)
    return summaries[0] if summaries else None
