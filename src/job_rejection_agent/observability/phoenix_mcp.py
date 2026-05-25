"""Phoenix MCP and trace-query helpers."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Iterable

from job_rejection_agent.config import Settings, get_settings
from .tracing import apply_phoenix_environment


def build_phoenix_mcp_toolset(settings: Settings | None = None):
    settings = settings or get_settings()
    if not settings.phoenix_mcp_enabled or not settings.phoenix_api_key:
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
                args=list(settings.phoenix_mcp_args),
            ),
        ),
    )


def _build_phoenix_client(settings: Settings) -> Any | None:
    if not apply_phoenix_environment(settings):
        return None
    try:
        from phoenix.client import Client
    except Exception:
        return None
    return Client(base_url=settings.phoenix_base_url, api_key=settings.phoenix_api_key)


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


def _extract_span_attributes(span: dict[str, Any]) -> dict[str, Any]:
    attributes = span.get("attributes", {})
    return attributes if isinstance(attributes, dict) else {}


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


def _extract_mcp_payload(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
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
) -> list[dict[str, Any]]:
    annotations_by_span: dict[str, list[dict[str, Any]]] = {}
    for annotation in annotations:
        span_id = annotation.get("span_id")
        if not isinstance(span_id, str):
            continue
        annotations_by_span.setdefault(span_id, []).append(annotation)

    summaries: list[dict[str, Any]] = []
    for span in spans:
        context = span.get("context", {})
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
        attributes = _extract_span_attributes(span)
        summaries.append(
            {
                "trace_id": context.get("trace_id", "unknown"),
                "span_id": span_id or "unknown",
                "session_id": attributes.get("job_rejection.session_id", ""),
                "name": span.get("name", "unknown"),
                "status": span.get("status_code", "unknown"),
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
) -> list[dict[str, Any]]:
    if not settings.phoenix_mcp_enabled:
        return []
    try:
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError:
        return []

    env = {
        "PHOENIX_API_KEY": settings.phoenix_api_key or "",
        "PHOENIX_HOST": settings.phoenix_base_url,
        "PHOENIX_PROJECT": settings.phoenix_project_name,
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
    }
    server_params = StdioServerParameters(
        command=settings.phoenix_mcp_command,
        args=list(settings.phoenix_mcp_args),
        env=env,
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

            spans: list[dict[str, Any]] = []
            cursor = None
            target = max(limit * 8, 20)
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
                    spans.extend(item for item in batch if isinstance(item, dict))
                cursor = payload.get("nextCursor")
                if not cursor or not batch:
                    break

            session_spans = [
                span for span in spans if span.get("name") == "job_rejection_session"
            ][:target]
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
                            "projectName": settings.phoenix_project_name,
                            "spanIds": batch_ids,
                            "includeAnnotationNames": eval_names,
                            "limit": 1000,
                            **({"cursor": cursor} if cursor else {}),
                        },
                    )
                    payload = _extract_mcp_payload(response)
                    batch = payload.get("annotations", [])
                    if isinstance(batch, list):
                        annotations.extend(item for item in batch if isinstance(item, dict))
                    cursor = payload.get("nextCursor")
                    if not cursor or not batch:
                        break
            return _normalize_trace_summaries(session_spans, annotations, limit=limit)


def _query_trace_summaries_via_mcp(limit: int, settings: Settings) -> list[dict[str, Any]]:
    try:
        return asyncio.run(_query_trace_summaries_via_mcp_async(limit, settings))
    except RuntimeError:
        return []


def _query_trace_summaries_via_client(limit: int, settings: Settings) -> list[dict[str, Any]]:
    client = _build_phoenix_client(settings)
    if client is None:
        return []
    eval_names = [
        "actionability",
        "evidence_grounding",
        "specificity",
        "non_hallucination",
    ]
    try:
        spans = client.spans.get_spans(
            project_identifier=settings.phoenix_project_name,
            name="job_rejection_session",
            limit=max(limit * 8, 20),
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
    return _normalize_trace_summaries(spans, annotations, limit=limit)


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
