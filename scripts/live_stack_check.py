"""Live verification for ADK -> Phoenix trace/evals -> Phoenix MCP readback."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent.observability import evaluate_packet
from job_rejection_agent.observability.live_verifier import verify_live_stack_run, wait_for_trace_readback
from job_rejection_agent.services import DiagnosticService


def _print_summary(summary: dict[str, object] | None) -> int:
    if not summary:
        print("verification_status=failed reason=missing_trace_readback")
        return 1
    annotation_names = [item.get("name") for item in summary.get("annotations", [])] if isinstance(summary.get("annotations", []), list) else []
    print(
        {
            "verification_status": "ok",
            "query_source": summary.get("query_source"),
            "readback_session_id": summary.get("session_id"),
            "readback_trace_id": summary.get("trace_id"),
            "readback_span_id": summary.get("span_id"),
            "annotation_names": annotation_names,
            "composite_score": summary.get("composite_score"),
        }
    )
    return 0


def _readback_session(session_id: str) -> int:
    summary = wait_for_trace_readback(session_id=session_id)
    return _print_summary(summary)


def _readback_trace(trace_id: str, span_id: str) -> int:
    summary = wait_for_trace_readback(trace_id=trace_id, span_id=span_id)
    return _print_summary(summary)


def _debug_readback(trace_id: str, span_id: str, session_id: str = "") -> int:
    from job_rejection_agent.config import get_settings
    from job_rejection_agent.observability.phoenix_mcp import (
        _extract_mcp_payload,
        _phoenix_mcp_args,
        _phoenix_mcp_env,
        _query_trace_summaries_via_client,
        _query_trace_summaries_via_mcp,
        _query_trace_summary_by_ids_via_client,
        _query_trace_summary_by_ids_via_mcp,
    )

    settings = get_settings()
    print(
        {
            "project_name": settings.phoenix_project_name,
            "base_url": settings.phoenix_query_base_url,
            "trace_id": trace_id,
            "span_id": span_id,
            "session_id": session_id,
        }
    )
    print("mcp_by_ids", _query_trace_summary_by_ids_via_mcp(trace_id=trace_id, span_id=span_id, settings=settings))
    print("client_by_ids", _query_trace_summary_by_ids_via_client(trace_id=trace_id, span_id=span_id, settings=settings))
    if session_id:
        try:
            print("mcp_by_session", _query_trace_summaries_via_mcp(limit=3, settings=settings, session_id=session_id))
        except Exception as exc:
            print("mcp_by_session_error", repr(exc))
        try:
            print("client_by_session", _query_trace_summaries_via_client(limit=3, settings=settings, session_id=session_id))
        except Exception as exc:
            print("client_by_session_error", repr(exc))
    try:
        from job_rejection_agent.observability.tracing import apply_phoenix_environment
        apply_phoenix_environment(settings)
        from phoenix.client import Client

        client = Client(base_url=settings.phoenix_query_base_url, api_key=settings.phoenix_api_key)
        traces = client.traces.get_traces(
            project_identifier=settings.phoenix_project_name,
            include_spans=True,
            limit=3,
        )
        print("client_recent_traces", [getattr(trace, "trace_id", None) or getattr(trace, "traceId", None) for trace in traces])
        spans = client.spans.get_spans(
            project_identifier=settings.phoenix_project_name,
            limit=3,
        )
        print("client_recent_spans", [_coerce_span_preview(span) for span in spans])
        root_spans = client.spans.get_spans(
            project_identifier=settings.phoenix_project_name,
            name="job_rejection_session",
            limit=5,
        )
        print("client_job_rejection_spans", [_coerce_span_preview(span) for span in root_spans])
    except Exception as exc:
        print("client_debug_error", repr(exc))
    try:
        import asyncio
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        async def _dump_mcp() -> None:
            server_params = StdioServerParameters(
                command=settings.phoenix_mcp_command,
                args=_phoenix_mcp_args(settings),
                env=_phoenix_mcp_env(settings),
            )
            async with stdio_client(server_params) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    tool_map = {tool.name: tool for tool in tools.tools}
                    print("mcp_tool_names", sorted(tool_map))
                    if "get-trace" in tool_map:
                        print("mcp_get_trace_schema", getattr(tool_map["get-trace"], "inputSchema", None))
                        raw_trace = await session.call_tool(
                            "get-trace",
                            {
                                "project_identifier": settings.phoenix_project_name,
                                "trace_id": trace_id,
                                "include_annotations": False,
                            },
                        )
                        print("mcp_get_trace_payload", _extract_mcp_payload(raw_trace))
                    if "get-span-annotations" in tool_map:
                        print("mcp_get_span_annotations_schema", getattr(tool_map["get-span-annotations"], "inputSchema", None))
                        raw_annotations = await session.call_tool(
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
                                "limit": 10,
                            },
                        )
                        print("mcp_get_span_annotations_payload", _extract_mcp_payload(raw_annotations))
                    if session_id and "list-traces" in tool_map:
                        print("mcp_list_traces_schema", getattr(tool_map["list-traces"], "inputSchema", None))
                        raw_traces = await session.call_tool(
                            "list-traces",
                            {
                                "project_identifier": settings.phoenix_project_name,
                                "limit": 3,
                                "include_annotations": False,
                            },
                        )
                        print("mcp_list_traces_payload", _extract_mcp_payload(raw_traces))

        asyncio.run(_dump_mcp())
    except Exception as exc:
        print("mcp_debug_error", repr(exc))
    return 0


def _repair_annotations(packet_id: str, span_id: str) -> int:
    service = DiagnosticService()
    packet = service.tracker.get(packet_id)
    if packet is None:
        print("verification_status=failed reason=missing_packet")
        return 1
    print(
        {
            "repair_packet_id": packet_id,
            "repair_span_id": span_id,
            "eval_scores": evaluate_packet(packet, span_id=span_id, settings=service.settings),
        }
    )
    return 0


def _coerce_span_preview(span: object) -> dict[str, object]:
    model_dump = getattr(span, "model_dump", None)
    if callable(model_dump):
        payload = model_dump()
    elif isinstance(getattr(span, "__dict__", None), dict):
        payload = dict(span.__dict__)
    elif isinstance(span, dict):
        payload = span
    else:
        payload = {}
    context = payload.get("context", {}) if isinstance(payload, dict) else {}
    return {
        "name": payload.get("name") if isinstance(payload, dict) else None,
        "trace_id": context.get("trace_id") if isinstance(context, dict) else None,
        "span_id": context.get("span_id") if isinstance(context, dict) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", default="", help="Only verify readback for an existing session ID.")
    parser.add_argument("--trace-id", default="", help="Trace ID for an existing verification run.")
    parser.add_argument("--span-id", default="", help="Root span ID for an existing verification run.")
    parser.add_argument("--packet-id", default="", help="Saved packet ID for re-logging evaluations.")
    parser.add_argument("--repair-annotations", action="store_true", help="Re-log evaluations for an existing packet/span.")
    parser.add_argument("--debug", action="store_true", help="Print raw lookup results for troubleshooting.")
    args = parser.parse_args()

    if args.repair_annotations:
        if not (args.packet_id and args.span_id):
            print("verification_status=failed reason=packet_and_span_required")
            return 1
        return _repair_annotations(args.packet_id, args.span_id)

    if args.debug:
        return _debug_readback(args.trace_id, args.span_id, args.session_id)

    if args.trace_id or args.span_id:
        if not (args.trace_id and args.span_id):
            print("verification_status=failed reason=trace_and_span_required_together")
            return 1
        return _readback_trace(args.trace_id, args.span_id)

    if args.session_id:
        return _readback_session(args.session_id)

    verification = verify_live_stack_run(
        resume_path=ROOT / "tests" / "fixtures" / "resumes" / "nisha_ml_newgrad.txt",
        jd_text=(ROOT / "tests" / "fixtures" / "jds" / "ml_platform_engineer.md").read_text(encoding="utf-8"),
        rejection_notes="Recruiter said the profile felt promising but not yet production-ready.",
        user_id="live-stack-check",
    )
    result = verification["result"]

    print(
        {
            "used_adk": result.get("used_adk"),
            "session_id": result.get("session_id"),
            "packet_id": result.get("packet_id"),
            "trace_id": result.get("trace_id"),
            "root_span_id": result.get("root_span_id"),
            "eval_scores": result.get("eval_scores"),
        }
    )
    return _print_summary(verification.get("readback_summary"))


if __name__ == "__main__":
    raise SystemExit(main())
