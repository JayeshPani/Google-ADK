from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent.config import Settings
from job_rejection_agent.observability.phoenix_mcp import (
    build_phoenix_mcp_toolset,
    query_low_scoring_trace_summaries,
    query_trace_summary_by_ids,
    query_trace_summary_by_session_id,
)


class PhoenixTraceSummaryTests(unittest.TestCase):
    def test_build_toolset_passes_env_and_extended_timeout(self) -> None:
        settings = Settings(
            phoenix_api_key="phoenix-key",
            phoenix_collector_endpoint="https://app.phoenix.arize.com/s/demo-space",
            phoenix_project_name="job-rejection-agent",
            phoenix_mcp_command="npx",
            phoenix_mcp_args=("-y", "@arizeai/phoenix-mcp@latest"),
        )
        with mock.patch("google.adk.tools.mcp_tool.McpToolset", autospec=True) as toolset_cls:
            with mock.patch(
                "google.adk.tools.mcp_tool.mcp_session_manager.StdioConnectionParams",
                autospec=True,
            ) as connection_params_cls:
                with mock.patch("mcp.StdioServerParameters", autospec=True) as server_params_cls:
                    build_phoenix_mcp_toolset(settings)

        server_params_cls.assert_called_once()
        _, server_kwargs = server_params_cls.call_args
        self.assertEqual(server_kwargs["command"], "npx")
        self.assertEqual(
            server_kwargs["args"],
            ["-y", "@arizeai/phoenix-mcp@latest", "--baseUrl", "https://app.phoenix.arize.com/s/demo-space"],
        )
        self.assertEqual(server_kwargs["env"]["PHOENIX_API_KEY"], "phoenix-key")
        self.assertEqual(server_kwargs["env"]["PHOENIX_PROJECT"], "job-rejection-agent")
        self.assertEqual(server_kwargs["env"]["PHOENIX_HOST"], "https://app.phoenix.arize.com/s/demo-space")
        connection_params_cls.assert_called_once()
        _, connection_kwargs = connection_params_cls.call_args
        self.assertEqual(connection_kwargs["timeout"], 30.0)
        toolset_cls.assert_called_once()

    def test_prefers_mcp_trace_query_when_available(self) -> None:
        settings = Settings(phoenix_api_key="phoenix-key", phoenix_project_name="job-rejection-agent")
        mcp_summaries = [
            {
                "trace_id": "trace-mcp",
                "span_id": "span-mcp",
                "composite_score": 0.25,
                "annotations": [],
                "failure_explanations": [],
            }
        ]
        with mock.patch(
            "job_rejection_agent.observability.phoenix_mcp._query_trace_summaries_via_mcp",
            return_value=mcp_summaries,
        ) as mcp_query:
            with mock.patch(
                "job_rejection_agent.observability.phoenix_mcp._query_trace_summaries_via_client",
                return_value=[],
            ) as client_query:
                summaries = query_low_scoring_trace_summaries(limit=1, settings=settings)

        self.assertEqual(summaries, mcp_summaries)
        mcp_query.assert_called_once()
        client_query.assert_not_called()

    def test_returns_lowest_scoring_session_spans_first(self) -> None:
        spans = [
            {
                "name": "job_rejection_session",
                "status_code": "OK",
                "context": {"trace_id": "trace-1", "span_id": "span-1"},
                "attributes": {
                    "job_rejection.packet_id": "packet-1",
                    "job_rejection.role_title": "ML Intern",
                    "job_rejection.company_name": "Acme",
                    "job_rejection.recommended_decision": "apply_after_patch",
                    "job_rejection.output_preview": "Generic output",
                },
            },
            {
                "name": "job_rejection_session",
                "status_code": "OK",
                "context": {"trace_id": "trace-2", "span_id": "span-2"},
                "attributes": {
                    "job_rejection.packet_id": "packet-2",
                    "job_rejection.role_title": "Backend Intern",
                    "job_rejection.company_name": "Beta",
                    "job_rejection.recommended_decision": "apply_now",
                    "job_rejection.output_preview": "Grounded output",
                },
            },
        ]
        annotations = [
            {
                "span_id": "span-1",
                "name": "actionability",
                "result": {"label": "bad", "score": 0.0, "explanation": "Advice was generic."},
            },
            {
                "span_id": "span-1",
                "name": "specificity",
                "result": {"label": "no", "score": 0.0, "explanation": "No project names were referenced."},
            },
            {
                "span_id": "span-2",
                "name": "actionability",
                "result": {"label": "good", "score": 1.0, "explanation": "Good edits."},
            },
            {
                "span_id": "span-2",
                "name": "specificity",
                "result": {"label": "yes", "score": 1.0, "explanation": "Project names were referenced."},
            },
        ]
        client = mock.Mock()
        client.spans.get_spans.return_value = spans
        client.spans.get_span_annotations.return_value = annotations
        settings = Settings(phoenix_api_key="phoenix-key", phoenix_project_name="job-rejection-agent")

        with mock.patch(
            "job_rejection_agent.observability.phoenix_mcp._query_trace_summaries_via_mcp",
            return_value=[],
        ):
            with mock.patch(
                "job_rejection_agent.observability.phoenix_mcp._build_phoenix_client",
                return_value=client,
            ):
                summaries = query_low_scoring_trace_summaries(limit=2, settings=settings)

        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0]["span_id"], "span-1")
        self.assertLess(summaries[0]["composite_score"], summaries[1]["composite_score"])
        self.assertIn("Advice was generic.", summaries[0]["failure_explanations"])
        self.assertEqual(summaries[0]["packet_id"], "packet-1")

    def test_query_trace_summary_by_session_id_uses_mcp_then_fallback(self) -> None:
        settings = Settings(phoenix_api_key="phoenix-key", phoenix_project_name="job-rejection-agent")
        expected = {
            "trace_id": "trace-123",
            "span_id": "span-123",
            "session_id": "session-123",
            "composite_score": 0.5,
            "annotations": [],
            "failure_explanations": [],
        }
        with mock.patch(
            "job_rejection_agent.observability.phoenix_mcp._query_trace_summaries_via_mcp",
            return_value=[expected],
        ) as mcp_query:
            with mock.patch(
                "job_rejection_agent.observability.phoenix_mcp._query_trace_summaries_via_client",
                return_value=[],
            ) as client_query:
                summary = query_trace_summary_by_session_id("session-123", settings=settings)

        self.assertEqual(summary["trace_id"], expected["trace_id"])
        self.assertEqual(summary["query_source"], "mcp")
        mcp_query.assert_called_once_with(limit=1, settings=settings, session_id="session-123")
        client_query.assert_not_called()

    def test_query_trace_summary_by_ids_prefers_mcp(self) -> None:
        settings = Settings(phoenix_api_key="phoenix-key", phoenix_project_name="job-rejection-agent")
        expected = {
            "trace_id": "trace-123",
            "span_id": "span-123",
            "session_id": "session-123",
            "composite_score": 0.5,
            "annotations": [],
            "failure_explanations": [],
        }
        with mock.patch(
            "job_rejection_agent.observability.phoenix_mcp._query_trace_summary_by_ids_via_mcp",
            return_value=dict(expected),
        ) as mcp_query:
            with mock.patch(
                "job_rejection_agent.observability.phoenix_mcp._query_trace_summary_by_ids_via_client",
                return_value=None,
            ) as client_query:
                summary = query_trace_summary_by_ids(trace_id="trace-123", span_id="span-123", settings=settings)

        self.assertEqual(summary["trace_id"], "trace-123")
        self.assertEqual(summary["query_source"], "mcp")
        mcp_query.assert_called_once_with(trace_id="trace-123", span_id="span-123", settings=settings)
        client_query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
