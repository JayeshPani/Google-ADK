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
from job_rejection_agent.observability.phoenix_mcp import query_low_scoring_trace_summaries


class PhoenixTraceSummaryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
