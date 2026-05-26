from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent.config import get_settings
from job_rejection_agent.observability.live_verifier import (
    REQUIRED_ANNOTATION_NAMES,
    get_live_network_test_skip_reason,
    verify_live_stack_run,
)


class LiveStackIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = get_settings()
        skip_reason = get_live_network_test_skip_reason(cls.settings)
        if skip_reason:
            raise unittest.SkipTest(skip_reason)

    def test_adk_trace_eval_and_mcp_readback(self) -> None:
        verification = verify_live_stack_run(
            resume_path=ROOT / "tests" / "fixtures" / "resumes" / "nisha_ml_newgrad.txt",
            jd_text=(ROOT / "tests" / "fixtures" / "jds" / "ml_platform_engineer.md").read_text(encoding="utf-8"),
            rejection_notes="Recruiter said the profile felt promising but not yet production-ready.",
            settings=self.settings,
            attempts=6,
            poll_interval_seconds=5.0,
        )

        self.assertEqual(verification["verification_status"], "ok", verification)
        self.assertEqual(verification["query_source"], "mcp", verification)

        result = verification["result"]
        self.assertTrue(result.get("used_adk"), verification)
        self.assertTrue(result.get("session_id"), verification)
        self.assertTrue(result.get("packet_id"), verification)
        self.assertTrue(result.get("trace_id"), verification)
        self.assertTrue(result.get("root_span_id"), verification)

        summary = verification["readback_summary"]
        self.assertIsNotNone(summary, verification)
        self.assertEqual(summary.get("session_id"), result.get("session_id"), verification)

        annotation_names = set(verification["annotation_names"])
        self.assertTrue(REQUIRED_ANNOTATION_NAMES.issubset(annotation_names), verification)


if __name__ == "__main__":
    unittest.main()
