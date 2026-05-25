from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent.services import DiagnosticService


ROOT = Path(__file__).resolve().parents[1]


class EndToEndDiagnosticTests(unittest.TestCase):
    def test_fixture_flow_returns_packet(self) -> None:
        service = DiagnosticService()
        result = service.diagnose(
            resume_path=ROOT / "fixtures" / "resumes" / "arjun_backend_student.txt",
            jd_text=(ROOT / "fixtures" / "jds" / "backend_newgrad.md").read_text(encoding="utf-8"),
            rejection_notes="We liked the project work but wanted more evidence of API ownership.",
            persist=False,
        )
        self.assertTrue(result.report_markdown.startswith("## Match Score"))
        self.assertGreater(result.packet.report.score_overall, 0)
        self.assertEqual(result.packet.job_requirements.role_title, "Backend Software Engineer, New Grad")


if __name__ == "__main__":
    unittest.main()
