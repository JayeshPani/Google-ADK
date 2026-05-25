from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent.ingestion.resume_parser import parse_resume_file


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "resumes"


class ResumeParserTests(unittest.TestCase):
    def test_text_resume_parses_contact_signals(self) -> None:
        parsed = parse_resume_file(FIXTURES / "nisha_ml_newgrad.txt")
        self.assertTrue(parsed.contact_signals["email"])
        self.assertTrue(parsed.contact_signals["linkedin"])
        self.assertIsInstance(parsed.ats_findings, list)


if __name__ == "__main__":
    unittest.main()
