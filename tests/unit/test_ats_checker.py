from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent.analysis import evaluate_ats_checks, extract_job_requirements, extract_resume_facts
from job_rejection_agent.ingestion import parse_job_description, parse_resume_file


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


class ATSCheckerTests(unittest.TestCase):
    def test_checks_cover_expected_categories(self) -> None:
        resume = extract_resume_facts(parse_resume_file(FIXTURE_ROOT / "resumes" / "arjun_backend_student.txt"))
        requirements = extract_job_requirements(
            parse_job_description((FIXTURE_ROOT / "jds" / "backend_newgrad.md").read_text(encoding="utf-8"))
        )

        checks = evaluate_ats_checks(resume, requirements)

        categories = {item.category for item in checks}
        self.assertEqual(
            categories,
            {
                "section_structure",
                "contact_info",
                "readability",
                "formatting_risk",
                "keyword_coverage",
                "file_hygiene",
            },
        )


if __name__ == "__main__":
    unittest.main()
