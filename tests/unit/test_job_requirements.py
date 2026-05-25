from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent.analysis import extract_job_requirements
from job_rejection_agent.ingestion import parse_job_description


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "jds"


class JobRequirementTests(unittest.TestCase):
    def test_extracts_required_skills_and_level(self) -> None:
        jd_text = (FIXTURES / "backend_newgrad.md").read_text(encoding="utf-8")
        requirements = extract_job_requirements(parse_job_description(jd_text))
        self.assertIn("python", requirements.required_skills)
        self.assertIn(requirements.experience_level, {"entry", "junior"})


if __name__ == "__main__":
    unittest.main()
