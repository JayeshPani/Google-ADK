from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent.config import Settings  # noqa: E402
from job_rejection_agent.domain import SavedJobPacket  # noqa: E402
from job_rejection_agent.services import DiagnosticService  # noqa: E402
from job_rejection_agent.services.diagnostic_service import GeminiAugmenter  # noqa: E402


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


class ScoringSystemTests(unittest.TestCase):
    def _service(self, *, google: bool = False) -> DiagnosticService:
        return DiagnosticService(settings=Settings(google_api_key="test-key" if google else None, phoenix_api_key=None))

    def _diagnose_fixture(self, resume_name: str, jd_name: str, *, google: bool = False):
        return self._service(google=google).diagnose(
            resume_path=FIXTURE_ROOT / "resumes" / resume_name,
            jd_text=(FIXTURE_ROOT / "jds" / jd_name).read_text(encoding="utf-8"),
            user_id="score-user",
            session_id="score-session",
            persist=False,
        ).packet

    def test_soft_requirements_do_not_become_missing_skill_penalties(self) -> None:
        jd_text = """
Backend Analyst Intern
Acme

Responsibilities
- Ability to coordinate multiple activities and manage tasks.
- Ability to prepare data summaries and presentations.
- Build REST APIs in Python and SQL.
"""
        packet = self._service().diagnose(
            resume_path=FIXTURE_ROOT / "resumes" / "arjun_backend_student.txt",
            jd_text=jd_text,
            persist=False,
        ).packet
        rendered_gaps = " ".join(gap.title + " " + gap.details for gap in packet.report.top_gaps).lower()

        self.assertIn("python", packet.job_requirements.required_skills)
        self.assertIn("rest api", packet.job_requirements.required_skills)
        self.assertTrue(packet.job_requirements.soft_requirements)
        self.assertNotIn("ability to coordinate multiple activities", packet.report.missing_skills)
        self.assertNotIn("ability to manage tasks", packet.report.missing_skills)
        self.assertNotIn("missing keyword: ability", rendered_gaps)
        self.assertNotEqual(packet.report.recommended_decision, "defer")

    def test_curated_fixture_matches_do_not_defer_or_score_below_seven(self) -> None:
        cases = [
            ("arjun_backend_student.txt", "backend_newgrad.md"),
            ("nisha_ml_newgrad.txt", "ml_platform_engineer.md"),
            ("meera_data_analyst.txt", "data_analyst_rotational.md"),
            ("rahul_fullstack_intern.txt", "ai_products_intern.md"),
        ]
        for resume_name, jd_name in cases:
            with self.subTest(resume=resume_name, jd=jd_name):
                packet = self._diagnose_fixture(resume_name, jd_name)
                self.assertGreaterEqual(packet.report.score_overall, 7.0)
                self.assertIn(packet.report.recommended_decision, {"apply_now", "apply_after_patch"})

    def test_valid_gemini_score_review_overrides_baseline(self) -> None:
        payload = {
            "overall": 8.6,
            "ats": 8.4,
            "evidence": 8.8,
            "level_fit": 8.2,
            "recommended_decision": "apply_now",
            "rationale": {
                "overall": "Strong hard-skill match with project evidence and only minor patch needs.",
                "decision": "Ready to apply because hard requirements are covered.",
            },
        }
        with (
            mock.patch.object(GeminiAugmenter, "refine_resume_facts", lambda self, resume_facts: resume_facts),
            mock.patch.object(GeminiAugmenter, "refine_job_requirements", lambda self, requirements, jd_text: requirements),
            mock.patch.object(GeminiAugmenter, "_call", return_value=payload),
        ):
            packet = self._diagnose_fixture("arjun_backend_student.txt", "backend_newgrad.md", google=True)

        self.assertEqual(packet.report.scoring_source, "gemini")
        self.assertEqual(packet.report.score_overall, 8.6)
        self.assertEqual(packet.report.recommended_decision, "apply_now")
        self.assertIn("overall", packet.report.score_rationale)

    def test_too_harsh_gemini_score_review_is_rejected_for_strong_fit(self) -> None:
        payload = {
            "overall": 4.2,
            "ats": 8.0,
            "evidence": 4.0,
            "level_fit": 8.0,
            "recommended_decision": "defer",
            "rationale": {"overall": "Too low without a hard-skill reason."},
        }
        with (
            mock.patch.object(GeminiAugmenter, "refine_resume_facts", lambda self, resume_facts: resume_facts),
            mock.patch.object(GeminiAugmenter, "refine_job_requirements", lambda self, requirements, jd_text: requirements),
            mock.patch.object(GeminiAugmenter, "_call", return_value=payload),
        ):
            packet = self._diagnose_fixture("arjun_backend_student.txt", "backend_newgrad.md", google=True)

        self.assertEqual(packet.report.scoring_source, "deterministic")
        self.assertNotIn(packet.report.recommended_decision, {"defer", "not_fit"})

    def test_old_packets_default_new_score_metadata(self) -> None:
        packet = self._diagnose_fixture("arjun_backend_student.txt", "backend_newgrad.md")
        payload = packet.to_dict()
        payload["job_requirements"].pop("soft_requirements", None)
        payload["report"].pop("scoring_source", None)
        payload["report"].pop("score_rationale", None)

        loaded = SavedJobPacket.from_dict(payload)

        self.assertEqual(loaded.job_requirements.soft_requirements, [])
        self.assertEqual(loaded.report.scoring_source, "deterministic")
        self.assertEqual(loaded.report.score_rationale, {})


if __name__ == "__main__":
    unittest.main()
