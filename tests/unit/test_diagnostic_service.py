from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent.config import Settings
from job_rejection_agent.domain import JobRequirements, ResumeFacts
from job_rejection_agent.services import DiagnosticService
from job_rejection_agent.services.diagnostic_service import GeminiAugmenter


ROOT = Path(__file__).resolve().parents[1]


class DiagnosticServiceTests(unittest.TestCase):
    def test_diagnosis_generates_edits_and_plan(self) -> None:
        service = DiagnosticService()
        result = service.diagnose(
            resume_path=ROOT / "fixtures" / "resumes" / "rahul_fullstack_intern.txt",
            jd_text=(ROOT / "fixtures" / "jds" / "ai_products_intern.md").read_text(encoding="utf-8"),
            rejection_notes="Role wanted more visible LLM work.",
            persist=False,
        )
        self.assertGreater(len(result.packet.report.exact_edits), 0)
        self.assertGreaterEqual(len(result.packet.report.action_plan), 5)
        self.assertIn(result.packet.report.recommended_decision, {"apply_now", "apply_after_patch", "defer", "not_fit"})
        self.assertTrue(result.packet.report.ats_checks)
        self.assertIsNotNone(result.packet.report.rewritten_resume)
        self.assertIn("Skills", result.packet.report.rewritten_resume.skills.title)

    def test_interview_session_flow_is_persisted_in_packet(self) -> None:
        service = DiagnosticService()
        packet = service.diagnose(
            resume_path=ROOT / "fixtures" / "resumes" / "arjun_backend_student.txt",
            jd_text=(ROOT / "fixtures" / "jds" / "backend_newgrad.md").read_text(encoding="utf-8"),
            user_id="user-1",
            session_id="session-a",
            persist=True,
        ).packet

        created = service.create_interview_session(packet_id=packet.packet_id, user_id="user-1")
        self.assertIsNotNone(created)
        _, session = created
        self.assertEqual(session.status, "in_progress")

        updated = service.submit_interview_answer(
            packet_id=packet.packet_id,
            session_id=session.session_id,
            user_id="user-1",
            answer="I built a FastAPI backend for CampusCart and improved checkout latency by 28% for real users.",
        )
        self.assertIsNotNone(updated)
        packet_after, updated_session = updated
        self.assertEqual(len(updated_session.turns), 1)
        self.assertGreater(updated_session.turns[0].feedback.overall_score, 0)
        self.assertEqual(packet_after.interview_sessions[0].session_id, session.session_id)

    def test_compare_job_descriptions_creates_ranked_bundle(self) -> None:
        service = DiagnosticService()
        comparison = service.compare_job_descriptions(
            resume_path=ROOT / "fixtures" / "resumes" / "rahul_fullstack_intern.txt",
            jd_texts=[
                (ROOT / "fixtures" / "jds" / "ai_products_intern.md").read_text(encoding="utf-8"),
                (ROOT / "fixtures" / "jds" / "backend_newgrad.md").read_text(encoding="utf-8"),
            ],
            user_id="compare-user",
        )

        self.assertEqual(len(comparison.rows), 2)
        self.assertGreaterEqual(comparison.rows[0].score_overall, comparison.rows[1].score_overall)

    def test_gemini_resume_augmentation_handles_dict_payloads(self) -> None:
        augmenter = GeminiAugmenter(Settings())
        resume_facts = ResumeFacts(
            raw_text="",
            normalized_text="Built TrustCall and CampusCart with Python.",
            summary="Existing summary",
            skills=["python"],
            projects=["CampusCart"],
        )
        payload = {
            "summary": {"text": "Builder focused on AI products."},
            "skills": [{"name": "Docker"}, {"skill": "Python"}, ["FastAPI", {"label": "RAG"}]],
            "projects": [{"name": "TrustCall", "details": "Voice spoof detection"}, {"title": "CampusCart"}],
        }

        with patch.object(GeminiAugmenter, "_call", return_value=payload):
            refined = augmenter.refine_resume_facts(resume_facts)

        self.assertEqual(refined.summary, "Builder focused on AI products.")
        self.assertEqual(refined.skills, ["docker", "fastapi", "python", "rag"])
        self.assertEqual(refined.projects, ["CampusCart", "TrustCall"])

    def test_gemini_resume_augmentation_normalizes_existing_projects(self) -> None:
        augmenter = GeminiAugmenter(Settings())
        resume_facts = ResumeFacts(
            raw_text="",
            normalized_text="Built TrustCall and CampusCart with Python.",
            summary="Existing summary",
            skills=["python", {"name": "Docker"}],  # type: ignore[list-item]
            projects=[{"title": "CampusCart"}, "TrustCall"],  # type: ignore[list-item]
        )
        payload = {
            "skills": [],
            "projects": [{"name": "CampusCart"}, {"name": "MockMate"}],
        }

        with patch.object(GeminiAugmenter, "_call", return_value=payload):
            refined = augmenter.refine_resume_facts(resume_facts)

        self.assertEqual(refined.skills, ["docker", "python"])
        self.assertEqual(refined.projects, ["CampusCart", "TrustCall", "MockMate"])

    def test_gemini_job_augmentation_handles_dict_payloads(self) -> None:
        augmenter = GeminiAugmenter(Settings())
        requirements = JobRequirements(
            role_title="AI Intern",
            company_name="Demo",
            role_summary="Existing role summary",
            required_skills=["python"],
            preferred_skills=["sql"],
            keywords=["ml"],
        )
        payload = {
            "role_summary": {"text": "Build and evaluate agent workflows."},
            "required_skills": [{"name": "Kubernetes"}, {"label": "Python"}],
            "preferred_skills": {"preferred": [{"label": "RAG"}, "Vertex AI"]},
            "keywords": [["ADK"], {"value": "Gemini"}],
            "experience_level": {"label": "entry"},
        }

        with patch.object(GeminiAugmenter, "_call", return_value=payload):
            refined = augmenter.refine_job_requirements(requirements, "Build and evaluate agent workflows.")

        self.assertEqual(refined.role_summary, "Build and evaluate agent workflows.")
        self.assertEqual(refined.required_skills, ["kubernetes", "python"])
        self.assertEqual(refined.preferred_skills, ["rag", "sql", "vertex ai"])
        self.assertEqual(refined.keywords, ["adk", "gemini", "ml"])
        self.assertEqual(refined.experience_level, "entry")


if __name__ == "__main__":
    unittest.main()
