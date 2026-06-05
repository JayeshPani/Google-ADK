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
from job_rejection_agent.domain import JobRequirements, MultiJDComparison, ResumeFacts
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
        service = DiagnosticService(settings=Settings(google_api_key=None, phoenix_api_key=None))
        comparison = service.compare_job_descriptions(
            resume_path=ROOT / "fixtures" / "resumes" / "rahul_fullstack_intern.txt",
            jd_texts=[
                (ROOT / "fixtures" / "jds" / "ai_products_intern.md").read_text(encoding="utf-8"),
                (ROOT / "fixtures" / "jds" / "backend_newgrad.md").read_text(encoding="utf-8"),
            ],
            user_id="compare-user",
        )

        self.assertEqual(len(comparison.rows), 2)
        self.assertEqual(comparison.rows[0].rank, 1)
        self.assertTrue(comparison.best_packet_id)
        self.assertTrue(comparison.summary)
        self.assertTrue(comparison.shared_resume_strategy)
        self.assertTrue(comparison.rows[0].rank_reason)
        self.assertTrue(comparison.rows[0].strengths)
        self.assertTrue(comparison.rows[0].risks)
        self.assertTrue(comparison.rows[0].next_action)
        self.assertGreaterEqual(comparison.rows[0].hard_skill_coverage, 0)
        self.assertLessEqual(comparison.rows[0].hard_skill_coverage, 1)
        saved_packet = service.tracker.get(comparison.rows[0].packet_id)
        self.assertIsNotNone(saved_packet)
        self.assertIsNone(saved_packet.report.rewritten_resume)
        self.assertEqual(saved_packet.report.exact_edits, [])

    def test_compare_reuses_cached_resume_parse(self) -> None:
        service = DiagnosticService(settings=Settings(google_api_key=None, phoenix_api_key=None))
        resume_path = ROOT / "fixtures" / "resumes" / "rahul_fullstack_intern.txt"
        preview = service.preview_cached_resume(
            file_name="rahul_fullstack_intern.txt",
            content=resume_path.read_bytes(),
        )

        with patch("job_rejection_agent.services.diagnostic_service.parse_resume_file", side_effect=AssertionError("cache miss")):
            comparison = service.compare_job_descriptions(
                resume_path=resume_path,
                resume_cache_key=preview["resume_cache_key"],
                resume_name="rahul_fullstack_intern.txt",
                jd_texts=[
                    (ROOT / "fixtures" / "jds" / "ai_products_intern.md").read_text(encoding="utf-8"),
                    (ROOT / "fixtures" / "jds" / "backend_newgrad.md").read_text(encoding="utf-8"),
                ],
                user_id="cached-compare-user",
            )

        self.assertEqual(len(comparison.rows), 2)
        self.assertEqual(comparison.resume_name, "rahul_fullstack_intern.txt")

    def test_compare_soft_requirements_do_not_become_missing_hard_skills(self) -> None:
        service = DiagnosticService(settings=Settings(google_api_key=None, phoenix_api_key=None))
        soft_jd = """
        Operations Intern
        Responsibilities include ability to coordinate multiple activities, manage tasks, and prepare presentations.
        Preferred exposure to Python.
        """
        comparison = service.compare_job_descriptions(
            resume_path=ROOT / "fixtures" / "resumes" / "arjun_backend_student.txt",
            jd_texts=[soft_jd, soft_jd.replace("Operations", "Program")],
            user_id="soft-compare-user",
        )

        missing = " ".join(skill.lower() for row in comparison.rows for skill in row.missing_hard_skills)
        self.assertNotIn("ability to coordinate multiple activities", missing)
        self.assertNotIn("manage tasks", missing)
        self.assertNotIn("prepare presentations", missing)

    def test_old_comparison_payload_loads_with_defaults(self) -> None:
        comparison = MultiJDComparison.from_dict(
            {
                "comparison_id": "comparison-old",
                "user_id": "user-old",
                "resume_name": "resume.txt",
                "rows": [
                    {
                        "packet_id": "packet-old",
                        "role_title": "Backend Intern",
                        "company_name": "ExampleCo",
                        "score_overall": 7.2,
                        "score_ats": 8.0,
                        "score_evidence": 6.8,
                        "score_level_fit": 7.0,
                        "recommended_decision": "apply_after_patch",
                        "top_gap_title": "Missing keyword: SQL",
                    }
                ],
            }
        )

        self.assertEqual(comparison.best_packet_id, "")
        self.assertEqual(comparison.common_missing_skills, [])
        self.assertEqual(comparison.rows[0].rank, 0)
        self.assertEqual(comparison.rows[0].strengths, [])

    def test_resume_preparse_cache_is_reused_by_quick_diagnosis(self) -> None:
        service = DiagnosticService(settings=Settings(google_api_key=None, phoenix_api_key=None))
        preview = service.preview_cached_resume(
            file_name="arjun_backend_student.txt",
            content=(ROOT / "fixtures" / "resumes" / "arjun_backend_student.txt").read_bytes(),
        )

        with patch("job_rejection_agent.services.diagnostic_service.parse_resume_file", side_effect=AssertionError("cache miss")):
            result = service.diagnose_quick(
                resume_path=ROOT / "fixtures" / "resumes" / "arjun_backend_student.txt",
                resume_cache_key=preview["resume_cache_key"],
                jd_text=(ROOT / "fixtures" / "jds" / "backend_newgrad.md").read_text(encoding="utf-8"),
                persist=False,
            )

        self.assertGreater(result.packet.report.score_overall, 0)
        self.assertEqual(result.timings["resume_cache_hit"], 1.0)

    def test_quick_then_full_diagnosis_updates_same_packet(self) -> None:
        service = DiagnosticService(settings=Settings(google_api_key=None, phoenix_api_key=None))
        quick = service.diagnose_quick(
            resume_path=ROOT / "fixtures" / "resumes" / "arjun_backend_student.txt",
            jd_text=(ROOT / "fixtures" / "jds" / "backend_newgrad.md").read_text(encoding="utf-8"),
            user_id="staged-user",
            persist=True,
        )

        self.assertGreater(quick.packet.report.score_overall, 0)
        self.assertEqual(quick.packet.report.exact_edits, [])
        self.assertIsNone(quick.packet.report.rewritten_resume)

        full = service.complete_diagnosis(packet_id=quick.packet.packet_id, user_id="staged-user")
        self.assertIsNotNone(full)
        assert full is not None
        self.assertEqual(full.packet.packet_id, quick.packet.packet_id)
        self.assertGreater(len(full.packet.report.exact_edits), 0)
        self.assertGreaterEqual(len(full.packet.report.action_plan), 5)
        self.assertIsNotNone(full.packet.report.rewritten_resume)

    def test_gemini_profile_refinement_is_batched_before_score_review(self) -> None:
        service = DiagnosticService(settings=Settings(google_api_key="test-key", phoenix_api_key=None))
        with patch.object(GeminiAugmenter, "_call", return_value={}) as call:
            service.diagnose_quick(
                resume_path=ROOT / "fixtures" / "resumes" / "arjun_backend_student.txt",
                jd_text=(ROOT / "fixtures" / "jds" / "backend_newgrad.md").read_text(encoding="utf-8"),
                persist=False,
            )

        self.assertLessEqual(call.call_count, 2)

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
