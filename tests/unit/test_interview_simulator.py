from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent.coaching.interview_simulator import (  # noqa: E402
    _safe_evidence_prompts,
    start_interview_session,
    submit_interview_answer,
)
from job_rejection_agent.config import Settings  # noqa: E402
from job_rejection_agent.domain import InterviewFeedback, SavedJobPacket  # noqa: E402
from job_rejection_agent.services import DiagnosticService  # noqa: E402


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


class InterviewSimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        service = DiagnosticService(settings=Settings(google_api_key=None, phoenix_api_key=None))
        cls.base_packet = service.diagnose(
            resume_path=FIXTURE_ROOT / "resumes" / "arjun_backend_student.txt",
            jd_text=(FIXTURE_ROOT / "jds" / "backend_newgrad.md").read_text(encoding="utf-8"),
            user_id="interview-user",
            session_id="interview-session",
            persist=False,
        ).packet

    def setUp(self) -> None:
        self.packet = SavedJobPacket.from_dict(self.base_packet.to_dict())

    def test_start_session_uses_gemini_question_as_single_adaptive_prompt(self) -> None:
        question = (
            "For the Backend Software Engineer role, how would you explain your Expense Split Service "
            "API ownership and measurable backend result?"
        )
        with mock.patch(
            "job_rejection_agent.coaching.interview_simulator._call_gemini_json",
            return_value={"question": question},
        ):
            session = start_interview_session(self.packet, settings=Settings(google_api_key="test-key"))

        self.assertEqual(session.questions, [question])
        self.assertNotEqual(session.questions, self.packet.report.interview_questions[:5])

    def test_invalid_gemini_question_falls_back_without_old_static_prompt(self) -> None:
        with mock.patch(
            "job_rejection_agent.coaching.interview_simulator._call_gemini_json",
            return_value={"question": "Tell me about yourself."},
        ):
            session = start_interview_session(self.packet, settings=Settings(google_api_key="test-key"))

        self.assertEqual(len(session.questions), 1)
        self.assertNotEqual(session.questions[0], "Tell me about yourself.")
        self.assertNotEqual(session.questions[0], self.packet.report.interview_questions[0])

    def test_sanitized_evidence_excludes_pii_dates_and_diagnostic_text(self) -> None:
        packet = self.packet
        packet.resume_facts.metrics = [
            "Phone-Alt +91-9810315881 | Envelope jayeshpani14@gmail.com | LINKEDIN jayesh-pani",
            "2023 – 2028 (Expected)",
            "Improved API response time by 25% using FastAPI caching.",
        ]
        packet.report.top_gaps[0].supporting_evidence = [
            "The resume reads like a student, while the JD reads closer to Intern scope.",
        ]

        prompts = _safe_evidence_prompts(packet)
        rendered = " ".join(prompts).lower()

        self.assertIn("25%", rendered)
        self.assertNotIn("9810315881", rendered)
        self.assertNotIn("jayeshpani14", rendered)
        self.assertNotIn("linkedin", rendered)
        self.assertNotIn("2023", rendered)
        self.assertNotIn("resume reads like a student", rendered)

    def test_feedback_has_structured_fields_and_safe_evidence(self) -> None:
        packet = self.packet
        packet.resume_facts.metrics = [
            "Phone-Alt +91-9810315881 | Envelope jayeshpani14@gmail.com",
            "Reduced response time by 25% on a FastAPI endpoint.",
        ]
        session = start_interview_session(packet, settings=Settings(google_api_key=None))
        updated = submit_interview_answer(
            packet,
            session,
            "I built REST APIs for Expense Split Service and reduced response time by 25% using FastAPI.",
            settings=Settings(google_api_key=None),
        )
        feedback = updated.turns[0].feedback
        rendered = " ".join(
            [
                feedback.feedback,
                *feedback.strengths,
                *feedback.improvements,
                *feedback.evidence_prompts,
                feedback.answer_structure,
            ]
        ).lower()

        self.assertTrue(feedback.strengths)
        self.assertTrue(feedback.improvements)
        self.assertTrue(feedback.evidence_prompts)
        self.assertTrue(feedback.answer_structure)
        self.assertNotIn("9810315881", rendered)
        self.assertNotIn("jayeshpani14", rendered)

    def test_strong_structured_answer_scores_high_in_fallback(self) -> None:
        session = start_interview_session(self.packet, settings=Settings(google_api_key=None))
        updated = submit_interview_answer(
            self.packet,
            session,
            (
                "On Expense Split Service, I designed the REST API endpoints in Java and SQL, wrote validation "
                "around split creation, and reduced manual reconciliation by 30%. The situation was classmates "
                "tracking shared costs manually; my task was to own the backend flow; I built the service layer, "
                "tested edge cases, and tied the result to the role's API ownership needs."
            ),
            settings=Settings(google_api_key=None),
        )

        self.assertGreaterEqual(updated.turns[0].feedback.overall_score, 8.0)
        self.assertGreaterEqual(updated.turns[0].feedback.evidence_score, 8.0)

    def test_vague_short_answer_scores_low_even_with_keyword(self) -> None:
        session = start_interview_session(self.packet, settings=Settings(google_api_key=None))
        updated = submit_interview_answer(
            self.packet,
            session,
            "I used Python and APIs.",
            settings=Settings(google_api_key=None),
        )

        self.assertLessEqual(updated.turns[0].feedback.overall_score, 5.1)

    def test_gemini_numeric_feedback_scores_are_used_when_valid(self) -> None:
        session = start_interview_session(self.packet, settings=Settings(google_api_key=None))
        payload = {
            "overall_score": 9.1,
            "evidence_score": 9.2,
            "clarity_score": 8.8,
            "relevance_score": 9.0,
            "gap_score": 9.1,
            "feedback": "Strong project-backed answer with a clear metric.",
            "strengths": ["Concrete project evidence."],
            "improvements": ["Keep the closing role-fit sentence crisp."],
            "evidence_prompts": ["Prepare the Expense Split Service metric and API ownership story."],
            "answer_structure": "STAR: situation, task, action, result, role fit.",
        }

        with mock.patch(
            "job_rejection_agent.coaching.interview_simulator._call_gemini_json",
            return_value=payload,
        ):
            updated = submit_interview_answer(
                self.packet,
                session,
                (
                    "I built the Expense Split Service REST API, owned validation and SQL persistence, "
                    "and reduced manual reconciliation by 30% for a student project team."
                ),
                settings=Settings(google_api_key="test-key"),
            )

        feedback = updated.turns[0].feedback
        self.assertEqual(feedback.overall_score, 9.1)
        self.assertEqual(feedback.evidence_score, 9.2)
        self.assertEqual(feedback.strengths, ["Concrete project evidence."])

    def test_old_feedback_payloads_default_new_fields(self) -> None:
        feedback = InterviewFeedback.from_dict(
            {
                "overall_score": 6.0,
                "evidence_score": 6.0,
                "clarity_score": 6.0,
                "relevance_score": 6.0,
                "gap_score": 6.0,
                "feedback": "Old feedback.",
                "suggested_evidence": ["Old safe suggestion."],
            }
        )

        self.assertEqual(feedback.evidence_prompts, ["Old safe suggestion."])
        self.assertEqual(feedback.strengths, [])
        self.assertEqual(feedback.improvements, [])
        self.assertEqual(feedback.answer_structure, "")


if __name__ == "__main__":
    unittest.main()
