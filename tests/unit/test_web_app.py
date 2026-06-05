from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.web_app import COOKIE_NAME, GOOGLE_STATE_COOKIE_NAME, SESSION_COOKIE_NAME, create_app
from job_rejection_agent.agents.root_agent import AgentRuntime
from job_rejection_agent.config import Settings
from job_rejection_agent.domain import ImprovementRun


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


class FakeOptimizer:
    def __init__(self) -> None:
        self.record_successful_diagnosis = mock.Mock(return_value={})
        self.latest_snapshot = mock.Mock(
            return_value={
                "status": "idle",
                "auto_enabled": True,
                "auto_interval": 5,
                "successful_diagnosis_count": 0,
                "last_auto_run_diagnosis_count": 0,
                "diagnoses_since_last_run": 0,
                "diagnoses_until_next_run": 5,
                "last_started_at": "",
                "last_completed_at": "",
                "last_error": "",
                "last_trigger_packet_id": "",
                "last_trigger_session_id": "",
                "candidate_prompt": "",
                "improvement_run": None,
            }
        )


class WebAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_root = Path(self.temp_dir.name)
        self.settings = Settings(
            google_api_key=None,
            google_oauth_client_id="google-client-id",
            google_oauth_client_secret="google-client-secret",
            google_oauth_redirect_uri="http://testserver/auth/google/callback",
            phoenix_api_key=None,
            local_storage_path=temp_root / "packets.json",
            local_user_storage_path=temp_root / "users.json",
            session_db_url=f"sqlite+aiosqlite:///{temp_root / 'sessions.db'}",
            app_secret_key="test-secret",
        )
        self.runtime = AgentRuntime(settings=self.settings)
        self.optimizer = FakeOptimizer()
        self.app = create_app(settings=self.settings, runtime=self.runtime, optimizer=self.optimizer)
        self.client = TestClient(self.app)
        self.user_id = "guest-webapp"
        self.client.cookies.set(COOKIE_NAME, self.user_id)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _resume_fixture(self) -> Path:
        return FIXTURE_ROOT / "resumes" / "arjun_backend_student.txt"

    def _jd_fixture_text(self) -> str:
        return (FIXTURE_ROOT / "jds" / "backend_newgrad.md").read_text(encoding="utf-8")

    def test_home_page_renders(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Diagnose the gap.", response.text)
        self.assertIn("Run Diagnosis", response.text)
        self.assertIn("/static/css/app.css", response.text)
        self.assertNotIn("http://testserver/static/", response.text)
        self.assertNotIn("cdn.tailwindcss.com", response.text)
        self.assertNotIn("code.iconify.design", response.text)
        self.assertNotIn("api.fontshare.com", response.text)
        self.assertNotIn("unpkg.com/mammoth", response.text)

    def test_static_css_route_serves_local_bundle(self) -> None:
        response = self.client.get("/static/css/app.css")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/css", response.headers["content-type"])

    def test_login_page_renders(self) -> None:
        response = self.client.get("/login")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Save every diagnosis to your account.", response.text)
        self.assertIn("Sign In", response.text)
        self.assertIn("Continue with Google", response.text)

    def test_google_oauth_start_sets_state_cookie_and_redirects(self) -> None:
        response = self.client.get("/auth/google/start?next_path=/history", follow_redirects=False)

        self.assertEqual(response.status_code, 303)
        self.assertIn("accounts.google.com", response.headers["location"])
        self.assertIn(GOOGLE_STATE_COOKIE_NAME, response.headers.get("set-cookie", ""))

    def test_diagnose_validation_error_stays_in_branded_ui(self) -> None:
        response = self.client.post(
            "/diagnose",
            data={"jd_text": "", "rejection_notes": "Need stronger impact metrics.", "demo_case": ""},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("The diagnosis did not run.", response.text)
        self.assertIn("Upload a resume or select a demo case", response.text)
        self.assertIn("Need stronger impact metrics.", response.text)

    def test_diagnose_upload_redirects_to_result_packet(self) -> None:
        response = self.client.post(
            "/diagnose",
            data={"jd_text": self._jd_fixture_text(), "rejection_notes": "Need stronger API ownership evidence."},
            files={"resume": ("arjun_backend_student.txt", self._resume_fixture().read_bytes(), "text/plain")},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertIn("/?packet_id=", response.headers["location"])

        result_page = self.client.get(response.headers["location"])
        self.assertEqual(result_page.status_code, 200)
        self.assertIn("Analysis Complete", result_page.text)
        self.assertIn("arjun_backend_student.txt", result_page.text)
        self.optimizer.record_successful_diagnosis.assert_called_once()

    def test_diagnose_failure_keeps_resume_preview_visible(self) -> None:
        with mock.patch.object(
            self.runtime,
            "run_diagnostic_async",
            new=mock.AsyncMock(side_effect=RuntimeError("503 UNAVAILABLE. Model under high demand.")),
        ):
            response = self.client.post(
                "/diagnose",
                data={"jd_text": self._jd_fixture_text(), "rejection_notes": "Need stronger API ownership evidence."},
                files={"resume": ("arjun_backend_student.txt", self._resume_fixture().read_bytes(), "text/plain")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("The diagnosis did not run.", response.text)
        self.assertIn("503 UNAVAILABLE", response.text)
        self.assertIn("Resume Preview", response.text)
        self.assertIn("arjun_backend_student.txt", response.text)
        self.assertIn("Computer engineering student", response.text)

    def test_patch_route_rejects_foreign_packet(self) -> None:
        packet = self.runtime.service.diagnose(
            resume_path=self._resume_fixture(),
            jd_text=self._jd_fixture_text(),
            user_id=self.user_id,
            session_id="session-1",
            persist=True,
        ).packet

        foreign_client = TestClient(self.app)
        foreign_client.cookies.set(COOKIE_NAME, "guest-other")
        response = foreign_client.get(f"/patch/{packet.packet_id}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("could not be found for this session", response.text)

    def test_history_page_shows_saved_resume_name(self) -> None:
        packet = self.runtime.service.diagnose(
            resume_path=self._resume_fixture(),
            jd_text=self._jd_fixture_text(),
            user_id=self.user_id,
            session_id="session-2",
            persist=True,
        ).packet

        response = self.client.get(f"/history?packet_id={packet.packet_id}")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Saved Analyses", response.text)
        self.assertIn(packet.resume_name, response.text)

    def test_rewritten_resume_page_and_exports_render(self) -> None:
        packet = self.runtime.service.diagnose(
            resume_path=self._resume_fixture(),
            jd_text=self._jd_fixture_text(),
            user_id=self.user_id,
            session_id="session-resume",
            persist=True,
        ).packet

        page = self.client.get(f"/resume/{packet.packet_id}")
        self.assertEqual(page.status_code, 200)
        self.assertIn("Evidence-Backed Resume Draft", page.text)

        docx_response = self.client.get(f"/resume/{packet.packet_id}/export.docx")
        self.assertEqual(docx_response.status_code, 200)
        self.assertEqual(
            docx_response.headers["content-type"],
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

        pdf_response = self.client.get(f"/resume/{packet.packet_id}/export.pdf")
        self.assertEqual(pdf_response.status_code, 200)
        self.assertEqual(pdf_response.headers["content-type"], "application/pdf")

    def test_interview_flow_start_and_answer(self) -> None:
        packet = self.runtime.service.diagnose(
            resume_path=self._resume_fixture(),
            jd_text=self._jd_fixture_text(),
            user_id=self.user_id,
            session_id="session-interview",
            persist=True,
        ).packet

        start = self.client.post(f"/interview/{packet.packet_id}/start", follow_redirects=False)
        self.assertEqual(start.status_code, 303)
        self.assertIn("/interview/", start.headers["location"])
        session_id = start.headers["location"].split("session_id=")[-1]

        answer = self.client.post(
            f"/interview/{packet.packet_id}/answer",
            data={
                "session_id": session_id,
                "answer": "I built backend APIs for a student project and improved response time by 25% using FastAPI.",
            },
            follow_redirects=False,
        )
        self.assertEqual(answer.status_code, 303)

        page = self.client.get(answer.headers["location"])
        self.assertEqual(page.status_code, 200)
        self.assertIn("Transcript and Feedback", page.text)
        self.assertIn("Coach Feedback", page.text)
        self.assertIn("What worked", page.text)
        self.assertIn("Improve next", page.text)
        self.assertIn("Evidence to prepare", page.text)
        self.assertIn("Answer structure", page.text)
        self.assertNotIn("Suggested evidence to mention next time", page.text)

    def test_compare_flow_creates_comparison_results(self) -> None:
        response = self.client.post(
            "/compare",
            data={
                "rejection_notes": "Need stronger user-facing evidence.",
                "jd_1": self._jd_fixture_text(),
                "jd_2": (FIXTURE_ROOT / "jds" / "ai_products_intern.md").read_text(encoding="utf-8"),
            },
            files={"resume": ("arjun_backend_student.txt", self._resume_fixture().read_bytes(), "text/plain")},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertIn("/compare/", response.headers["location"])

        page = self.client.get(response.headers["location"])
        self.assertEqual(page.status_code, 200)
        self.assertIn("Multi-JD Comparison", page.text)
        self.assertIn("Open Analysis", page.text)

    def test_signup_migrates_guest_history_and_sets_session_cookie(self) -> None:
        packet = self.runtime.service.diagnose(
            resume_path=self._resume_fixture(),
            jd_text=self._jd_fixture_text(),
            user_id=self.user_id,
            session_id="session-3",
            persist=True,
        ).packet

        response = self.client.post(
            "/signup",
            data={"email": "jayesh@example.com", "password": "supersecure123", "next_path": "/history"},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/history")
        self.assertIn(SESSION_COOKIE_NAME, response.headers.get("set-cookie", ""))

        user = self.app.state.auth_service.user_repository.load_by_email("jayesh@example.com")
        self.assertIsNotNone(user)
        self.assertEqual(self.runtime.service.tracker.list_entries(self.user_id), [])
        migrated = self.runtime.service.tracker.get(packet.packet_id)
        self.assertIsNotNone(migrated)
        self.assertEqual(migrated.user_id, user.user_id)

        history = self.client.get("/history")
        self.assertEqual(history.status_code, 200)
        self.assertIn(packet.resume_name, history.text)
        self.assertNotIn("This history belongs to a guest session.", history.text)

    def test_google_oauth_callback_sets_session_cookie(self) -> None:
        response = self.client.get("/auth/google/start?next_path=/history", follow_redirects=False)
        state_cookie = self.client.cookies.get(GOOGLE_STATE_COOKIE_NAME)
        self.assertTrue(state_cookie)

        fake_user = self.app.state.auth_service.register(email="google@example.com", password="password123")
        with mock.patch.object(
            self.app.state.auth_service,
            "authenticate_google_code",
            new=mock.AsyncMock(return_value=fake_user),
        ):
            callback = self.client.get(
                f"/auth/google/callback?state={state_cookie}&code=fake-code",
                follow_redirects=False,
            )

        self.assertEqual(callback.status_code, 303)
        self.assertEqual(callback.headers["location"], "/history")
        self.assertIn(SESSION_COOKIE_NAME, callback.headers.get("set-cookie", ""))

    def test_settings_page_describes_automatic_improvement_without_manual_button(self) -> None:
        self.optimizer.latest_snapshot.return_value = {
            "status": "idle",
            "auto_enabled": True,
            "auto_interval": 5,
            "successful_diagnosis_count": 7,
            "last_auto_run_diagnosis_count": 5,
            "diagnoses_since_last_run": 2,
            "diagnoses_until_next_run": 3,
            "last_started_at": "",
            "last_completed_at": "2026-05-27T09:30:00+00:00",
            "last_error": "",
            "last_trigger_packet_id": "packet-1",
            "last_trigger_session_id": "session-1",
            "candidate_prompt": "Prompt preview",
            "improvement_run": ImprovementRun(
                run_id="run-1",
                baseline_prompt_version="baseline-v1",
                candidate_prompt_version="baseline-v1-candidate",
                source_span_ids=["span-1"],
                baseline_scores={"composite_score": 0.7},
                candidate_scores={"composite_score": 0.76},
                promoted=True,
                analysis="Candidate improved the held-out suite without regressions.",
            ),
        }

        response = self.client.get("/settings")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Every 5 successful diagnoses", response.text)
        self.assertIn("Next Auto Run", response.text)
        self.assertNotIn("Run Improvement Loop", response.text)


if __name__ == "__main__":
    unittest.main()
