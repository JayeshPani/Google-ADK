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

from app.web_app import COOKIE_NAME, DEMO_CASES, GOOGLE_STATE_COOKIE_NAME, SESSION_COOKIE_NAME, create_app
from job_rejection_agent.agents.root_agent import AgentRuntime
from job_rejection_agent.config import Settings


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
        self.user = self.app.state.auth_service.register(email="test@example.com", password="password123")
        self.user_id = self.user.user_id
        session_token = self.app.state.auth_service.create_session_token(self.user, ttl_seconds=3600)
        self.client.cookies.set(SESSION_COOKIE_NAME, session_token)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _resume_fixture(self) -> Path:
        return FIXTURE_ROOT / "resumes" / "arjun_backend_student.txt"

    def _jd_fixture_text(self) -> str:
        return (FIXTURE_ROOT / "jds" / "backend_newgrad.md").read_text(encoding="utf-8")

    def test_home_page_renders(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Know what to fix before the next application.", response.text)
        self.assertIn("Get Started", response.text)
        self.assertIn('/login?next_path=/diagnose', response.text)
        self.assertIn("Resume diagnosis", response.text)
        self.assertIn("JD comparison", response.text)
        self.assertIn("/static/css/app.css", response.text)
        self.assertNotIn("http://testserver/static/", response.text)
        self.assertNotIn("cdn.tailwindcss.com", response.text)
        self.assertNotIn("code.iconify.design", response.text)
        self.assertNotIn("api.fontshare.com", response.text)
        self.assertIn("data-theme-toggle", response.text)
        self.assertIn("refine_theme", response.text)
        self.assertIn("theme-icon-sun", response.text)
        self.assertIn("theme-icon-moon", response.text)
        self.assertNotIn('href="/settings"', response.text)
        self.assertNotIn(">Account</a>", response.text)
        self.assertNotIn("Guest ", response.text)
        self.assertIn("test@example.com", response.text)

    def test_diagnose_page_renders_for_signed_in_user(self) -> None:
        response = self.client.get("/diagnose")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Diagnose the gap.", response.text)
        self.assertIn("Run Diagnosis", response.text)
        self.assertIn("/static/css/app.css", response.text)
        self.assertNotIn("http://testserver/static/", response.text)
        self.assertNotIn("cdn.tailwindcss.com", response.text)
        self.assertNotIn("code.iconify.design", response.text)
        self.assertNotIn("api.fontshare.com", response.text)
        self.assertNotIn("unpkg.com/mammoth", response.text)
        self.assertIn("data-theme-toggle", response.text)
        self.assertIn("refine_theme", response.text)
        self.assertIn("theme-icon-sun", response.text)
        self.assertIn("theme-icon-moon", response.text)
        self.assertNotIn('href="/settings"', response.text)
        self.assertNotIn(">Account</a>", response.text)
        self.assertNotIn("Guest ", response.text)
        self.assertIn("test@example.com", response.text)

    def test_static_css_route_serves_local_bundle(self) -> None:
        response = self.client.get("/static/css/app.css")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/css", response.headers["content-type"])

    def test_login_page_renders(self) -> None:
        anonymous_client = TestClient(self.app)
        response = anonymous_client.get("/login")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Sign in, then let", response.text)
        self.assertIn("Pick Up Easily", response.text)
        self.assertIn("Sign In", response.text)
        self.assertIn("Continue with Google", response.text)
        self.assertNotIn("Guest ", response.text)

    def test_unauthenticated_users_start_on_login_before_diagnosis(self) -> None:
        anonymous_client = TestClient(self.app)

        home = anonymous_client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn("Know what to fix before the next application.", home.text)

        response = anonymous_client.get("/diagnose", follow_redirects=False)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/login?next_path=%2Fdiagnose")

    def test_compare_page_includes_resume_preview_controls(self) -> None:
        response = self.client.get("/compare")

        self.assertEqual(response.status_code, 200)
        self.assertIn("compare-resume-cache-key", response.text)
        self.assertIn("compare-resume-preview-shell", response.text)
        self.assertIn("/resume/preview-parse", response.text)
        self.assertIn("Resume Preview", response.text)

    def test_google_oauth_start_sets_state_cookie_and_redirects(self) -> None:
        anonymous_client = TestClient(self.app)
        response = anonymous_client.get("/auth/google/start?next_path=/history", follow_redirects=False)

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
        self.assertIn("/diagnose/progress/", response.headers["location"])

        job_id = response.headers["location"].rstrip("/").split("/")[-1]
        status = self.client.get(f"/diagnose/jobs/{job_id}.json")
        self.assertEqual(status.status_code, 200)
        payload = status.json()
        self.assertIn(payload["status"], {"quick_ready", "completed"})
        self.assertTrue(payload["packet_id"])

        progress_page = self.client.get(response.headers["location"])
        self.assertEqual(progress_page.status_code, 200)
        self.assertIn("Diagnosis in progress", progress_page.text)

        result_page = self.client.get(f"/diagnose?packet_id={payload['packet_id']}&job_id={job_id}")
        self.assertEqual(result_page.status_code, 200)
        self.assertIn("Analysis Complete", result_page.text)
        self.assertIn("arjun_backend_student.txt", result_page.text)
        self.optimizer.record_successful_diagnosis.assert_called_once()

    def test_all_demo_cases_create_diagnosis_jobs(self) -> None:
        for demo_key, demo in DEMO_CASES.items():
            with self.subTest(demo_case=demo_key):
                response = self.client.post(
                    "/diagnose",
                    data={"jd_text": "", "rejection_notes": "", "demo_case": demo_key},
                    follow_redirects=False,
                )

                self.assertEqual(response.status_code, 303)
                self.assertIn("/diagnose/progress/", response.headers["location"])

                job_id = response.headers["location"].rstrip("/").split("/")[-1]
                status = self.client.get(f"/diagnose/jobs/{job_id}.json")
                self.assertEqual(status.status_code, 200)
                payload = status.json()
                self.assertIn(payload["status"], {"quick_ready", "completed"})
                self.assertTrue(payload["packet_id"])

                result_page = self.client.get(f"/diagnose?packet_id={payload['packet_id']}&job_id={job_id}")
                self.assertEqual(result_page.status_code, 200)
                self.assertIn("Analysis Complete", result_page.text)
                self.assertIn(demo["resume"], result_page.text)

    def test_resume_preview_parse_returns_cache_key(self) -> None:
        response = self.client.post(
            "/resume/preview-parse",
            files={"resume": ("arjun_backend_student.txt", self._resume_fixture().read_bytes(), "text/plain")},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["resume_cache_key"])
        self.assertIn("Computer engineering student", payload["text"])

    def test_diagnose_failure_surfaces_on_progress_endpoint(self) -> None:
        with mock.patch.object(
            self.runtime.service,
            "diagnose_quick",
            side_effect=RuntimeError("503 UNAVAILABLE. Model under high demand."),
        ):
            response = self.client.post(
                "/diagnose",
                data={"jd_text": self._jd_fixture_text(), "rejection_notes": "Need stronger API ownership evidence."},
                files={"resume": ("arjun_backend_student.txt", self._resume_fixture().read_bytes(), "text/plain")},
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 303)
        job_id = response.headers["location"].rstrip("/").split("/")[-1]
        status = self.client.get(f"/diagnose/jobs/{job_id}.json")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["status"], "failed")
        self.assertIn("503 UNAVAILABLE", status.json()["error"])

    def test_patch_route_rejects_foreign_packet(self) -> None:
        packet = self.runtime.service.diagnose(
            resume_path=self._resume_fixture(),
            jd_text=self._jd_fixture_text(),
            user_id=self.user_id,
            session_id="session-1",
            persist=True,
        ).packet

        foreign_client = TestClient(self.app)
        foreign_user = self.app.state.auth_service.register(email="other@example.com", password="password123")
        foreign_token = self.app.state.auth_service.create_session_token(foreign_user, ttl_seconds=3600)
        foreign_client.cookies.set(SESSION_COOKIE_NAME, foreign_token)
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
        self.assertIn("Apply Priority", page.text)
        self.assertIn("Shared Resume Strategy", page.text)
        self.assertIn("Best First Application", page.text)
        self.assertIn("What makes this viable", page.text)
        self.assertIn("What can block it", page.text)
        self.assertIn("Open Analysis", page.text)
        self.assertIn("Finish Full Report", page.text)
        self.assertNotIn("Rewritten Resume", page.text)

    def test_compare_direct_post_requires_login(self) -> None:
        direct_client = TestClient(self.app)
        response = direct_client.post(
            "/compare",
            data={
                "jd_1": self._jd_fixture_text(),
                "jd_2": (FIXTURE_ROOT / "jds" / "ai_products_intern.md").read_text(encoding="utf-8"),
            },
            files={"resume": ("arjun_backend_student.txt", self._resume_fixture().read_bytes(), "text/plain")},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/login?next_path=%2Fcompare")
        self.assertNotIn(COOKIE_NAME, response.headers.get("set-cookie", ""))

    def test_compare_validation_requires_resume_and_two_jds(self) -> None:
        no_resume = self.client.post(
            "/compare",
            data={"jd_1": self._jd_fixture_text(), "jd_2": self._jd_fixture_text()},
        )
        self.assertEqual(no_resume.status_code, 200)
        self.assertIn("Upload one resume", no_resume.text)

        one_jd = self.client.post(
            "/compare",
            data={"jd_1": self._jd_fixture_text(), "jd_2": ""},
            files={"resume": ("arjun_backend_student.txt", self._resume_fixture().read_bytes(), "text/plain")},
        )
        self.assertEqual(one_jd.status_code, 200)
        self.assertIn("Paste at least two job descriptions", one_jd.text)

    def test_compare_can_use_preview_cache_and_finish_full_report(self) -> None:
        preview = self.client.post(
            "/resume/preview-parse",
            files={"resume": ("arjun_backend_student.txt", self._resume_fixture().read_bytes(), "text/plain")},
        )
        self.assertEqual(preview.status_code, 200)
        resume_cache_key = preview.json()["resume_cache_key"]

        response = self.client.post(
            "/compare",
            data={
                "resume_cache_key": resume_cache_key,
                "jd_1": self._jd_fixture_text(),
                "jd_2": (FIXTURE_ROOT / "jds" / "ai_products_intern.md").read_text(encoding="utf-8"),
            },
            files={"resume": ("arjun_backend_student.txt", self._resume_fixture().read_bytes(), "text/plain")},
            follow_redirects=False,
        )
        self.assertEqual(response.status_code, 303)
        comparison_id = response.headers["location"].rstrip("/").split("/")[-1]
        comparison = self.runtime.service.tracker.get_comparison(comparison_id)
        self.assertIsNotNone(comparison)
        packet_id = comparison.rows[0].packet_id
        packet = self.runtime.service.tracker.get(packet_id)
        self.assertIsNotNone(packet)
        self.assertIsNone(packet.report.rewritten_resume)

        completion = self.client.post(f"/diagnose/packets/{packet_id}/complete", follow_redirects=False)
        self.assertEqual(completion.status_code, 303)
        self.assertIn("/diagnose/progress/", completion.headers["location"])
        completed_packet = self.runtime.service.tracker.get(packet_id)
        self.assertIsNotNone(completed_packet.report.rewritten_resume)

    def test_signup_migrates_guest_history_and_sets_session_cookie(self) -> None:
        guest_client = TestClient(self.app)
        guest_user_id = "guest-webapp"
        guest_client.cookies.set(COOKIE_NAME, guest_user_id)
        packet = self.runtime.service.diagnose(
            resume_path=self._resume_fixture(),
            jd_text=self._jd_fixture_text(),
            user_id=guest_user_id,
            session_id="session-3",
            persist=True,
        ).packet

        response = guest_client.post(
            "/signup",
            data={"email": "jayesh@example.com", "password": "supersecure123", "next_path": "/history"},
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/history")
        self.assertIn(SESSION_COOKIE_NAME, response.headers.get("set-cookie", ""))

        user = self.app.state.auth_service.user_repository.load_by_email("jayesh@example.com")
        self.assertIsNotNone(user)
        self.assertEqual(self.runtime.service.tracker.list_entries(guest_user_id), [])
        migrated = self.runtime.service.tracker.get(packet.packet_id)
        self.assertIsNotNone(migrated)
        self.assertEqual(migrated.user_id, user.user_id)

        history = guest_client.get("/history")
        self.assertEqual(history.status_code, 200)
        self.assertIn(packet.resume_name, history.text)
        self.assertNotIn("This history belongs to a guest session.", history.text)

    def test_google_oauth_callback_sets_session_cookie(self) -> None:
        anonymous_client = TestClient(self.app)
        response = anonymous_client.get("/auth/google/start?next_path=/history", follow_redirects=False)
        state_cookie = anonymous_client.cookies.get(GOOGLE_STATE_COOKIE_NAME)
        self.assertTrue(state_cookie)

        fake_user = self.app.state.auth_service.register(email="google@example.com", password="password123")
        with mock.patch.object(
            self.app.state.auth_service,
            "authenticate_google_code",
            new=mock.AsyncMock(return_value=fake_user),
        ):
            callback = anonymous_client.get(
                f"/auth/google/callback?state={state_cookie}&code=fake-code",
                follow_redirects=False,
            )

        self.assertEqual(callback.status_code, 303)
        self.assertEqual(callback.headers["location"], "/history")
        self.assertIn(SESSION_COOKIE_NAME, callback.headers.get("set-cookie", ""))

    def test_settings_page_redirects_to_home(self) -> None:
        response = self.client.get("/settings", follow_redirects=False)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/")


if __name__ == "__main__":
    unittest.main()
