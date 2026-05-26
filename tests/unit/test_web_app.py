from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from app.web_app import COOKIE_NAME, SESSION_COOKIE_NAME, create_app
from job_rejection_agent.agents.root_agent import AgentRuntime
from job_rejection_agent.config import Settings


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


class WebAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_root = Path(self.temp_dir.name)
        self.settings = Settings(
            google_api_key=None,
            phoenix_api_key=None,
            local_storage_path=temp_root / "packets.json",
            local_user_storage_path=temp_root / "users.json",
            session_db_url=f"sqlite+aiosqlite:///{temp_root / 'sessions.db'}",
            app_secret_key="test-secret",
        )
        self.runtime = AgentRuntime(settings=self.settings)
        self.optimizer = types.SimpleNamespace(optimize=lambda: ("", None))
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

    def test_login_page_renders(self) -> None:
        response = self.client.get("/login")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Save every diagnosis to your account.", response.text)
        self.assertIn("Sign In", response.text)

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


if __name__ == "__main__":
    unittest.main()
