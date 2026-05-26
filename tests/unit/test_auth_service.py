from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent.config import Settings
from job_rejection_agent.persistence import JobTracker, build_packet_repository
from job_rejection_agent.services import AuthError, AuthService
from job_rejection_agent.services.diagnostic_service import DiagnosticService


FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


class AuthServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_root = Path(self.temp_dir.name)
        self.settings = Settings(
            google_api_key=None,
            phoenix_api_key=None,
            local_storage_path=temp_root / "packets.json",
            local_user_storage_path=temp_root / "users.json",
            session_db_url=f"sqlite+aiosqlite:///{temp_root / 'sessions.db'}",
            app_secret_key="auth-test-secret",
        )
        self.tracker = JobTracker(repository=build_packet_repository(self.settings))
        self.auth = AuthService(settings=self.settings, tracker=self.tracker)
        self.diagnostic_service = DiagnosticService(settings=self.settings, tracker=self.tracker)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_register_authenticate_and_verify_session(self) -> None:
        user = self.auth.register(email="user@example.com", password="password123")

        self.assertEqual(user.email, "user@example.com")

        authenticated = self.auth.authenticate(email="user@example.com", password="password123")
        self.assertEqual(authenticated.user_id, user.user_id)

        token = self.auth.create_session_token(user, ttl_seconds=3600)
        session = self.auth.verify_session_token(token)
        self.assertIsNotNone(session)
        self.assertEqual(session.user_id, user.user_id)

    def test_register_rejects_duplicate_email(self) -> None:
        self.auth.register(email="user@example.com", password="password123")

        with self.assertRaises(AuthError):
            self.auth.register(email="USER@example.com", password="password123")

    def test_authenticate_migrates_guest_packets(self) -> None:
        guest_user_id = "guest-abc123"
        resume_path = FIXTURE_ROOT / "resumes" / "arjun_backend_student.txt"
        jd_text = (FIXTURE_ROOT / "jds" / "backend_newgrad.md").read_text(encoding="utf-8")
        packet = self.diagnostic_service.diagnose(
            resume_path=resume_path,
            jd_text=jd_text,
            user_id=guest_user_id,
            session_id="guest-session",
            persist=True,
        ).packet
        user = self.auth.register(email="user@example.com", password="password123")

        authenticated = self.auth.authenticate(
            email="user@example.com",
            password="password123",
            guest_user_id=guest_user_id,
        )

        self.assertEqual(authenticated.user_id, user.user_id)
        migrated = self.tracker.get(packet.packet_id)
        self.assertIsNotNone(migrated)
        self.assertEqual(migrated.user_id, user.user_id)
        self.assertEqual(self.tracker.list_entries(guest_user_id), [])


if __name__ == "__main__":
    unittest.main()
