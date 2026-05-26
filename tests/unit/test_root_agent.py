from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent.agents.root_agent import AgentRuntime
from job_rejection_agent.config import Settings
from job_rejection_agent.services import DiagnosticService, render_packet_markdown


FIXTURE_ROOT = Path(__file__).resolve().parents[1]


class RootAgentModelFallbackTests(unittest.TestCase):
    def _build_packet(self):
        service = DiagnosticService(Settings(google_api_key=None))
        return service.diagnose(
            resume_path=FIXTURE_ROOT / "fixtures" / "resumes" / "arjun_backend_student.txt",
            jd_text=(FIXTURE_ROOT / "fixtures" / "jds" / "backend_newgrad.md").read_text(encoding="utf-8"),
            rejection_notes="Need stronger API ownership evidence.",
            persist=False,
        ).packet

    def _settings(self) -> Settings:
        with tempfile.TemporaryDirectory() as temp_dir:
            return Settings(
                google_api_key="google-key",
                phoenix_api_key=None,
                model_id="gemini-primary",
                model_fallbacks=("gemini-secondary",),
                eval_model_id="gemini-primary",
                eval_model_fallbacks=("gemini-secondary",),
                local_storage_path=Path(temp_dir) / "packets.json",
            )

    def test_run_diagnostic_uses_secondary_model_after_primary_quota_error(self) -> None:
        packet = self._build_packet()
        settings = self._settings()
        tracker = types.SimpleNamespace(
            find_by_session=mock.Mock(return_value=packet),
        )
        service = types.SimpleNamespace(tracker=tracker, diagnose=mock.Mock())
        runtime = AgentRuntime(settings=settings, service=service)
        session_service = types.SimpleNamespace(
            get_session=mock.AsyncMock(return_value=None),
            create_session=mock.AsyncMock(return_value=None),
            delete_session=mock.AsyncMock(return_value=None),
        )
        attempts: list[str] = []

        async def fake_attempt(**kwargs):
            attempts.append(kwargs["model_id"])
            if kwargs["model_id"] == "gemini-primary":
                raise RuntimeError("429 RESOURCE_EXHAUSTED")
            return "ADK final response"

        with (
            mock.patch.object(runtime, "adk_available", return_value=True),
            mock.patch("job_rejection_agent.agents.root_agent.create_session_service", return_value=session_service),
            mock.patch.object(runtime, "_reset_adk_session", new=mock.AsyncMock(return_value=None)),
            mock.patch.object(runtime, "_run_adk_attempt", side_effect=fake_attempt),
        ):
            result = runtime.run_diagnostic(
                resume_path=str(FIXTURE_ROOT / "fixtures" / "resumes" / "arjun_backend_student.txt"),
                jd_text=(FIXTURE_ROOT / "fixtures" / "jds" / "backend_newgrad.md").read_text(encoding="utf-8"),
                rejection_notes="Need stronger API ownership evidence.",
                user_id="test-user",
            )

        self.assertEqual(attempts, ["gemini-primary", "gemini-secondary"])
        self.assertTrue(result["used_adk"])
        self.assertEqual(result["text"], "ADK final response")
        self.assertEqual(result["packet_id"], packet.packet_id)
        service.diagnose.assert_not_called()

    def test_run_diagnostic_uses_secondary_model_after_primary_unavailable_error(self) -> None:
        packet = self._build_packet()
        settings = self._settings()
        tracker = types.SimpleNamespace(
            find_by_session=mock.Mock(return_value=packet),
        )
        service = types.SimpleNamespace(tracker=tracker, diagnose=mock.Mock())
        runtime = AgentRuntime(settings=settings, service=service)
        session_service = types.SimpleNamespace(
            get_session=mock.AsyncMock(return_value=None),
            create_session=mock.AsyncMock(return_value=None),
            delete_session=mock.AsyncMock(return_value=None),
        )
        attempts: list[str] = []

        async def fake_attempt(**kwargs):
            attempts.append(kwargs["model_id"])
            if kwargs["model_id"] == "gemini-primary":
                raise RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
            return "ADK final response"

        with (
            mock.patch.object(runtime, "adk_available", return_value=True),
            mock.patch("job_rejection_agent.agents.root_agent.create_session_service", return_value=session_service),
            mock.patch.object(runtime, "_reset_adk_session", new=mock.AsyncMock(return_value=None)),
            mock.patch.object(runtime, "_run_adk_attempt", side_effect=fake_attempt),
        ):
            result = runtime.run_diagnostic(
                resume_path=str(FIXTURE_ROOT / "fixtures" / "resumes" / "arjun_backend_student.txt"),
                jd_text=(FIXTURE_ROOT / "fixtures" / "jds" / "backend_newgrad.md").read_text(encoding="utf-8"),
                rejection_notes="Need stronger API ownership evidence.",
                user_id="test-user",
            )

        self.assertEqual(attempts, ["gemini-primary", "gemini-secondary"])
        self.assertTrue(result["used_adk"])
        self.assertEqual(result["text"], "ADK final response")
        self.assertEqual(result["packet_id"], packet.packet_id)
        service.diagnose.assert_not_called()

    def test_run_diagnostic_falls_back_to_deterministic_when_all_models_exhausted(self) -> None:
        packet = self._build_packet()
        settings = self._settings()
        tracker = types.SimpleNamespace(
            find_by_session=mock.Mock(return_value=None),
        )
        diagnose_result = types.SimpleNamespace(packet=packet)
        service = types.SimpleNamespace(tracker=tracker, diagnose=mock.Mock(return_value=diagnose_result))
        runtime = AgentRuntime(settings=settings, service=service)
        session_service = types.SimpleNamespace(
            get_session=mock.AsyncMock(return_value=None),
            create_session=mock.AsyncMock(return_value=None),
            delete_session=mock.AsyncMock(return_value=None),
        )

        async def exhausted_attempt(**kwargs):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

        with (
            mock.patch.object(runtime, "adk_available", return_value=True),
            mock.patch("job_rejection_agent.agents.root_agent.create_session_service", return_value=session_service),
            mock.patch.object(runtime, "_reset_adk_session", new=mock.AsyncMock(return_value=None)),
            mock.patch.object(runtime, "_run_adk_attempt", side_effect=exhausted_attempt),
        ):
            result = runtime.run_diagnostic(
                resume_path=str(FIXTURE_ROOT / "fixtures" / "resumes" / "arjun_backend_student.txt"),
                jd_text=(FIXTURE_ROOT / "fixtures" / "jds" / "backend_newgrad.md").read_text(encoding="utf-8"),
                rejection_notes="Need stronger API ownership evidence.",
                user_id="test-user",
            )

        self.assertFalse(result["used_adk"])
        self.assertEqual(result["packet_id"], packet.packet_id)
        self.assertEqual(result["text"], render_packet_markdown(packet))
        service.diagnose.assert_called_once()

    def test_run_diagnostic_falls_back_when_adk_does_not_persist_current_session_packet(self) -> None:
        packet = self._build_packet()
        settings = self._settings()
        tracker = types.SimpleNamespace(find_by_session=mock.Mock(return_value=None))
        diagnose_result = types.SimpleNamespace(packet=packet)
        service = types.SimpleNamespace(tracker=tracker, diagnose=mock.Mock(return_value=diagnose_result))
        runtime = AgentRuntime(settings=settings, service=service)
        session_service = types.SimpleNamespace(
            get_session=mock.AsyncMock(return_value=None),
            create_session=mock.AsyncMock(return_value=None),
            delete_session=mock.AsyncMock(return_value=None),
        )

        with (
            mock.patch.object(runtime, "adk_available", return_value=True),
            mock.patch("job_rejection_agent.agents.root_agent.create_session_service", return_value=session_service),
            mock.patch.object(runtime, "_reset_adk_session", new=mock.AsyncMock(return_value=None)),
            mock.patch.object(runtime, "_run_adk_attempt", new=mock.AsyncMock(return_value="ADK final response")),
        ):
            result = runtime.run_diagnostic(
                resume_path=str(FIXTURE_ROOT / "fixtures" / "resumes" / "arjun_backend_student.txt"),
                jd_text=(FIXTURE_ROOT / "fixtures" / "jds" / "backend_newgrad.md").read_text(encoding="utf-8"),
                rejection_notes="Need stronger API ownership evidence.",
                user_id="test-user",
            )

        self.assertFalse(result["used_adk"])
        self.assertEqual(result["packet_id"], packet.packet_id)
        self.assertEqual(result["text"], render_packet_markdown(packet))
        service.diagnose.assert_called_once()


if __name__ == "__main__":
    unittest.main()
