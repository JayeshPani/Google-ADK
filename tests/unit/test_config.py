from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent import config


class ConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        config.get_settings.cache_clear()

    def test_default_mcp_command_prefers_installed_binary(self) -> None:
        with mock.patch("job_rejection_agent.config.shutil.which", return_value="/usr/local/bin/phoenix-mcp"):
            self.assertEqual(config._default_mcp_command(), "phoenix-mcp")

        with mock.patch("job_rejection_agent.config.shutil.which", return_value=None):
            self.assertEqual(config._default_mcp_command(), "npx")

    def test_get_settings_hydrates_direct_binary_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch("job_rejection_agent.config.load_dotenv", return_value=False):
                with mock.patch("job_rejection_agent.config.shutil.which", return_value="/usr/local/bin/phoenix-mcp"):
                    with mock.patch.dict(
                        os.environ,
                        {
                            "PHOENIX_API_KEY": "phoenix-key",
                            "PHOENIX_COLLECTOR_ENDPOINT": "https://app.phoenix.arize.com/s/demo-space",
                            "LOCAL_STORAGE_PATH": str(Path(temp_dir) / "job_packets.json"),
                        },
                        clear=True,
                    ):
                        settings = config.get_settings()

        self.assertEqual(settings.phoenix_mcp_command, "phoenix-mcp")
        self.assertEqual(settings.phoenix_mcp_args, ("--baseUrl", "https://app.phoenix.arize.com"))

    def test_read_mcp_args_replaces_known_placeholders(self) -> None:
        args = config._read_mcp_args(
            "--baseUrl,{PHOENIX_COLLECTOR_ENDPOINT},--apiKey,{PHOENIX_API_KEY}",
            command="phoenix-mcp",
            substitutions={
                "PHOENIX_API_KEY": "phoenix-key",
                "PHOENIX_BASE_URL": "https://app.phoenix.arize.com",
                "PHOENIX_COLLECTOR_ENDPOINT": "https://app.phoenix.arize.com/s/demo-space",
            },
        )
        self.assertEqual(
            args,
            (
                "--baseUrl",
                "https://app.phoenix.arize.com/s/demo-space",
                "--apiKey",
                "phoenix-key",
            ),
        )

    def test_settings_expose_model_fallback_candidates(self) -> None:
        settings = config.Settings(
            model_id="gemini-2.5-flash",
            model_fallbacks=("gemini-2.5-flash-lite", "gemini-2.5-flash"),
            eval_model_id="gemini-2.5-flash",
            eval_model_fallbacks=("gemini-3.1-flash-lite",),
        )
        self.assertEqual(
            settings.generation_model_candidates,
            ("gemini-2.5-flash", "gemini-2.5-flash-lite"),
        )
        self.assertEqual(
            settings.evaluation_model_candidates,
            ("gemini-2.5-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite"),
        )

    def test_google_genai_enabled_supports_google_ai_studio_and_vertex(self) -> None:
        studio_settings = config.Settings(
            google_api_key="studio-key",
            google_genai_use_vertexai=False,
        )
        vertex_settings = config.Settings(
            google_api_key=None,
            google_genai_use_vertexai=True,
            google_cloud_project="demo-project",
            google_cloud_location="us-central1",
        )
        invalid_vertex_settings = config.Settings(
            google_api_key=None,
            google_genai_use_vertexai=True,
            google_cloud_project=None,
            google_cloud_location="us-central1",
        )

        self.assertTrue(studio_settings.google_genai_enabled)
        self.assertEqual(studio_settings.google_genai_backend, "google_ai_studio")
        self.assertTrue(vertex_settings.google_genai_enabled)
        self.assertEqual(vertex_settings.google_genai_backend, "vertexai")
        self.assertFalse(invalid_vertex_settings.google_genai_enabled)


if __name__ == "__main__":
    unittest.main()
