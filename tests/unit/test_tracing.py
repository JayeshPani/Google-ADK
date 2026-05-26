from __future__ import annotations

from pathlib import Path
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent.config import Settings
from job_rejection_agent.observability import tracing


class TracingTests(unittest.TestCase):
    def setUp(self) -> None:
        tracing._TRACING_READY = False

    def tearDown(self) -> None:
        tracing._TRACING_READY = False

    def test_returns_false_without_phoenix_api_key(self) -> None:
        settings = Settings(phoenix_api_key=None)
        self.assertFalse(tracing.configure_tracing(settings))

    def test_registers_phoenix_and_explicitly_instruments_adk_and_mcp(self) -> None:
        register = mock.Mock(return_value="tracer-provider")
        adk_instrumentor = mock.Mock()
        adk_instrumentor.is_instrumented_by_opentelemetry = False
        mcp_instrumentor = mock.Mock()
        mcp_instrumentor.is_instrumented_by_opentelemetry = False

        phoenix_pkg = types.ModuleType("phoenix")
        phoenix_otel = types.ModuleType("phoenix.otel")
        phoenix_otel.register = register

        openinference_pkg = types.ModuleType("openinference")
        openinference_instr = types.ModuleType("openinference.instrumentation")
        openinference_google_adk = types.ModuleType("openinference.instrumentation.google_adk")
        openinference_google_adk.GoogleADKInstrumentor = mock.Mock(return_value=adk_instrumentor)
        openinference_mcp = types.ModuleType("openinference.instrumentation.mcp")
        openinference_mcp.MCPInstrumentor = mock.Mock(return_value=mcp_instrumentor)

        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                phoenix_api_key="phoenix-key",
                phoenix_collector_endpoint="https://app.phoenix.arize.com/s/test-space",
                phoenix_project_name="job-rejection-agent",
                local_storage_path=Path(temp_dir) / "job_packets.json",
            )
            with mock.patch.dict(
                sys.modules,
                {
                    "phoenix": phoenix_pkg,
                    "phoenix.otel": phoenix_otel,
                    "openinference": openinference_pkg,
                    "openinference.instrumentation": openinference_instr,
                    "openinference.instrumentation.google_adk": openinference_google_adk,
                    "openinference.instrumentation.mcp": openinference_mcp,
                },
            ):
                with mock.patch.dict(os.environ, {}, clear=True):
                    self.assertTrue(tracing.configure_tracing(settings))
                    self.assertEqual(os.environ["PHOENIX_API_KEY"], "phoenix-key")
                    self.assertEqual(
                        os.environ["PHOENIX_COLLECTOR_ENDPOINT"],
                        "https://app.phoenix.arize.com/s/test-space",
                    )
                    self.assertTrue(os.environ["PHOENIX_WORKING_DIR"].endswith("/phoenix"))

        register.assert_called_once_with(
            project_name="job-rejection-agent",
            auto_instrument=False,
        )
        adk_instrumentor.instrument.assert_called_once_with(tracer_provider="tracer-provider")
        mcp_instrumentor.instrument.assert_called_once_with(tracer_provider="tracer-provider")


if __name__ == "__main__":
    unittest.main()
