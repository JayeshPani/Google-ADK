from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent.config import get_settings
from job_rejection_agent.observability.live_verifier import get_live_network_test_skip_reason
from job_rejection_agent.observability.prompt_optimizer import CORE_ANNOTATION_NAMES, PromptOptimizer


class PromptOptimizerLiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = get_settings()
        skip_reason = get_live_network_test_skip_reason(cls.settings)
        if skip_reason:
            raise unittest.SkipTest(skip_reason)
        if os.getenv("RUN_LIVE_PROMPT_GATE_TESTS", "").strip().lower() not in {"1", "true", "yes", "on"}:
            raise unittest.SkipTest("Set RUN_LIVE_PROMPT_GATE_TESTS=1 to run live prompt-gate integration tests.")

    def test_held_out_suite_runs_against_live_stack(self) -> None:
        optimizer = PromptOptimizer(settings=self.settings)
        baseline_prompt = self.settings.prompt_path.read_text(encoding="utf-8")
        cases = optimizer._load_held_out_cases()[:1]
        suite = optimizer._run_prompt_suite(baseline_prompt, cases)

        self.assertTrue(suite.valid, suite.analysis)
        self.assertEqual(suite.scores["success_rate"], 1.0, suite.analysis)
        self.assertEqual(suite.scores["mcp_readback_rate"], 1.0, suite.analysis)
        for metric in CORE_ANNOTATION_NAMES:
            self.assertIn(metric, suite.scores)
        self.assertGreaterEqual(suite.scores["composite_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
