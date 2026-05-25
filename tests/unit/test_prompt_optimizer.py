from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from job_rejection_agent.config import Settings
from job_rejection_agent.observability.prompt_optimizer import PromptOptimizer


class PromptOptimizerTests(unittest.TestCase):
    def test_candidate_prompt_uses_low_scoring_trace_failures(self) -> None:
        baseline_prompt = "\n".join(
            [
                "Diagnose ats, evidence, and level-fit gaps.",
                "Provide exact edits.",
            ]
        )
        traces = [
            {
                "span_id": "span-1",
                "composite_score": 0.25,
                "annotations": [
                    {
                        "name": "actionability",
                        "score": 0.0,
                        "explanation": "The advice lacked concrete next steps.",
                    },
                    {
                        "name": "specificity",
                        "score": 0.0,
                        "explanation": "The response never named the user's projects.",
                    },
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            prompt_path = temp_root / "coaching_system_prompt.txt"
            candidate_path = temp_root / "coaching_system_prompt_candidate.txt"
            history_dir = temp_root / "prompt_history"
            prompt_path.write_text(baseline_prompt, encoding="utf-8")
            settings = Settings(
                google_api_key=None,
                phoenix_api_key="phoenix-key",
                prompt_path=prompt_path,
                prompt_candidate_path=candidate_path,
                prompt_history_dir=history_dir,
            )
            optimizer = PromptOptimizer(settings=settings)
            with mock.patch(
                "job_rejection_agent.observability.prompt_optimizer.query_low_scoring_trace_summaries",
                return_value=traces,
            ):
                candidate_prompt, improvement_run = optimizer.optimize()

        self.assertIn("five concrete steps", candidate_prompt.lower())
        self.assertIn("name the exact project", candidate_prompt.lower())
        self.assertIn("actionability failed", improvement_run.analysis.lower())
        self.assertEqual(improvement_run.source_span_ids, ["span-1"])


if __name__ == "__main__":
    unittest.main()
