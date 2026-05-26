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
from job_rejection_agent.observability.prompt_optimizer import PromptEvaluationSuite, PromptOptimizer


def _suite(
    *,
    composite_score: float,
    actionability: float,
    evidence_grounding: float,
    specificity: float,
    non_hallucination: float,
    expectation_score: float,
    valid: bool = True,
    analysis: str = "held-out suite ok",
) -> PromptEvaluationSuite:
    return PromptEvaluationSuite(
        scores={
            "composite_score": composite_score,
            "actionability": actionability,
            "evidence_grounding": evidence_grounding,
            "specificity": specificity,
            "non_hallucination": non_hallucination,
            "expectation_score": expectation_score,
            "success_rate": 1.0 if valid else 0.0,
            "mcp_readback_rate": 1.0 if valid else 0.0,
        },
        case_results=[],
        valid=valid,
        analysis=analysis,
        source_span_ids=["span-a"] if valid else [],
    )


class PromptOptimizerTests(unittest.TestCase):
    def test_candidate_prompt_uses_low_scoring_trace_failures_when_live_gate_unavailable(self) -> None:
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
        self.assertFalse(improvement_run.promoted)
        self.assertIn("run_live_network_tests", improvement_run.analysis.lower())
        self.assertEqual(improvement_run.source_span_ids, ["span-1"])

    def test_prompt_promotes_only_when_held_out_suite_improves_without_regression(self) -> None:
        baseline_prompt = "Diagnose ats, evidence, and level-fit gaps.\nProvide exact edits.\n"
        traces = [{"span_id": "span-1", "annotations": []}]
        baseline_suite = _suite(
            composite_score=0.71,
            actionability=0.65,
            evidence_grounding=0.8,
            specificity=0.7,
            non_hallucination=1.0,
            expectation_score=0.75,
        )
        candidate_suite = _suite(
            composite_score=0.78,
            actionability=0.8,
            evidence_grounding=0.85,
            specificity=0.75,
            non_hallucination=1.0,
            expectation_score=0.82,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            prompt_path = temp_root / "coaching_system_prompt.txt"
            candidate_path = temp_root / "coaching_system_prompt_candidate.txt"
            history_dir = temp_root / "prompt_history"
            prompt_path.write_text(baseline_prompt, encoding="utf-8")
            settings = Settings(
                google_api_key="dummy-google",
                phoenix_api_key="phoenix-key",
                prompt_path=prompt_path,
                prompt_candidate_path=candidate_path,
                prompt_history_dir=history_dir,
            )
            optimizer = PromptOptimizer(settings=settings)
            with (
                mock.patch.object(optimizer, "_call_llm", return_value=None),
                mock.patch(
                    "job_rejection_agent.observability.prompt_optimizer.query_low_scoring_trace_summaries",
                    return_value=traces,
                ),
                mock.patch.object(optimizer, "_run_prompt_suite", side_effect=[baseline_suite, candidate_suite]),
            ):
                candidate_prompt, improvement_run = optimizer.optimize()

            promoted_prompt = prompt_path.read_text(encoding="utf-8")

        self.assertTrue(improvement_run.promoted)
        self.assertEqual(improvement_run.baseline_scores, baseline_suite.scores)
        self.assertEqual(improvement_run.candidate_scores, candidate_suite.scores)
        self.assertEqual(promoted_prompt, candidate_prompt)

    def test_prompt_is_not_promoted_when_candidate_regresses_guard_metric(self) -> None:
        baseline_prompt = "Diagnose ats, evidence, and level-fit gaps.\nProvide exact edits.\n"
        baseline_suite = _suite(
            composite_score=0.74,
            actionability=0.75,
            evidence_grounding=0.82,
            specificity=0.8,
            non_hallucination=1.0,
            expectation_score=0.76,
        )
        candidate_suite = _suite(
            composite_score=0.79,
            actionability=0.82,
            evidence_grounding=0.85,
            specificity=0.7,
            non_hallucination=1.0,
            expectation_score=0.81,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            prompt_path = temp_root / "coaching_system_prompt.txt"
            candidate_path = temp_root / "coaching_system_prompt_candidate.txt"
            history_dir = temp_root / "prompt_history"
            prompt_path.write_text(baseline_prompt, encoding="utf-8")
            settings = Settings(
                google_api_key="dummy-google",
                phoenix_api_key="phoenix-key",
                prompt_path=prompt_path,
                prompt_candidate_path=candidate_path,
                prompt_history_dir=history_dir,
            )
            optimizer = PromptOptimizer(settings=settings)
            with (
                mock.patch.object(optimizer, "_call_llm", return_value=None),
                mock.patch(
                    "job_rejection_agent.observability.prompt_optimizer.query_low_scoring_trace_summaries",
                    return_value=[],
                ),
                mock.patch.object(optimizer, "_run_prompt_suite", side_effect=[baseline_suite, candidate_suite]),
            ):
                candidate_prompt, improvement_run = optimizer.optimize()

            active_prompt = prompt_path.read_text(encoding="utf-8")

        self.assertFalse(improvement_run.promoted)
        self.assertIn("regressed on guarded metrics", improvement_run.analysis.lower())
        self.assertNotEqual(active_prompt, candidate_prompt)


if __name__ == "__main__":
    unittest.main()
