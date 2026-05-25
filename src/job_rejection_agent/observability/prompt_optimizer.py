"""Guarded prompt improvement loop."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from job_rejection_agent.config import Settings, get_settings
from job_rejection_agent.domain import ImprovementRun

from .phoenix_mcp import query_low_scoring_trace_summaries


@dataclass(slots=True)
class PromptComparison:
    baseline_score: float
    candidate_score: float
    promoted: bool
    analysis: str


class PromptOptimizer:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.prompt_history_dir.mkdir(exist_ok=True)

    def _load_prompt(self, path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def _load_eval_cases(self) -> list[dict]:
        case_path = self.settings.prompt_path.parents[3] / "tests" / "fixtures" / "eval_cases" / "prompt_eval_cases.json"
        if not case_path.exists():
            return []
        return json.loads(case_path.read_text(encoding="utf-8"))

    def _score_prompt(self, prompt_text: str, cases: list[dict]) -> float:
        if not prompt_text:
            return 0.0
        keywords = {
            "ats": 1.0,
            "evidence": 1.0,
            "level-fit": 1.0,
            "exact edits": 1.0,
            "under three hours": 1.0,
            "do not invent": 1.0,
            "recruiter": 0.5,
            "phoenix": 0.5,
        }
        base = sum(weight for key, weight in keywords.items() if key in prompt_text.lower())
        if not cases:
            return round(base, 2)
        scenario_bonus = 0.0
        lowered = prompt_text.lower()
        for case in cases:
            expected = case.get("must_include", [])
            matches = sum(1 for token in expected if token.lower() in lowered)
            scenario_bonus += matches / max(1, len(expected))
        return round(base + (scenario_bonus / len(cases)), 2)

    def _failure_summary(self, traces: list[dict]) -> tuple[str, list[str]]:
        if not traces:
            return "No low-scoring Phoenix traces were available.", []
        failures = Counter[str]()
        explanations: list[str] = []
        for trace in traces:
            for annotation in trace.get("annotations", []):
                score = annotation.get("score")
                if score is not None and score < 0.75:
                    failures[str(annotation.get("name", "unknown"))] += 1
                    explanation = annotation.get("explanation")
                    if explanation:
                        explanations.append(str(explanation))
        failure_bits = [f"{name} failed in {count} trace(s)" for name, count in failures.most_common()]
        summary = "; ".join(failure_bits) if failure_bits else "Recent traces had annotations but no failing scores."
        return summary, explanations[:5]

    def _call_llm(self, prompt: str) -> str | None:
        if not self.settings.google_api_key:
            return None
        try:
            from google import genai
        except ImportError:
            return None
        try:
            client = genai.Client(api_key=self.settings.google_api_key)
            response = client.models.generate_content(
                model=self.settings.model_id,
                contents=prompt,
            )
        except Exception:
            return None
        text = getattr(response, "text", None)
        return text.strip() if text else None

    def _heuristic_additions(self, baseline: str, traces: list[dict]) -> tuple[list[str], str]:
        additions: list[str] = []
        summary, explanations = self._failure_summary(traces)
        failing_names = {
            annotation.get("name")
            for trace in traces
            for annotation in trace.get("annotations", [])
            if annotation.get("score") is not None and annotation.get("score") < 0.75
        }
        if "actionability" in failing_names:
            additions.append(
                "- Every action plan must contain at least five concrete steps, each scoped to under three hours and tied to a named resume bullet or project."
            )
        if "evidence_grounding" in failing_names:
            additions.append(
                "- Every top gap and exact edit must cite resume, job-description, or recruiter-feedback evidence explicitly."
            )
        if "specificity" in failing_names:
            additions.append(
                "- Name the exact project, skill, or resume line being revised; never use vague phrases like 'your projects' without naming them."
            )
        if "non_hallucination" in failing_names or "do not invent" not in baseline.lower():
            additions.append(
                "- Do not invent metrics or experience. If measurable outcomes are missing, use a clearly marked placeholder and tell the user to replace it."
            )
        if "provenance" not in baseline.lower():
            additions.append("- Add a provenance line for every major recommendation.")
        if "recruiter feedback" not in baseline.lower():
            additions.append("- Treat recruiter or rejection feedback as highest-priority evidence when it exists.")
        if not additions:
            additions.append("- Tighten responses so every top gap includes evidence, consequence, and immediate edit.")
        analysis = summary
        if explanations:
            analysis += " Key explanations: " + " | ".join(explanations)
        return additions, analysis

    def _build_candidate(self, baseline: str, traces: list[dict]) -> tuple[str, str]:
        additions, analysis = self._heuristic_additions(baseline, traces)
        if traces:
            llm_prompt = "\n".join(
                [
                    "Rewrite this system prompt for a job rejection diagnostic agent.",
                    "Keep the existing response structure and tone.",
                    "Strengthen the prompt using the observed low-scoring Phoenix trace patterns below.",
                    "Do not add markdown fences. Return only the revised prompt text.",
                    "",
                    "Current prompt:",
                    baseline,
                    "",
                    "Observed low-scoring trace summaries:",
                    json.dumps(traces[:5], indent=2),
                    "",
                    "Guardrails that must remain explicit:",
                    "- Diagnose ATS, evidence, and level-fit gaps.",
                    "- Provide exact edits and under-three-hour action steps.",
                    "- Use recruiter feedback when present.",
                    "- Mention provenance for major recommendations.",
                    "- Do not invent metrics or experience.",
                ]
            )
            candidate = self._call_llm(llm_prompt)
            if candidate:
                return candidate, analysis
        candidate = baseline.rstrip() + "\n" + "\n".join(additions) + "\n"
        return candidate, analysis

    def optimize(self) -> tuple[str, ImprovementRun]:
        baseline = self._load_prompt(self.settings.prompt_path)
        traces = query_low_scoring_trace_summaries(settings=self.settings)
        candidate_prompt, analysis = self._build_candidate(baseline, traces)
        cases = self._load_eval_cases()
        baseline_score = self._score_prompt(baseline, cases)
        candidate_score = self._score_prompt(candidate_prompt, cases)
        promoted = candidate_score > baseline_score

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        history_path = self.settings.prompt_history_dir / f"prompt_{timestamp}.md"
        history_path.write_text(
            "\n".join(
                [
                    f"# Prompt optimization {timestamp}",
                    "",
                    "## Analysis",
                    analysis,
                    "",
                    "## Baseline prompt",
                    "```text",
                    baseline,
                    "```",
                    "",
                    "## Candidate prompt",
                    "```text",
                    candidate_prompt,
                    "```",
                ]
            ),
            encoding="utf-8",
        )
        self.settings.prompt_candidate_path.write_text(candidate_prompt, encoding="utf-8")
        if promoted:
            self.settings.prompt_path.write_text(candidate_prompt, encoding="utf-8")

        improvement_run = ImprovementRun(
            run_id=str(uuid.uuid4()),
            baseline_prompt_version=self.settings.prompt_version,
            candidate_prompt_version=f"{self.settings.prompt_version}-candidate",
            source_span_ids=[trace["span_id"] for trace in traces],
            baseline_scores={"prompt_guardrail_score": baseline_score},
            candidate_scores={"prompt_guardrail_score": candidate_score},
            promoted=promoted,
            analysis=analysis,
        )
        return candidate_prompt, improvement_run
