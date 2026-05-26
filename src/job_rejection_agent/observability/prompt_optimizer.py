"""Guarded prompt improvement loop with held-out replay gating."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
import uuid
from typing import Any

from job_rejection_agent.config import Settings, get_settings
from job_rejection_agent.domain import ImprovementRun
from job_rejection_agent.google_models import build_google_genai_client, is_resource_exhausted_error

from .live_verifier import REQUIRED_ANNOTATION_NAMES, get_live_network_test_skip_reason, wait_for_trace_readback
from .phoenix_mcp import query_low_scoring_trace_summaries


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORE_ANNOTATION_NAMES = (
    "actionability",
    "evidence_grounding",
    "specificity",
    "non_hallucination",
)
DEFAULT_REQUIRED_SECTIONS = (
    "Match Score",
    "Top rejection drivers",
    "Exact edits",
    "One-week action plan",
    "Interview prep",
)
PROMOTION_MIN_COMPOSITE_DELTA = 0.03
PROMOTION_MIN_METRIC_DELTA = 0.05
PROMOTION_GUARD_METRICS = (*CORE_ANNOTATION_NAMES, "expectation_score")


@dataclass(slots=True)
class PromptEvaluationSuite:
    scores: dict[str, float]
    case_results: list[dict[str, Any]]
    valid: bool
    analysis: str
    source_span_ids: list[str]


class PromptOptimizer:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.settings.prompt_history_dir.mkdir(exist_ok=True)
        self.settings.improvement_state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_lock = Lock()

    def _default_state(self) -> dict[str, Any]:
        return {
            "status": "idle",
            "successful_diagnosis_count": 0,
            "last_auto_run_diagnosis_count": 0,
            "last_started_at": "",
            "last_completed_at": "",
            "last_error": "",
            "last_trigger_packet_id": "",
            "last_trigger_session_id": "",
            "candidate_prompt": "",
            "improvement_run": None,
        }

    def _load_state(self) -> dict[str, Any]:
        state_path = self.settings.improvement_state_path
        if not state_path.exists():
            return self._default_state()
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._default_state()
        return {**self._default_state(), **payload}

    def _save_state(self, state: dict[str, Any]) -> None:
        self.settings.improvement_state_path.write_text(
            json.dumps(state, indent=2),
            encoding="utf-8",
        )

    def _persist_snapshot(
        self,
        *,
        candidate_prompt: str | None = None,
        improvement_run: ImprovementRun | None = None,
        status: str | None = None,
        last_error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._state_lock:
            state = self._load_state()
            if candidate_prompt is not None:
                state["candidate_prompt"] = candidate_prompt
            if improvement_run is not None:
                state["improvement_run"] = improvement_run.to_dict()
            if status is not None:
                state["status"] = status
            if last_error is not None:
                state["last_error"] = last_error
            if extra:
                state.update(extra)
            self._save_state(state)
            return state

    def latest_snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            state = self._load_state()
        interval = max(1, self.settings.auto_prompt_improvement_every_n_diagnoses)
        successful_count = int(state.get("successful_diagnosis_count", 0) or 0)
        last_auto_run_count = int(state.get("last_auto_run_diagnosis_count", 0) or 0)
        diagnoses_since_last_run = max(0, successful_count - last_auto_run_count)
        diagnoses_until_next_run = 0 if state.get("status") == "running" else max(0, interval - diagnoses_since_last_run)
        improvement_payload = state.get("improvement_run")
        return {
            "status": str(state.get("status", "idle")),
            "auto_enabled": bool(self.settings.auto_prompt_improvement_enabled),
            "auto_interval": interval,
            "successful_diagnosis_count": successful_count,
            "last_auto_run_diagnosis_count": last_auto_run_count,
            "diagnoses_since_last_run": diagnoses_since_last_run,
            "diagnoses_until_next_run": diagnoses_until_next_run,
            "last_started_at": str(state.get("last_started_at", "")),
            "last_completed_at": str(state.get("last_completed_at", "")),
            "last_error": str(state.get("last_error", "")),
            "last_trigger_packet_id": str(state.get("last_trigger_packet_id", "")),
            "last_trigger_session_id": str(state.get("last_trigger_session_id", "")),
            "candidate_prompt": str(state.get("candidate_prompt", "")),
            "improvement_run": ImprovementRun.from_dict(improvement_payload) if isinstance(improvement_payload, dict) else None,
        }

    def _load_prompt(self, path: Path) -> str:
        return path.read_text(encoding="utf-8") if path.exists() else ""

    def _load_held_out_cases(self) -> list[dict[str, Any]]:
        case_path = PROJECT_ROOT / "tests" / "fixtures" / "eval_cases" / "held_out_diagnostic_cases.json"
        if not case_path.exists():
            return []
        return json.loads(case_path.read_text(encoding="utf-8"))

    def _failure_summary(self, traces: list[dict[str, Any]]) -> tuple[str, list[str]]:
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
        if not self.settings.google_genai_enabled:
            return None
        try:
            client = build_google_genai_client(self.settings)
        except ImportError:
            return None
        for model_id in self.settings.generation_model_candidates:
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                )
            except Exception as exc:
                if is_resource_exhausted_error(exc):
                    continue
                continue
            text = getattr(response, "text", None)
            if text:
                return text.strip()
        return None

    def _heuristic_additions(self, baseline: str, traces: list[dict[str, Any]]) -> tuple[list[str], str]:
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

    def _build_candidate(self, baseline: str, traces: list[dict[str, Any]]) -> tuple[str, str]:
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

    def _resolve_case_path(self, relative_or_absolute: str) -> Path:
        candidate = Path(relative_or_absolute)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / relative_or_absolute

    def _annotation_scores(self, summary: dict[str, Any] | None) -> tuple[dict[str, float], list[str]]:
        scores = {name: 0.0 for name in CORE_ANNOTATION_NAMES}
        annotation_names: set[str] = set()
        for annotation in (summary or {}).get("annotations", []):
            if not isinstance(annotation, dict):
                continue
            name = annotation.get("name")
            if not isinstance(name, str):
                continue
            annotation_names.add(name)
            score = annotation.get("score")
            try:
                scores[name] = float(score)
            except (TypeError, ValueError):
                continue
        missing = sorted(set(REQUIRED_ANNOTATION_NAMES) - annotation_names)
        return scores, missing

    def _expectation_score(self, case: dict[str, Any], result: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        text = str(result.get("text", ""))
        lowered = text.lower()
        required_references = [str(item) for item in case.get("required_references", [])]
        required_sections = [str(item) for item in case.get("required_sections", DEFAULT_REQUIRED_SECTIONS)]
        expected_decision = str(case.get("expected_decision", "")).strip().lower()

        reference_hits = sum(1 for token in required_references if token.lower() in lowered)
        section_hits = sum(1 for token in required_sections if token.lower() in lowered)
        reference_coverage = reference_hits / len(required_references) if required_references else 1.0
        section_coverage = section_hits / len(required_sections) if required_sections else 1.0

        packet = result.get("packet")
        actual_decision = getattr(getattr(packet, "report", None), "recommended_decision", "")
        decision_match = 1.0 if not expected_decision or str(actual_decision).strip().lower() == expected_decision else 0.0

        expectation_score = round((decision_match + reference_coverage + section_coverage) / 3, 3)
        return expectation_score, {
            "expected_decision": expected_decision,
            "actual_decision": actual_decision,
            "decision_match": bool(decision_match),
            "reference_coverage": round(reference_coverage, 3),
            "section_coverage": round(section_coverage, 3),
        }

    def _run_prompt_suite(self, prompt_text: str, cases: list[dict[str, Any]]) -> PromptEvaluationSuite:
        skip_reason = get_live_network_test_skip_reason(self.settings)
        if skip_reason:
            return PromptEvaluationSuite(
                scores={
                    "composite_score": 0.0,
                    "expectation_score": 0.0,
                    "success_rate": 0.0,
                    "mcp_readback_rate": 0.0,
                    **{name: 0.0 for name in CORE_ANNOTATION_NAMES},
                },
                case_results=[],
                valid=False,
                analysis=skip_reason,
                source_span_ids=[],
            )
        if not prompt_text.strip():
            return PromptEvaluationSuite(
                scores={
                    "composite_score": 0.0,
                    "expectation_score": 0.0,
                    "success_rate": 0.0,
                    "mcp_readback_rate": 0.0,
                    **{name: 0.0 for name in CORE_ANNOTATION_NAMES},
                },
                case_results=[],
                valid=False,
                analysis="Prompt text was empty, so no held-out replay could run.",
                source_span_ids=[],
            )
        if not cases:
            return PromptEvaluationSuite(
                scores={
                    "composite_score": 0.0,
                    "expectation_score": 0.0,
                    "success_rate": 0.0,
                    "mcp_readback_rate": 0.0,
                    **{name: 0.0 for name in CORE_ANNOTATION_NAMES},
                },
                case_results=[],
                valid=False,
                analysis="No held-out diagnostic cases were configured.",
                source_span_ids=[],
            )

        from job_rejection_agent.agents.root_agent import AgentRuntime

        runtime = AgentRuntime(settings=self.settings)
        case_results: list[dict[str, Any]] = []
        for case in cases:
            case_name = str(case.get("name", f"case-{len(case_results) + 1}"))
            try:
                result = runtime.run_diagnostic(
                    resume_path=str(self._resolve_case_path(str(case["resume_path"]))),
                    jd_text=self._resolve_case_path(str(case["jd_path"])).read_text(encoding="utf-8"),
                    rejection_notes=str(case.get("rejection_notes", "")),
                    user_id=f"prompt-gate-{case_name}-{uuid.uuid4().hex[:8]}",
                    prompt_text_override=prompt_text,
                )
            except Exception as exc:
                case_results.append(
                    {
                        "name": case_name,
                        "success": False,
                        "failure_reason": f"diagnostic_run_failed: {exc!r}",
                        "query_source": "",
                        "scores": {name: 0.0 for name in CORE_ANNOTATION_NAMES},
                        "expectation_score": 0.0,
                        "composite_score": 0.0,
                        "span_id": "",
                        "trace_id": "",
                    }
                )
                continue

            summary = wait_for_trace_readback(
                session_id=result.get("session_id", ""),
                trace_id=result.get("trace_id", ""),
                span_id=result.get("root_span_id", ""),
                attempts=6,
                poll_interval_seconds=5.0,
            )
            annotation_scores, missing_annotations = self._annotation_scores(summary)
            expectation_score, expectation_details = self._expectation_score(case, result)
            core_average = sum(annotation_scores.values()) / len(CORE_ANNOTATION_NAMES)
            composite_score = round((core_average * 0.85) + (expectation_score * 0.15), 3)
            success = bool(summary) and not missing_annotations

            case_results.append(
                {
                    "name": case_name,
                    "success": success,
                    "failure_reason": "" if success else "missing_trace_readback_or_annotations",
                    "query_source": (summary or {}).get("query_source", ""),
                    "scores": annotation_scores,
                    "expectation_score": expectation_score,
                    "expectation_details": expectation_details,
                    "composite_score": composite_score,
                    "span_id": result.get("root_span_id", ""),
                    "trace_id": result.get("trace_id", ""),
                    "session_id": result.get("session_id", ""),
                    "missing_annotations": missing_annotations,
                }
            )

        successful_cases = [case for case in case_results if case.get("success")]
        divisor = len(successful_cases) or 1
        aggregate_scores = {
            name: round(sum(case["scores"][name] for case in successful_cases) / divisor, 3)
            for name in CORE_ANNOTATION_NAMES
        }
        aggregate_scores["expectation_score"] = round(
            sum(float(case.get("expectation_score", 0.0)) for case in successful_cases) / divisor,
            3,
        )
        aggregate_scores["composite_score"] = round(
            sum(float(case.get("composite_score", 0.0)) for case in successful_cases) / divisor,
            3,
        )
        aggregate_scores["success_rate"] = round(len(successful_cases) / len(case_results), 3) if case_results else 0.0
        aggregate_scores["mcp_readback_rate"] = round(
            sum(1 for case in successful_cases if case.get("query_source") == "mcp") / divisor,
            3,
        )
        valid = bool(case_results) and len(successful_cases) == len(case_results) and aggregate_scores["mcp_readback_rate"] == 1.0

        failing_names = [case["name"] for case in case_results if not case.get("success")]
        case_fragments = [
            (
                f"{case['name']}: composite={case['composite_score']:.3f}, "
                f"expectation={case['expectation_score']:.3f}, query={case.get('query_source') or 'none'}"
            )
            for case in case_results
        ]
        analysis = (
            "Held-out suite passed via live ADK runs, Phoenix eval annotations, and Phoenix MCP readback."
            if valid
            else "Held-out suite did not meet the promotion gate."
        )
        if failing_names:
            analysis += " Failed cases: " + ", ".join(failing_names) + "."
        if case_fragments:
            analysis += " Case breakdown: " + " | ".join(case_fragments)

        return PromptEvaluationSuite(
            scores=aggregate_scores,
            case_results=case_results,
            valid=valid,
            analysis=analysis,
            source_span_ids=[str(case.get("span_id", "")) for case in successful_cases if case.get("span_id")],
        )

    def _promotion_decision(
        self,
        baseline_suite: PromptEvaluationSuite,
        candidate_suite: PromptEvaluationSuite,
    ) -> tuple[bool, str]:
        if not baseline_suite.valid:
            return False, f"Baseline held-out suite is invalid. {baseline_suite.analysis}"
        if not candidate_suite.valid:
            return False, f"Candidate held-out suite is invalid. {candidate_suite.analysis}"

        regressions = [
            metric
            for metric in PROMOTION_GUARD_METRICS
            if candidate_suite.scores.get(metric, 0.0) + 1e-9 < baseline_suite.scores.get(metric, 0.0)
        ]
        if regressions:
            return False, "Candidate regressed on guarded metrics: " + ", ".join(regressions)

        composite_delta = candidate_suite.scores.get("composite_score", 0.0) - baseline_suite.scores.get("composite_score", 0.0)
        if composite_delta < PROMOTION_MIN_COMPOSITE_DELTA:
            return False, (
                f"Candidate composite improvement ({composite_delta:.3f}) did not clear the minimum delta "
                f"of {PROMOTION_MIN_COMPOSITE_DELTA:.2f}."
            )

        improved_metrics = [
            metric
            for metric in PROMOTION_GUARD_METRICS
            if candidate_suite.scores.get(metric, 0.0) - baseline_suite.scores.get(metric, 0.0) >= PROMOTION_MIN_METRIC_DELTA
        ]
        if not improved_metrics:
            return False, (
                "Candidate did not materially improve any guarded metric by at least "
                f"{PROMOTION_MIN_METRIC_DELTA:.2f}."
            )

        return True, "Candidate improved the held-out suite without any metric regressions."

    def _history_body(
        self,
        *,
        timestamp: str,
        baseline: str,
        candidate_prompt: str,
        trace_analysis: str,
        baseline_suite: PromptEvaluationSuite,
        candidate_suite: PromptEvaluationSuite,
        promotion_analysis: str,
    ) -> str:
        def _score_lines(label: str, suite: PromptEvaluationSuite) -> list[str]:
            lines = [f"## {label} suite", suite.analysis, ""]
            for key, value in suite.scores.items():
                lines.append(f"- {key}: {value:.3f}")
            if suite.case_results:
                lines.append("")
                lines.append(f"### {label} cases")
                for case in suite.case_results:
                    lines.append(
                        "- "
                        + f"{case['name']} | success={case.get('success')} | composite={case.get('composite_score', 0.0):.3f} "
                        + f"| expectation={case.get('expectation_score', 0.0):.3f} | query={case.get('query_source') or 'none'}"
                    )
            lines.append("")
            return lines

        return "\n".join(
            [
                f"# Prompt optimization {timestamp}",
                "",
                "## Trace analysis",
                trace_analysis,
                "",
                "## Promotion decision",
                promotion_analysis,
                "",
                *_score_lines("Baseline", baseline_suite),
                *_score_lines("Candidate", candidate_suite),
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
        )

    def record_successful_diagnosis(
        self,
        *,
        packet_id: str = "",
        session_id: str = "",
    ) -> dict[str, Any]:
        interval = max(1, self.settings.auto_prompt_improvement_every_n_diagnoses)
        if not self.settings.auto_prompt_improvement_enabled:
            return self.latest_snapshot()

        should_run = False
        target_count = 0
        started_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        with self._state_lock:
            state = self._load_state()
            successful_count = int(state.get("successful_diagnosis_count", 0) or 0) + 1
            last_auto_run_count = int(state.get("last_auto_run_diagnosis_count", 0) or 0)
            state.update(
                {
                    "successful_diagnosis_count": successful_count,
                    "last_trigger_packet_id": packet_id,
                    "last_trigger_session_id": session_id,
                }
            )
            diagnoses_since_last_run = successful_count - last_auto_run_count
            if state.get("status") != "running" and diagnoses_since_last_run >= interval:
                should_run = True
                target_count = successful_count
                state.update(
                    {
                        "status": "running",
                        "last_started_at": started_at,
                        "last_error": "",
                    }
                )
            self._save_state(state)

        if not should_run:
            return self.latest_snapshot()

        try:
            candidate_prompt, improvement_run = self.optimize()
        except Exception as exc:
            failure_run = ImprovementRun(
                run_id=str(uuid.uuid4()),
                baseline_prompt_version=self.settings.prompt_version,
                candidate_prompt_version=f"{self.settings.prompt_version}-candidate",
                source_span_ids=[],
                baseline_scores={},
                candidate_scores={},
                promoted=False,
                analysis=f"Automatic prompt improvement failed: {exc!r}",
            )
            self._persist_snapshot(
                candidate_prompt="",
                improvement_run=failure_run,
                status="idle",
                last_error=str(exc),
                extra={
                    "last_completed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                    "last_auto_run_diagnosis_count": target_count,
                },
            )
            return self.latest_snapshot()

        self._persist_snapshot(
            candidate_prompt=candidate_prompt,
            improvement_run=improvement_run,
            status="idle",
            last_error="",
            extra={
                "last_completed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "last_auto_run_diagnosis_count": target_count,
            },
        )
        return self.latest_snapshot()

    def optimize(self) -> tuple[str, ImprovementRun]:
        baseline = self._load_prompt(self.settings.prompt_path)
        traces = query_low_scoring_trace_summaries(settings=self.settings)
        candidate_prompt, trace_analysis = self._build_candidate(baseline, traces)
        cases = self._load_held_out_cases()
        baseline_suite = self._run_prompt_suite(baseline, cases)
        candidate_suite = self._run_prompt_suite(candidate_prompt, cases)
        promoted, promotion_analysis = self._promotion_decision(baseline_suite, candidate_suite)
        analysis = trace_analysis + "\n\n" + promotion_analysis + "\n\n" + candidate_suite.analysis

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        history_path = self.settings.prompt_history_dir / f"prompt_{timestamp}.md"
        history_path.write_text(
            self._history_body(
                timestamp=timestamp,
                baseline=baseline,
                candidate_prompt=candidate_prompt,
                trace_analysis=trace_analysis,
                baseline_suite=baseline_suite,
                candidate_suite=candidate_suite,
                promotion_analysis=promotion_analysis,
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
            source_span_ids=[trace["span_id"] for trace in traces] + candidate_suite.source_span_ids,
            baseline_scores=baseline_suite.scores,
            candidate_scores=candidate_suite.scores,
            promoted=promoted,
            analysis=analysis,
        )
        self._persist_snapshot(
            candidate_prompt=candidate_prompt,
            improvement_run=improvement_run,
            status="idle",
            last_error="",
        )
        return candidate_prompt, improvement_run
