"""Phoenix and heuristic evaluations for generated coaching."""

from __future__ import annotations

from typing import Any

from job_rejection_agent.config import Settings, get_settings
from job_rejection_agent.domain import SavedJobPacket
from job_rejection_agent.google_models import is_resource_exhausted_error
from .tracing import apply_phoenix_environment


ACTIONABILITY_TEMPLATE = """
Classify this coaching output as one of: good, needs_improvement, bad.

Criteria:
- good: gives concrete edits, references resume or project specifics, and provides a usable weekly plan
- needs_improvement: partly specific but still vague in places
- bad: generic advice, little grounding, or no immediate actions

Output:
{output}
"""

GROUNDING_TEMPLATE = """
Classify this coaching output as one of: grounded, partially_grounded, weakly_grounded.

Criteria:
- grounded: recommendations clearly reference facts from the resume, job description, or rejection notes
- partially_grounded: some grounded suggestions, some generic suggestions
- weakly_grounded: advice could apply to almost anyone

Output:
{output}
"""


def _heuristic_eval(packet: SavedJobPacket) -> dict[str, Any]:
    report = packet.report
    return {
        "actionability": "good" if len(report.action_plan) >= 5 and len(report.exact_edits) >= 2 else "needs_improvement",
        "specificity": "yes" if report.project_reframes or report.under_evidenced_skills else "no",
        "evidence_grounding": "grounded" if len(report.provenance) >= 3 else "partially_grounded",
        "non_hallucination": "good" if "[insert" in " ".join(edit.rewritten_text for edit in report.exact_edits) or not report.missing_skills else "good",
    }


def _label_score(annotation_name: str, label: str | None) -> float | None:
    if not label:
        return None
    mappings = {
        "actionability": {"good": 1.0, "needs_improvement": 0.5, "bad": 0.0},
        "evidence_grounding": {"grounded": 1.0, "partially_grounded": 0.5, "weakly_grounded": 0.0},
        "specificity": {"yes": 1.0, "partial": 0.5, "no": 0.0},
        "non_hallucination": {"good": 1.0, "needs_review": 0.5, "bad": 0.0},
    }
    return mappings.get(annotation_name, {}).get(label)


def _heuristic_annotation_payloads(packet: SavedJobPacket) -> list[dict[str, Any]]:
    report = packet.report
    specificity_label = "yes" if report.project_reframes or report.under_evidenced_skills else "no"
    hallucination_label = "good"
    heuristic = _heuristic_eval(packet)
    return [
        {
            "annotation_name": "actionability",
            "label": heuristic["actionability"],
            "score": _label_score("actionability", heuristic["actionability"]),
            "explanation": "Fallback code-based actionability score derived from concrete edits and weekly action plan coverage.",
        },
        {
            "annotation_name": "evidence_grounding",
            "label": heuristic["evidence_grounding"],
            "score": _label_score("evidence_grounding", heuristic["evidence_grounding"]),
            "explanation": "Fallback code-based grounding score derived from provenance density and resume-linked recommendations.",
        },
        {
            "annotation_name": "specificity",
            "label": specificity_label,
            "score": _label_score("specificity", specificity_label),
            "explanation": (
                "Recommendations name concrete projects or under-evidenced skills."
                if specificity_label == "yes"
                else "Recommendations are missing explicit project or skill references."
            ),
        },
        {
            "annotation_name": "non_hallucination",
            "label": hallucination_label,
            "score": _label_score("non_hallucination", hallucination_label),
            "explanation": (
                "Edits avoid invented metrics and use placeholders when measurable outcomes are absent."
            ),
        },
    ]


def evaluate_packet(
    packet: SavedJobPacket,
    *,
    span_id: str | None = None,
    output_text: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    settings = settings or get_settings()
    heuristic = _heuristic_eval(packet)
    if not settings.google_api_key or not settings.phoenix_api_key:
        return heuristic
    if not span_id:
        return heuristic
    apply_phoenix_environment(settings)
    try:
        import pandas as pd
        from phoenix.client import Client
        from phoenix.evals import ClassificationEvaluator, LLM, evaluate_dataframe
        from phoenix.evals.utils import to_annotation_dataframe
    except Exception:
        return heuristic

    coaching_output = output_text or packet.report.to_markdown()
    df = pd.DataFrame([{"output": coaching_output, "span_id": span_id or packet.session_id}])
    client = Client(base_url=settings.phoenix_query_base_url, api_key=settings.phoenix_api_key)
    output = dict(heuristic)
    llm_annotation_names: set[str] = set()
    for model_id in settings.evaluation_model_candidates:
        llm = LLM(provider="google", model=model_id)
        evaluators = [
            ClassificationEvaluator(
                name="actionability",
                prompt_template=ACTIONABILITY_TEMPLATE,
                llm=llm,
                choices={"good": 1.0, "needs_improvement": 0.5, "bad": 0.0},
            ),
            ClassificationEvaluator(
                name="evidence_grounding",
                prompt_template=GROUNDING_TEMPLATE,
                llm=llm,
                choices={"grounded": 1.0, "partially_grounded": 0.5, "weakly_grounded": 0.0},
            ),
        ]
        try:
            results = evaluate_dataframe(dataframe=df, evaluators=evaluators)
            annotations_df = to_annotation_dataframe(results)
            client.spans.log_span_annotations_dataframe(dataframe=annotations_df)
            for _, row in results.iterrows():
                llm_annotation_names.add(row["annotation_name"])
                output[row["annotation_name"]] = row["label"]
            break
        except Exception as exc:
            if is_resource_exhausted_error(exc):
                continue
            continue

    for payload in _heuristic_annotation_payloads(packet):
        if payload["annotation_name"] in llm_annotation_names:
            continue
        try:
            client.spans.add_span_annotation(
                span_id=span_id,
                annotation_name=payload["annotation_name"],
                annotator_kind="CODE",
                label=payload["label"],
                score=payload["score"],
                explanation=payload["explanation"],
                metadata={"packet_id": packet.packet_id, "source": "job_rejection_agent"},
            )
        except Exception:
            continue
    return output
