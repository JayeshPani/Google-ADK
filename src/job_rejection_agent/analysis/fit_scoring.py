"""Scoring and report scaffolding."""

from __future__ import annotations

from dataclasses import dataclass

from job_rejection_agent.domain import ATSCheckResult, EvidenceGap, JobRequirements, ProvenanceNote, ResumeFacts
from job_rejection_agent.ingestion.rejection_notes import RejectionSignals

from .ats_checker import evaluate_ats_checks
from .evidence_mapper import build_gap_inventory
from .level_fit import assess_level_fit


@dataclass(slots=True)
class ScoreBundle:
    score_overall: float
    score_ats: float
    score_evidence: float
    score_level_fit: float
    matched_skills: list[str]
    missing_skills: list[str]
    under_evidenced_skills: list[str]
    ats_findings: list[str]
    ats_checks: list[ATSCheckResult]
    top_gaps: list[EvidenceGap]
    provenance: list[ProvenanceNote]
    recommended_decision: str
    narrative_summary: str
    scoring_source: str = "deterministic"
    score_rationale: dict[str, str] | None = None


def _score_ats(ats_checks: list[ATSCheckResult]) -> float:
    score = 10.0
    fail_count = sum(1 for item in ats_checks if item.status == "fail")
    warn_count = sum(1 for item in ats_checks if item.status == "warn")
    score -= min(fail_count * 2.0, 5.0)
    score -= min(warn_count * 0.8, 3.2)
    return round(max(3.0, score), 1)


def _score_evidence(
    matched_skills: list[str],
    missing_skills: list[str],
    under_evidenced_skills: list[str],
    required_count: int,
    resume_facts: ResumeFacts,
) -> float:
    if required_count == 0:
        base = 6.8
    else:
        match_ratio = len(matched_skills) / required_count
        missing_ratio = len(missing_skills) / required_count
        under_ratio = len(under_evidenced_skills) / max(1, len(matched_skills))
        base = 4.0 + (match_ratio * 4.6) - min(missing_ratio * 2.4, 2.4) - min(under_ratio * 1.0, 1.0)
    project_bonus = 0.5 if resume_facts.projects else 0.0
    metric_bonus = 0.7 if resume_facts.metrics else 0.0
    experience_bonus = 0.3 if resume_facts.experiences else 0.0
    score = base + project_bonus + metric_bonus + experience_bonus
    if required_count > 0 and missing_skills:
        missing_ratio = len(missing_skills) / required_count
        score = min(score, 7.4 if missing_ratio > 0.3 else 8.4)
    return round(max(2.0, min(score, 9.8)), 1)


def _recommend(overall: float, evidence_score: float, level_score: float, missing_count: int, required_count: int) -> str:
    missing_ratio = missing_count / max(1, required_count)
    if level_score <= 4.0 and overall < 6.0:
        return "not_fit"
    if missing_ratio >= 0.6 and overall < 6.5:
        return "defer"
    if overall >= 8.0 and missing_ratio <= 0.15 and level_score >= 7.5 and evidence_score >= 7.2:
        return "apply_now"
    if overall >= 6.5 and missing_ratio <= 0.4 and level_score >= 5.0:
        return "apply_after_patch"
    if overall >= 5.8 and missing_count <= 1 and level_score >= 6.0:
        return "apply_after_patch"
    return "defer"


def score_resume_match(
    resume_facts: ResumeFacts,
    requirements: JobRequirements,
    rejection_signals: RejectionSignals,
) -> ScoreBundle:
    ats_checks = evaluate_ats_checks(resume_facts, requirements)
    matched_skills, missing_skills, under_evidenced_skills, gaps, provenance = build_gap_inventory(
        resume_facts,
        requirements,
        rejection_signals,
        ats_checks=ats_checks,
    )
    level_assessment = assess_level_fit(resume_facts, requirements)
    if level_assessment.gap:
        gaps.insert(0, level_assessment.gap)
        provenance.append(
            ProvenanceNote(
                label="Level-fit heuristic",
                evidence=level_assessment.narrative,
                source="heuristic",
            )
        )
    score_ats = _score_ats(ats_checks)
    score_evidence = _score_evidence(
        matched_skills,
        missing_skills,
        under_evidenced_skills,
        len(requirements.required_skills),
        resume_facts,
    )
    score_level_fit = round(level_assessment.score, 1)
    score_overall = round((score_ats * 0.25) + (score_evidence * 0.45) + (score_level_fit * 0.30), 1)
    decision = _recommend(score_overall, score_evidence, score_level_fit, len(missing_skills), len(requirements.required_skills))
    summary = (
        f"Overall fit is {score_overall}/10. The strongest alignment is in "
        f"{', '.join(matched_skills[:3]) or 'general technical foundation'}, "
        f"but the biggest rejection risk comes from {gaps[0].title.lower() if gaps else 'thin evidence'}."
    )
    rationale = {
        "overall": "Weighted blend of ATS readiness, hard-skill evidence, and seniority alignment.",
        "ats": "Deterministic pass/warn/fail checks over structure, contact signals, readability, keywords, and file hygiene.",
        "evidence": "Hard-skill coverage plus project, experience, and metric evidence; soft requirements are excluded from missing-skill penalties.",
        "level_fit": level_assessment.narrative,
        "decision": f"Decision uses hard missing-skill ratio, evidence score, and level fit; missing hard skills={len(missing_skills)}.",
    }
    return ScoreBundle(
        score_overall=score_overall,
        score_ats=score_ats,
        score_evidence=score_evidence,
        score_level_fit=score_level_fit,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        under_evidenced_skills=under_evidenced_skills,
        ats_findings=[item.details for item in ats_checks if item.status != "pass"],
        ats_checks=ats_checks,
        top_gaps=gaps[:6],
        provenance=provenance[:8],
        recommended_decision=decision,
        narrative_summary=summary,
        score_rationale=rationale,
    )
