"""Scoring and report scaffolding."""

from __future__ import annotations

from dataclasses import dataclass

from job_rejection_agent.domain import EvidenceGap, JobRequirements, ProvenanceNote, ResumeFacts
from job_rejection_agent.ingestion.rejection_notes import RejectionSignals

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
    top_gaps: list[EvidenceGap]
    provenance: list[ProvenanceNote]
    recommended_decision: str
    narrative_summary: str


def _score_ats(resume_facts: ResumeFacts, requirements: JobRequirements) -> float:
    score = 10.0
    score -= min(len(resume_facts.ats_findings), 4) * 1.2
    if requirements.ats_checks:
        score -= 0.4
    return round(max(3.0, score), 1)


def _score_evidence(matched_skills: list[str], missing_skills: list[str], under_evidenced_skills: list[str], required_count: int) -> float:
    if required_count == 0:
        return 7.0
    match_ratio = len(matched_skills) / required_count
    missing_penalty = min(len(missing_skills) * 0.9, 4.0)
    evidence_penalty = min(len(under_evidenced_skills) * 0.5, 2.5)
    score = (match_ratio * 10.0) - missing_penalty - evidence_penalty
    return round(max(2.0, min(score, 9.8)), 1)


def _recommend(overall: float, level_score: float, missing_count: int) -> str:
    if overall >= 7.8 and missing_count <= 1:
        return "apply_now"
    if overall >= 6.0 and level_score >= 5.5:
        return "apply_after_patch"
    if level_score < 4.5 or missing_count >= 4:
        return "defer"
    return "not_fit"


def score_resume_match(
    resume_facts: ResumeFacts,
    requirements: JobRequirements,
    rejection_signals: RejectionSignals,
) -> ScoreBundle:
    matched_skills, missing_skills, under_evidenced_skills, gaps, provenance = build_gap_inventory(
        resume_facts,
        requirements,
        rejection_signals,
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
    score_ats = _score_ats(resume_facts, requirements)
    score_evidence = _score_evidence(
        matched_skills,
        missing_skills,
        under_evidenced_skills,
        max(1, len(requirements.required_skills)),
    )
    score_level_fit = round(level_assessment.score, 1)
    score_overall = round((score_ats * 0.25) + (score_evidence * 0.45) + (score_level_fit * 0.30), 1)
    decision = _recommend(score_overall, score_level_fit, len(missing_skills))
    summary = (
        f"Overall fit is {score_overall}/10. The strongest alignment is in "
        f"{', '.join(matched_skills[:3]) or 'general technical foundation'}, "
        f"but the biggest rejection risk comes from {gaps[0].title.lower() if gaps else 'thin evidence'}."
    )
    return ScoreBundle(
        score_overall=score_overall,
        score_ats=score_ats,
        score_evidence=score_evidence,
        score_level_fit=score_level_fit,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        under_evidenced_skills=under_evidenced_skills,
        ats_findings=resume_facts.ats_findings,
        top_gaps=gaps[:6],
        provenance=provenance[:8],
        recommended_decision=decision,
        narrative_summary=summary,
    )
