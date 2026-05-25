"""Candidate-versus-role seniority estimation."""

from __future__ import annotations

from dataclasses import dataclass

from job_rejection_agent.domain import EvidenceGap, JobRequirements, ResumeFacts

from .constants import LEVEL_RANK


@dataclass(slots=True)
class LevelAssessment:
    score: float
    narrative: str
    gap: EvidenceGap | None


def assess_level_fit(resume_facts: ResumeFacts, requirements: JobRequirements) -> LevelAssessment:
    candidate_rank = LEVEL_RANK.get(resume_facts.inferred_level, 1)
    role_rank = LEVEL_RANK.get(requirements.experience_level, 1)
    if candidate_rank >= role_rank:
        return LevelAssessment(
            score=9.0,
            narrative="The candidate level appears aligned with the role's seniority requirements.",
            gap=None,
        )
    delta = role_rank - candidate_rank
    score = max(2.0, 8.5 - (delta * 2.0))
    details = (
        f"The resume reads like a {resume_facts.inferred_level.replace('_', ' ')}, "
        f"while the JD reads closer to {requirements.experience_level} scope."
    )
    gap = EvidenceGap(
        category="level_fit",
        title="Level-fit mismatch",
        severity="high" if delta >= 2 else "medium",
        details=details,
        recommended_fix="Target roles one step closer to your current evidence or add stronger ownership signals.",
        supporting_evidence=[details],
    )
    return LevelAssessment(score=score, narrative=details, gap=gap)

