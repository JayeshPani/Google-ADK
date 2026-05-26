"""Gap mapping between resume evidence and job requirements."""

from __future__ import annotations

from job_rejection_agent.domain import ATSCheckResult, EvidenceGap, JobRequirements, ProvenanceNote, ResumeFacts
from job_rejection_agent.ingestion.rejection_notes import RejectionSignals


def build_gap_inventory(
    resume_facts: ResumeFacts,
    requirements: JobRequirements,
    rejection_signals: RejectionSignals,
    *,
    ats_checks: list[ATSCheckResult],
) -> tuple[list[str], list[str], list[str], list[EvidenceGap], list[ProvenanceNote]]:
    resume_skill_set = set(resume_facts.skills)
    required_skill_set = set(requirements.required_skills)
    matched_skills = sorted(resume_skill_set & required_skill_set)
    missing_skills = sorted(required_skill_set - resume_skill_set)
    under_evidenced_skills: list[str] = []
    gaps: list[EvidenceGap] = []
    provenance: list[ProvenanceNote] = []

    for skill in matched_skills:
        evidence = resume_facts.evidence_by_skill.get(skill, [])
        has_metric = any(any(char.isdigit() for char in snippet) for snippet in evidence)
        if len(evidence) < 2 or not has_metric:
            under_evidenced_skills.append(skill)
            gaps.append(
                EvidenceGap(
                    category="under_evidenced_skill",
                    title=f"{skill.title()} is present but weakly evidenced",
                    severity="medium",
                    details=f"The resume mentions {skill} but does not clearly show depth, results, or shipped outcomes.",
                    recommended_fix=f"Add a bullet that ties {skill} to a real project outcome or measurable result.",
                    supporting_evidence=evidence[:2] or [f"No strong snippets found for {skill}."],
                )
            )

    for skill in missing_skills[:5]:
        gaps.append(
            EvidenceGap(
                category="missing_skill",
                title=f"Missing keyword: {skill.title()}",
                severity="high",
                details=f"The job description explicitly asks for {skill}, but the resume never mentions it.",
                recommended_fix=f"Only add {skill} if you truly have it; otherwise build or document a relevant project before applying again.",
                supporting_evidence=[f"Required by JD for {requirements.role_title} at {requirements.company_name}."],
            )
        )

    for check in [item for item in ats_checks if item.status != "pass"][:3]:
        gaps.append(
            EvidenceGap(
                category="ats",
                title=check.title,
                severity="high" if check.status == "fail" else "medium",
                details=check.details,
                recommended_fix=check.recommendation,
                supporting_evidence=[check.details],
            )
        )

    for note in rejection_signals.notes[:2]:
        gaps.append(
            EvidenceGap(
                category="rejection_signal",
                title="Recruiter feedback signal",
                severity="high" if rejection_signals.categories else "medium",
                details=note,
                recommended_fix="Treat direct recruiter feedback as higher-priority than inferred fixes.",
                supporting_evidence=[note],
            )
        )

    for skill in matched_skills[:4]:
        snippets = resume_facts.evidence_by_skill.get(skill, [])
        if snippets:
            provenance.append(ProvenanceNote(label=f"Skill evidence: {skill}", evidence=snippets[0], source="resume"))
    for skill in missing_skills[:3]:
        provenance.append(
            ProvenanceNote(
                label=f"Missing requirement: {skill}",
                evidence=f"{skill} is present in the job requirements but absent from the resume.",
                source="job_description",
            )
        )
    for note in rejection_signals.notes[:2]:
        provenance.append(ProvenanceNote(label="Recruiter feedback", evidence=note, source="rejection_note"))
    for check in [item for item in ats_checks if item.status != "pass"][:2]:
        provenance.append(
            ProvenanceNote(
                label=f"ATS check: {check.category.replace('_', ' ')}",
                evidence=check.details,
                source="heuristic",
            )
        )

    gaps.sort(key=lambda item: {"high": 0, "medium": 1, "low": 2}[item.severity])
    return matched_skills, missing_skills, sorted(set(under_evidenced_skills)), gaps, provenance
