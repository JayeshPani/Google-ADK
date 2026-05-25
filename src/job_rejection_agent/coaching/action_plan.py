"""Generate a one-week action plan."""

from __future__ import annotations

from job_rejection_agent.domain import DiagnosticReport, JobRequirements, ResumeFacts
from job_rejection_agent.ingestion.rejection_notes import RejectionSignals


def generate_action_plan(
    report: DiagnosticReport,
    resume_facts: ResumeFacts,
    requirements: JobRequirements,
    rejection_signals: RejectionSignals,
) -> list[str]:
    project_anchor = resume_facts.projects[0] if resume_facts.projects else "your strongest project"
    missing_skill = report.missing_skills[0] if report.missing_skills else "the top missing requirement"
    weak_skill = report.under_evidenced_skills[0] if report.under_evidenced_skills else "your strongest matched skill"
    feedback_anchor = rejection_signals.notes[0] if rejection_signals.notes else "No recruiter feedback was provided, so follow the evidence gaps first."
    return [
        f"Day 1: Rewrite the summary and top two bullets so `{weak_skill}` appears with a real project outcome anchored to {project_anchor}.",
        f"Day 2: Build or document a micro-proof for `{missing_skill}` and add a GitHub link or demo note only if it is real.",
        "Day 3: Cut any bullet over two lines and move the best quantified evidence into the first half of the page.",
        f"Day 4: Tailor the skills section to the JD by mirroring exact terms such as {', '.join(requirements.required_skills[:4]) or requirements.role_title}.",
        f"Day 5: Record a 90-second project walkthrough for {project_anchor}; use it to tighten the bullet wording and prep for screens.",
        f"Day 6: Re-apply only after the patched resume clears the top gaps. Current recruiter signal: {feedback_anchor}",
        "Day 7: Apply to 3 roles with similar requirements and save each diagnosis so you can compare recurring rejection patterns.",
    ]

