"""Interview-prep question generation."""

from __future__ import annotations

from job_rejection_agent.domain import DiagnosticReport, JobRequirements, ResumeFacts


def generate_interview_questions(
    report: DiagnosticReport,
    resume_facts: ResumeFacts,
    requirements: JobRequirements,
) -> list[str]:
    anchor_project = resume_facts.projects[0] if resume_facts.projects else "one of your recent projects"
    questions = [
        f"Walk me through {anchor_project} and explain what changed because of your work.",
        f"How have you used {report.under_evidenced_skills[0] if report.under_evidenced_skills else 'your main technical stack'} in a real project?",
        f"Why are you a fit for {requirements.role_title} even if you are early in your career?",
        "Which bullet on your resume best proves ownership, and how would you defend it under follow-up questioning?",
        f"If you lack direct {report.missing_skills[0] if report.missing_skills else 'production'} experience, what adjacent evidence would you offer instead?",
    ]
    return questions

