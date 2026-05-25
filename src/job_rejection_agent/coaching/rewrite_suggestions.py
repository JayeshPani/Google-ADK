"""Exact rewrite suggestions anchored in detected gaps."""

from __future__ import annotations

from dataclasses import dataclass

from job_rejection_agent.domain import DiagnosticReport, JobRequirements, ResumeFacts, RewritePatch


@dataclass(slots=True)
class RewritePackage:
    exact_edits: list[RewritePatch]
    project_reframes: list[str]


def _build_summary_patch(resume_facts: ResumeFacts, requirements: JobRequirements) -> RewritePatch:
    target_skills = ", ".join(requirements.required_skills[:3]) or "software engineering"
    current = resume_facts.summary or "No summary detected."
    rewritten = (
        f"{resume_facts.inferred_level.replace('_', ' ').title()} candidate targeting {requirements.role_title} roles "
        f"with hands-on work in {target_skills}; strongest evidence comes from {resume_facts.projects[0] if resume_facts.projects else 'academic and project work'}, "
        "and each bullet should quantify shipped impact or measurable outcomes."
    )
    return RewritePatch(
        section="summary",
        original_text=current,
        rewritten_text=rewritten,
        reason="Align the top-line pitch with the role and make the strongest evidence legible immediately.",
        confidence=0.84,
    )


def _rewrite_project_line(project_line: str, missing_or_weak_skill: str) -> RewritePatch:
    rewritten = (
        f"{project_line} | Reframe as: Built and iterated this project with emphasis on {missing_or_weak_skill}, "
        "then add a concrete result placeholder such as [insert latency, accuracy, user, or throughput impact]."
    )
    return RewritePatch(
        section="project",
        original_text=project_line,
        rewritten_text=rewritten,
        reason=f"Make {missing_or_weak_skill} recruiter-visible and force the bullet to carry outcome evidence.",
        confidence=0.79,
    )


def _build_skill_patch(skill: str) -> RewritePatch:
    return RewritePatch(
        section="skills",
        original_text="Skills section lacks explicit alignment.",
        rewritten_text=f"Add `{skill}` only if it was genuinely used in a real project, internship, or coursework artifact.",
        reason="Close the keyword gap without fabricating experience.",
        confidence=0.72,
    )


def generate_rewrite_package(
    resume_facts: ResumeFacts,
    requirements: JobRequirements,
    report: DiagnosticReport,
) -> RewritePackage:
    patches: list[RewritePatch] = [_build_summary_patch(resume_facts, requirements)]
    project_reframes: list[str] = []
    project_lines = resume_facts.projects or resume_facts.experiences or ["Your strongest project bullet"]

    for skill in report.under_evidenced_skills[:2]:
        patches.append(_rewrite_project_line(project_lines[min(len(patches) - 1, len(project_lines) - 1)], skill))
        project_reframes.append(
            f"Translate project work into recruiter language for `{skill}`: action -> stack -> result -> business relevance."
        )

    for skill in report.missing_skills[:2]:
        patches.append(_build_skill_patch(skill))
        project_reframes.append(
            f"If you have coursework or a side project using `{skill}`, rename the bullet so the keyword appears in the first 12 words."
        )

    if not project_reframes:
        project_reframes.append("Tighten the strongest project bullet so the first sentence says what you built, for whom, and what changed.")

    return RewritePackage(exact_edits=patches[:5], project_reframes=project_reframes[:4])

