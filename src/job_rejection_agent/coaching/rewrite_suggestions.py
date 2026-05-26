"""Exact rewrite suggestions anchored in detected gaps."""

from __future__ import annotations

from dataclasses import dataclass
import re

from job_rejection_agent.domain import (
    DiagnosticReport,
    JobRequirements,
    ProvenanceNote,
    ResumeFacts,
    ResumeSectionBlock,
    RewrittenResume,
    RewritePatch,
)


@dataclass(slots=True)
class RewritePackage:
    exact_edits: list[RewritePatch]
    project_reframes: list[str]


def _build_summary_patch(resume_facts: ResumeFacts, requirements: JobRequirements) -> RewritePatch:
    shared_skills = [skill for skill in requirements.required_skills if skill in resume_facts.skills]
    target_skills = ", ".join((shared_skills or resume_facts.skills)[:3]) or "software engineering"
    current = resume_facts.summary or "No summary detected."
    rewritten = (
        f"{resume_facts.inferred_level.replace('_', ' ').title()} candidate targeting {requirements.role_title} roles "
        f"with hands-on work in {target_skills}; strongest evidence comes from {resume_facts.projects[0] if resume_facts.projects else 'academic and project work'}."
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


PLACEHOLDER_TOKEN = "[confirm measurable outcome or scope]"


def _contains_metric(text: str) -> bool:
    return bool(re.search(r"\d", text))


def _ensure_placeholder(text: str) -> str:
    cleaned = text.strip()
    if PLACEHOLDER_TOKEN in cleaned or _contains_metric(cleaned):
        return cleaned
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    return f"{cleaned} {PLACEHOLDER_TOKEN}."


def _pick_section_lines(resume_facts: ResumeFacts, section_name: str, fallback_values: list[str]) -> list[str]:
    lines = resume_facts.section_map.get(section_name, [])
    if lines:
        return lines[:6]
    return [item for item in fallback_values if item][:6]


def _build_skill_block(resume_facts: ResumeFacts, requirements: JobRequirements) -> ResumeSectionBlock:
    matched = [skill for skill in resume_facts.skills if skill in requirements.keywords or skill in requirements.required_skills]
    remaining = [skill for skill in resume_facts.skills if skill not in matched]
    lines: list[str] = []
    if matched:
        lines.append("Role-aligned: " + ", ".join(matched[:8]))
    if remaining:
        lines.append("Additional: " + ", ".join(remaining[:10]))
    note = ""
    if not lines:
        note = "List only tools and technologies you can defend with real coursework, projects, or work examples."
    return ResumeSectionBlock(title="Skills", items=lines, note=note)


def _rewrite_section_items(items: list[str], *, add_placeholder: bool = False) -> list[str]:
    rewritten: list[str] = []
    for item in items:
        cleaned = " ".join(item.split()).strip()
        if not cleaned:
            continue
        rewritten.append(_ensure_placeholder(cleaned) if add_placeholder else cleaned)
    return rewritten


def _provenance_slice(report: DiagnosticReport, *sources: str) -> list[ProvenanceNote]:
    if not sources:
        return report.provenance[:3]
    selected = [note for note in report.provenance if note.source in sources]
    return selected[:4] or report.provenance[:3]


def generate_rewritten_resume(
    resume_facts: ResumeFacts,
    requirements: JobRequirements,
    report: DiagnosticReport,
    rewrite_package: RewritePackage,
) -> RewrittenResume:
    header_lines = resume_facts.header_lines[:4]
    if not header_lines:
        header_lines = ["Paste your current name and contact header exactly as it appears on the real resume."]

    summary_patch = next((patch for patch in rewrite_package.exact_edits if patch.section == "summary"), None)
    summary_text = summary_patch.rewritten_text if summary_patch else (resume_facts.summary or "")
    summary_block = ResumeSectionBlock(
        title="Summary",
        items=[summary_text] if summary_text else [],
        note="" if summary_text else "Add a one-line summary grounded only in roles, skills, and projects already on the resume.",
    )

    experience_lines = _pick_section_lines(resume_facts, "experience", resume_facts.experiences)
    project_lines = _pick_section_lines(resume_facts, "projects", resume_facts.projects)
    education_lines = _pick_section_lines(resume_facts, "education", resume_facts.education)

    experience_block = ResumeSectionBlock(
        title="Experience",
        items=_rewrite_section_items(experience_lines, add_placeholder=True),
        note="" if experience_lines else "No standalone experience section was detected; use verified project evidence instead.",
    )
    projects_block = ResumeSectionBlock(
        title="Projects",
        items=_rewrite_section_items(project_lines, add_placeholder=True),
        note="" if project_lines else "No project section was detected; add only real projects or coursework you can defend.",
    )
    education_block = ResumeSectionBlock(
        title="Education",
        items=_rewrite_section_items(education_lines),
        note="" if education_lines else "Education details were not parsed clearly from the current resume.",
    )

    ats_notes = [
        f"{check.title}: {check.recommendation}"
        for check in report.ats_checks
        if check.status != "pass"
    ]

    return RewrittenResume(
        header=ResumeSectionBlock(title="Header", items=header_lines),
        summary=summary_block,
        experience=experience_block,
        projects=projects_block,
        skills=_build_skill_block(resume_facts, requirements),
        education=education_block,
        ats_notes=ats_notes,
        provenance_map={
            "header": _provenance_slice(report, "resume"),
            "summary": _provenance_slice(report, "resume", "job_description"),
            "experience": _provenance_slice(report, "resume", "heuristic"),
            "projects": _provenance_slice(report, "resume"),
            "skills": _provenance_slice(report, "resume", "job_description"),
            "education": _provenance_slice(report, "resume"),
        },
    )
