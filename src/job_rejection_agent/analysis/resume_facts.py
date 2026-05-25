"""Resume-to-structured-facts extraction."""

from __future__ import annotations

from collections import defaultdict
import re

from job_rejection_agent.domain import ResumeFacts
from job_rejection_agent.ingestion.resume_parser import ParsedResume

from .constants import TECH_SKILL_LEXICON, normalize_skill_name


SECTION_PATTERN = re.compile(
    r"^(summary|education|experience|projects|technical skills|skills|leadership|research|certifications)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
METRIC_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%|\b\d+(?:,\d{3})*(?:\+)?\b")


def _split_sections(text: str) -> dict[str, str]:
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return {"full_text": text}
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        title = match.group(1).strip().lower()
        sections[title] = text[start:end].strip()
    return sections


def _extract_skills(text: str) -> list[str]:
    lowered = text.lower()
    found: set[str] = set()
    for skill in TECH_SKILL_LEXICON:
        if skill in lowered:
            found.add(normalize_skill_name(skill))
    return sorted(found)


def _extract_projects(project_section: str, fallback_text: str) -> list[str]:
    source = project_section or fallback_text
    lines = [line.strip(" -*•") for line in source.splitlines() if line.strip()]
    project_candidates: list[str] = []
    for line in lines:
        if line.lower().startswith(("project", "capstone", "research")):
            project_candidates.append(line)
        elif len(line.split()) <= 8 and line[:1].isupper():
            project_candidates.append(line)
    return project_candidates[:6]


def _extract_summary(sections: dict[str, str], text: str) -> str:
    if "summary" in sections and sections["summary"]:
        return sections["summary"].splitlines()[0]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    trimmed = [line for line in lines[:5] if "@" not in line and "linkedin" not in line.lower()]
    return " ".join(trimmed[:2]).strip()


def _extract_metric_snippets(text: str) -> list[str]:
    snippets: list[str] = []
    for line in text.splitlines():
        if METRIC_PATTERN.search(line):
            snippets.append(line.strip())
    return snippets[:12]


def _infer_level(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ("phd", "research assistant", "graduate student")):
        return "student"
    if any(token in lowered for token in ("b.tech", "bachelor", "student", "cgpa", "gpa")):
        if "intern" in lowered:
            return "student"
        return "new_grad"
    if re.search(r"\b[34]\+?\s+years\b", lowered):
        return "mid"
    if re.search(r"\b[56]\+?\s+years\b", lowered):
        return "senior"
    return "new_grad"


def _collect_evidence(text: str, skills: list[str]) -> dict[str, list[str]]:
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    evidence_map: dict[str, list[str]] = defaultdict(list)
    for sentence in sentences:
        lowered = sentence.lower()
        for skill in skills:
            variants = {skill, skill.replace("ml", "machine learning")}
            if any(variant in lowered for variant in variants):
                evidence_map[skill].append(sentence.strip())
    return dict(evidence_map)


def extract_resume_facts(parsed_resume: ParsedResume) -> ResumeFacts:
    sections = _split_sections(parsed_resume.normalized_text)
    skills = _extract_skills(parsed_resume.normalized_text)
    projects = _extract_projects(sections.get("projects", ""), parsed_resume.normalized_text)
    experience_section = sections.get("experience", "")
    education_section = sections.get("education", "")
    experiences = [line.strip(" -*•") for line in experience_section.splitlines() if line.strip()][:8]
    education = [line.strip(" -*•") for line in education_section.splitlines() if line.strip()][:5]
    metrics = _extract_metric_snippets(parsed_resume.normalized_text)
    evidence = _collect_evidence(parsed_resume.normalized_text, skills)
    return ResumeFacts(
        raw_text=parsed_resume.raw_text,
        normalized_text=parsed_resume.normalized_text,
        summary=_extract_summary(sections, parsed_resume.normalized_text),
        skills=skills,
        projects=projects,
        experiences=experiences,
        education=education,
        metrics=metrics,
        inferred_level=_infer_level(parsed_resume.normalized_text),
        evidence_by_skill=evidence,
        ats_findings=parsed_resume.ats_findings,
        contact_signals=parsed_resume.contact_signals,
    )

