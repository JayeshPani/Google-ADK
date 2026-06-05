"""Job description requirement extraction."""

from __future__ import annotations

import re

from job_rejection_agent.domain import JobRequirements
from job_rejection_agent.ingestion.jd_parser import ParsedJobDescription

from .constants import TECH_SKILL_LEXICON, normalize_skill_name


LEVEL_TOKENS = [
    ("staff", "staff"),
    ("principal", "staff"),
    ("senior", "senior"),
    ("lead", "senior"),
    ("mid", "mid"),
    ("2+ years", "junior"),
    ("1+ years", "entry"),
    ("entry", "entry"),
    ("new grad", "entry"),
    ("intern", "student"),
]

_KNOWN_HARD_SKILLS = {normalize_skill_name(skill) for skill in TECH_SKILL_LEXICON}
_CONTACT_PATTERN = re.compile(r"(@|https?://|www\.|linkedin|github|portfolio|\+?\d[\d\s().-]{7,}\d)", re.IGNORECASE)
_SOFT_REQUIREMENT_PATTERN = re.compile(
    r"\b(ability to|able to|coordinate|manage tasks?|multiple activities|prepare presentations?|"
    r"prepare data summaries?|communication|communicate|collaborate|stakeholders?|organized|"
    r"attention to detail|work independently|team player|time management|prioriti[sz]e)\b",
    re.IGNORECASE,
)
_HARD_SKILL_ALIASES = {
    "rest apis": "rest api",
    "restful api": "rest api",
    "restful apis": "rest api",
    "apis": "rest api",
    "llms": "llm",
    "large language model": "llm",
    "large language models": "llm",
    "google cloud platform": "gcp",
}


def _clean_requirement_text(value: str) -> str:
    return " ".join(value.strip(" -*•.:;").split()).strip()


def _canonical_hard_skill(value: str) -> str:
    text = _clean_requirement_text(value).lower()
    if not text or _CONTACT_PATTERN.search(text):
        return ""
    if text in _HARD_SKILL_ALIASES:
        return _HARD_SKILL_ALIASES[text]
    normalized = normalize_skill_name(text)
    if normalized in _KNOWN_HARD_SKILLS:
        return normalized
    for skill in sorted(TECH_SKILL_LEXICON, key=len, reverse=True):
        pattern = r"(?<![a-z0-9+.#-])" + re.escape(skill) + r"(?![a-z0-9+.#-])"
        if re.search(pattern, text):
            return normalize_skill_name(skill)
    return ""


def _is_soft_requirement(value: str) -> bool:
    text = _clean_requirement_text(value)
    return bool(text and not _CONTACT_PATTERN.search(text) and _SOFT_REQUIREMENT_PATTERN.search(text))


def split_hard_and_soft_requirements(values: list[str]) -> tuple[list[str], list[str]]:
    hard: list[str] = []
    soft: list[str] = []
    hard_seen: set[str] = set()
    soft_seen: set[str] = set()
    for value in values:
        cleaned = _clean_requirement_text(value)
        if not cleaned or _CONTACT_PATTERN.search(cleaned):
            continue
        canonical = _canonical_hard_skill(cleaned)
        if canonical and canonical not in hard_seen:
            hard_seen.add(canonical)
            hard.append(canonical)
            continue
        if _is_soft_requirement(cleaned):
            key = cleaned.lower()
            if key not in soft_seen:
                soft_seen.add(key)
                soft.append(cleaned[:180])
    return sorted(hard), soft[:8]


def _extract_title_and_company(lines: list[str]) -> tuple[str, str]:
    if not lines:
        return "Unknown Role", "Unknown Company"
    first_line = lines[0]
    if " at " in first_line.lower():
        pieces = re.split(r"\bat\b", first_line, maxsplit=1, flags=re.IGNORECASE)
        return pieces[0].strip(" -"), pieces[1].strip(" -")
    if len(lines) > 1 and len(first_line.split()) <= 8:
        return first_line, lines[1]
    return first_line, "Unknown Company"


def _extract_skills(text: str) -> tuple[list[str], list[str], list[str]]:
    lowered = text.lower()
    required: set[str] = set()
    preferred: set[str] = set()
    all_keywords: set[str] = set()
    for skill in TECH_SKILL_LEXICON:
        normalized = normalize_skill_name(skill)
        if skill in lowered:
            all_keywords.add(normalized)
            idx = lowered.find(skill)
            window = lowered[max(0, idx - 80): idx + 80]
            if any(token in window for token in ("preferred", "plus", "nice to have", "bonus")):
                preferred.add(normalized)
            else:
                required.add(normalized)
    return sorted(required), sorted(preferred - required), sorted(all_keywords)


def _extract_responsibilities(parsed: ParsedJobDescription) -> list[str]:
    items = [line.strip(" -*•") for line in parsed.bullets if len(line.split()) > 4]
    if items:
        return items[:8]
    fallback = [line for line in parsed.lines if len(line.split()) > 8]
    return fallback[:6]


def _extract_soft_requirements(responsibilities: list[str]) -> list[str]:
    _, soft_requirements = split_hard_and_soft_requirements(responsibilities)
    return soft_requirements


def _extract_level(text: str) -> str:
    lowered = text.lower()
    if "new grad" in lowered or "entry-level" in lowered or "entry level" in lowered:
        return "entry"
    if "intern" in lowered:
        return "student"
    title_window = " ".join(lowered.splitlines()[:2])[:180]
    senior_title_patterns = (
        r"\bstaff\b.{0,24}\b(engineer|developer|scientist|analyst)\b",
        r"\bprincipal\b.{0,24}\b(engineer|developer|scientist|analyst)\b",
        r"\bsenior\b.{0,24}\b(engineer|developer|scientist|analyst)\b",
        r"\blead\b.{0,24}\b(engineer|developer|scientist|analyst)\b",
    )
    for pattern, level in zip(senior_title_patterns, ("staff", "staff", "senior", "senior")):
        if re.search(pattern, title_window):
            return level
    for token, level in LEVEL_TOKENS:
        if token in {"staff", "principal", "senior", "lead", "intern", "entry", "new grad"}:
            continue
        if token in lowered:
            return level
    match = re.search(r"(\d+)\+?\s+years", lowered)
    if match:
        years = int(match.group(1))
        if years >= 5:
            return "senior"
        if years >= 3:
            return "mid"
        if years >= 1:
            return "junior"
    return "entry"


def extract_job_requirements(parsed_job_description: ParsedJobDescription) -> JobRequirements:
    role_title, company_name = _extract_title_and_company(parsed_job_description.lines)
    required_skills, preferred_skills, keywords = _extract_skills(parsed_job_description.normalized_text)
    responsibilities = _extract_responsibilities(parsed_job_description)
    soft_requirements = _extract_soft_requirements(responsibilities)
    role_summary = responsibilities[0] if responsibilities else parsed_job_description.lines[0] if parsed_job_description.lines else ""
    ats_checks = []
    if len(required_skills) >= 6:
        ats_checks.append("This JD is keyword-heavy; the resume needs explicit terminology alignment.")
    if "internship" in parsed_job_description.normalized_text.lower():
        ats_checks.append("Internship or student language detected; role may value clear project evidence over tenure.")
    return JobRequirements(
        role_title=role_title,
        company_name=company_name,
        role_summary=role_summary,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        keywords=keywords,
        responsibilities=responsibilities,
        soft_requirements=soft_requirements,
        experience_level=_extract_level(parsed_job_description.normalized_text),
        ats_checks=ats_checks,
    )
