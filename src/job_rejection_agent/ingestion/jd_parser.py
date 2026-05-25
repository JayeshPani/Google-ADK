"""Job description normalization."""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from .resume_parser import normalize_text


@dataclass(slots=True)
class ParsedJobDescription:
    raw_text: str
    normalized_text: str
    lines: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)


def parse_job_description(text: str) -> ParsedJobDescription:
    normalized = normalize_text(text)
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    bullets = []
    for line in lines:
        if re.match(r"^[-*•]\s+", line) or re.match(r"^\d+\.\s+", line):
            bullets.append(line)
    return ParsedJobDescription(raw_text=text, normalized_text=normalized, lines=lines, bullets=bullets)

