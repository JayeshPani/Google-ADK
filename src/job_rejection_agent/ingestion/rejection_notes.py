"""Optional recruiter feedback parsing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RejectionSignals:
    raw_text: str
    categories: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


KEYWORDS = {
    "experience": ("experience", "more senior", "years", "depth"),
    "skills": ("skill", "stack", "tooling", "technical fit"),
    "level_fit": ("senior", "level", "scope", "ownership"),
    "communication": ("communication", "clarity", "storytelling"),
    "visa": ("visa", "work authorization", "sponsorship"),
}


def parse_rejection_notes(text: str) -> RejectionSignals:
    lowered = text.lower().strip()
    categories: list[str] = []
    notes: list[str] = []
    if not lowered:
        return RejectionSignals(raw_text=text)
    for category, tokens in KEYWORDS.items():
        if any(token in lowered for token in tokens):
            categories.append(category)
    for sentence in text.split("."):
        sentence = sentence.strip()
        if sentence:
            notes.append(sentence)
    return RejectionSignals(raw_text=text, categories=sorted(set(categories)), notes=notes)

