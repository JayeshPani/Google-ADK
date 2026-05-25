"""Prompt fragments for specialized reasoning roles."""

from __future__ import annotations


RESUME_ANALYST_BRIEF = """
Focus on extracting what the resume actually proves:
- which skills are explicitly named
- which projects show ownership
- which bullets show measurable impact
- which sections are ATS-fragile
"""

MATCH_ANALYST_BRIEF = """
Focus on mismatch diagnosis:
- missing required skills
- under-evidenced matched skills
- level-fit or scope mismatch
- recruiter-feedback contradictions
"""

COACH_BRIEF = """
Focus on fixes:
- exact line edits
- project reframes
- under-3-hour action steps
- interview questions that prepare the candidate to defend weak spots
"""


def compose_specialist_context() -> str:
    return "\n\n".join([RESUME_ANALYST_BRIEF.strip(), MATCH_ANALYST_BRIEF.strip(), COACH_BRIEF.strip()])

