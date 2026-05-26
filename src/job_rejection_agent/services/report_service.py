"""Rendering helpers for Streamlit and scripts."""

from __future__ import annotations

from job_rejection_agent.domain import SavedJobPacket


def render_packet_markdown(packet: SavedJobPacket) -> str:
    report = packet.report
    provenance_lines = [f"- {note.label}: {note.evidence}" for note in report.provenance[:5]]
    reframe_lines = [f"- {item}" for item in report.project_reframes]
    interview_lines = [f"- {item}" for item in report.interview_questions]
    ats_lines = [f"- {item.title}: {item.details}" for item in report.ats_checks if item.status != "pass"]
    rewritten_summary = report.rewritten_resume.summary.items[0] if report.rewritten_resume and report.rewritten_resume.summary.items else ""
    return "\n".join(
        [
            report.to_markdown(),
            "",
            "### ATS checks",
            *(ats_lines or ["- No major ATS issues detected."]),
            "",
            "### Project reframes",
            *reframe_lines,
            "",
            "### Rewritten summary",
            f"- {rewritten_summary}" if rewritten_summary else "- Rewritten summary not generated.",
            "",
            "### Interview prep",
            *interview_lines,
            "",
            "### Provenance",
            *provenance_lines,
        ]
    )


def summarise_packet(packet: SavedJobPacket) -> dict[str, str | float]:
    report = packet.report
    return {
        "packet_id": packet.packet_id,
        "role_title": packet.job_requirements.role_title,
        "company_name": packet.job_requirements.company_name,
        "score_overall": report.score_overall,
        "score_ats": report.score_ats,
        "score_evidence": report.score_evidence,
        "score_level_fit": report.score_level_fit,
        "decision": report.recommended_decision,
    }
