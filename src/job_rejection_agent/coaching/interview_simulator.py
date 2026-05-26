"""Interactive interview simulation helpers."""

from __future__ import annotations

from dataclasses import replace
import uuid

from job_rejection_agent.domain import (
    DiagnosticReport,
    InterviewFeedback,
    InterviewSimulationSession,
    InterviewTurn,
    SavedJobPacket,
)


def start_interview_session(packet: SavedJobPacket) -> InterviewSimulationSession:
    questions = packet.report.interview_questions[:5] or [
        "Walk me through the strongest project on your resume and explain what changed because of your work."
    ]
    return InterviewSimulationSession(
        session_id=str(uuid.uuid4()),
        packet_id=packet.packet_id,
        user_id=packet.user_id,
        questions=questions,
    )


def _score_answer(packet: SavedJobPacket, question: str, answer: str) -> InterviewFeedback:
    normalized = " ".join(answer.split()).strip()
    answer_lower = normalized.lower()
    projects = [project.lower() for project in packet.resume_facts.projects]
    matched_skills = [skill.lower() for skill in packet.report.matched_skills[:4]]
    gap_skills = [skill.lower() for skill in packet.report.under_evidenced_skills[:3]]
    evidence_score = 10.0 if any(project in answer_lower for project in projects) else 6.0
    if any(skill in answer_lower for skill in matched_skills):
        evidence_score += 1.0
    evidence_score = min(evidence_score, 10.0)

    word_count = len(normalized.split())
    if word_count >= 80:
        clarity_score = 8.8
    elif word_count >= 40:
        clarity_score = 7.4
    elif word_count >= 20:
        clarity_score = 6.2
    else:
        clarity_score = 4.8

    relevance_hits = 0
    for token in (packet.job_requirements.role_title, question):
        token_lower = token.lower()
        if any(word in answer_lower for word in token_lower.split()[:4]):
            relevance_hits += 1
    relevance_score = 6.0 + min(relevance_hits * 1.7, 3.4)

    gap_hits = sum(1 for skill in gap_skills if skill in answer_lower)
    metric_hit = any(char.isdigit() for char in normalized)
    gap_score = 5.5 + min(gap_hits * 1.4, 2.8) + (1.2 if metric_hit else 0.0)
    gap_score = min(gap_score, 10.0)

    overall_score = round((evidence_score * 0.35) + (clarity_score * 0.2) + (relevance_score * 0.2) + (gap_score * 0.25), 1)

    suggested_evidence = []
    suggested_evidence.extend(packet.resume_facts.metrics[:2])
    for gap in packet.report.top_gaps[:2]:
        if gap.supporting_evidence:
            suggested_evidence.append(gap.supporting_evidence[0])
    suggested_evidence = suggested_evidence[:3]

    feedback_parts = []
    if not any(project in answer_lower for project in projects):
        feedback_parts.append("Name a concrete project, internship, or coursework example instead of answering abstractly.")
    if not metric_hit:
        feedback_parts.append("Add one verified metric, scope number, or measurable outcome to strengthen credibility.")
    if gap_hits == 0 and gap_skills:
        feedback_parts.append(f"Address at least one weak-evidence skill directly, such as {gap_skills[0]}.")
    if not feedback_parts:
        feedback_parts.append("Good grounding. Tighten the opening sentence and keep the outcome explicit.")

    return InterviewFeedback(
        overall_score=overall_score,
        evidence_score=round(evidence_score, 1),
        clarity_score=round(clarity_score, 1),
        relevance_score=round(relevance_score, 1),
        gap_score=round(gap_score, 1),
        feedback=" ".join(feedback_parts),
        suggested_evidence=suggested_evidence,
    )


def submit_interview_answer(
    packet: SavedJobPacket,
    session: InterviewSimulationSession,
    answer: str,
) -> InterviewSimulationSession:
    if session.status == "completed":
        return session
    question = session.questions[session.current_index]
    feedback = _score_answer(packet, question, answer)
    turn = InterviewTurn(question=question, answer=answer, feedback=feedback)
    turns = [*session.turns, turn]
    next_index = session.current_index + 1
    completed = next_index >= len(session.questions)
    final_summary = session.final_summary
    if completed:
        average = round(sum(item.feedback.overall_score for item in turns) / len(turns), 1)
        final_summary = (
            f"Interview practice complete. Average score: {average}/10. "
            "Reuse your strongest evidence snippets and bring one verified metric into each core answer."
        )
    return replace(
        session,
        turns=turns,
        current_index=min(next_index, len(session.questions) - 1),
        status="completed" if completed else "in_progress",
        final_summary=final_summary,
    )


def session_overview(report: DiagnosticReport, session: InterviewSimulationSession | None) -> dict[str, float | str]:
    if session is None or not session.turns:
        return {
            "average": 0.0,
            "evidence": 0.0,
            "relevance": 0.0,
            "status": "not_started",
        }
    turns = session.turns
    average = round(sum(item.feedback.overall_score for item in turns) / len(turns), 1)
    evidence = round(sum(item.feedback.evidence_score for item in turns) / len(turns), 1)
    relevance = round(sum(item.feedback.relevance_score for item in turns) / len(turns), 1)
    return {
        "average": average,
        "evidence": evidence,
        "relevance": relevance,
        "status": session.status,
        "weak_skill": report.under_evidenced_skills[0] if report.under_evidenced_skills else "",
    }
