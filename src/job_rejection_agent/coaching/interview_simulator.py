"""Interactive interview simulation helpers."""

from __future__ import annotations

from dataclasses import replace
import json
import re
import uuid
from typing import Any

from job_rejection_agent.config import Settings, get_settings
from job_rejection_agent.domain import (
    DiagnosticReport,
    InterviewFeedback,
    InterviewSimulationSession,
    InterviewTurn,
    SavedJobPacket,
)
from job_rejection_agent.google_models import build_google_genai_client, is_resource_exhausted_error


MAX_INTERVIEW_TURNS = 5
_MIN_QUESTION_CHARS = 35
_MAX_QUESTION_CHARS = 260

_CONTACT_PATTERN = re.compile(
    r"(@|https?://|www\.|linkedin|github|portfolio|phone|envelope|globe|"
    r"\+?\d[\d\s().-]{7,}\d)",
    re.IGNORECASE,
)
_YEAR_RANGE_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}\s*[-–]\s*(?:present|current|expected|(?:19|20)\d{2})\b|expected",
    re.IGNORECASE,
)
_DIAGNOSTIC_PATTERN = re.compile(
    r"(resume reads|jd reads|job description explicitly asks|required by jd|"
    r"no strong snippets|ats|candidate level|role may value|level-fit|"
    r"rejection risk|under-evidenced|weakly evidenced)",
    re.IGNORECASE,
)
_SECTION_HEADINGS = {
    "summary",
    "education",
    "experience",
    "projects",
    "skills",
    "technical skills",
    "certifications",
    "leadership",
    "research",
}
_ACTION_TERMS = {
    "added",
    "analyzed",
    "automated",
    "built",
    "collaborated",
    "contributed",
    "created",
    "deployed",
    "designed",
    "developed",
    "implemented",
    "improved",
    "integrated",
    "led",
    "optimized",
    "owned",
    "reduced",
    "resolved",
    "shipped",
    "tested",
    "trained",
    "used",
}
_GENERIC_QUESTION_PATTERNS = (
    "tell me about yourself",
    "what are your strengths",
    "what are your weaknesses",
    "why should we hire you",
    "walk me through your resume",
)


def start_interview_session(packet: SavedJobPacket, settings: Settings | None = None) -> InterviewSimulationSession:
    question = _next_interview_question(packet, settings=settings)
    return InterviewSimulationSession(
        session_id=str(uuid.uuid4()),
        packet_id=packet.packet_id,
        user_id=packet.user_id,
        questions=[question],
    )


def _normalize_spaces(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def _normalize_for_compare(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _call_gemini_json(settings: Settings | None, prompt: str) -> dict[str, Any] | None:
    settings = settings or get_settings()
    if not settings.google_genai_enabled:
        return None
    try:
        client = build_google_genai_client(settings)
    except ImportError:
        return None
    for model_id in settings.generation_model_candidates:
        try:
            response = client.models.generate_content(model=model_id, contents=prompt)
        except Exception as exc:
            if is_resource_exhausted_error(exc):
                continue
            continue
        payload = _extract_json_payload(getattr(response, "text", "") or "")
        if payload:
            return payload
    return None


def _has_actionable_context(text: str) -> bool:
    lowered = text.lower()
    has_action = any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in _ACTION_TERMS)
    has_metric = any(char.isdigit() for char in text)
    return has_action or has_metric


def _unsafe_text(text: str) -> bool:
    lowered = text.strip().lower()
    return (
        not lowered
        or lowered in _SECTION_HEADINGS
        or bool(_CONTACT_PATTERN.search(text))
        or bool(_DIAGNOSTIC_PATTERN.search(text))
        or bool(_YEAR_RANGE_PATTERN.search(text))
    )


def _clean_evidence_snippet(value: str) -> str:
    text = _normalize_spaces(value.strip(" -*•|"))
    if _unsafe_text(text) or len(text.split()) < 5 or not _has_actionable_context(text):
        return ""
    return text[:240]


def _clean_coaching_text(value: Any) -> str:
    text = _normalize_spaces(value)
    if _unsafe_text(text):
        return ""
    return text[:260]


def _safe_text_list(value: Any, *, limit: int = 3) -> list[str]:
    candidates = value if isinstance(value, list) else [value]
    cleaned: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = _clean_coaching_text(candidate)
        key = _normalize_for_compare(text)
        if not text or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def _skill_anchor(packet: SavedJobPacket) -> str:
    skills = (
        packet.report.under_evidenced_skills
        or packet.report.missing_skills
        or packet.report.matched_skills
        or packet.job_requirements.required_skills
    )
    return skills[0] if skills else "project impact"


def _project_anchor(packet: SavedJobPacket) -> str:
    for project in packet.resume_facts.projects:
        text = _normalize_spaces(project)
        if text and not _unsafe_text(text):
            return text
    return "your strongest relevant project"


def _safe_project_titles(packet: SavedJobPacket) -> list[str]:
    titles: list[str] = []
    seen: set[str] = set()
    for project in packet.resume_facts.projects:
        title = _normalize_spaces(project)
        key = _normalize_for_compare(title)
        if not title or _unsafe_text(title) or key in seen:
            continue
        seen.add(key)
        titles.append(title)
        if len(titles) >= 4:
            break
    return titles or ["your strongest relevant project"]


def _safe_evidence_prompts(packet: SavedJobPacket, *, limit: int = 3) -> list[str]:
    prompts: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        cleaned = _clean_evidence_snippet(value)
        key = _normalize_for_compare(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            prompts.append(cleaned)

    for snippet in packet.resume_facts.metrics:
        add(snippet)
    for snippet in packet.resume_facts.experiences:
        add(snippet)
    for snippet in packet.resume_facts.section_map.get("projects", []):
        add(snippet)
    for snippet in packet.resume_facts.section_map.get("experience", []):
        add(snippet)
    for snippets in packet.resume_facts.evidence_by_skill.values():
        for snippet in snippets:
            add(snippet)
    for gap in packet.report.top_gaps:
        for snippet in gap.supporting_evidence:
            add(snippet)

    for project in packet.resume_facts.projects[:2]:
        project_name = _normalize_spaces(project)
        key = _normalize_for_compare(project_name)
        if project_name and key and key not in seen and not _unsafe_text(project_name):
            prompts.append(
                f"Prepare {project_name} as a STAR example: problem, your action, tool used, and measurable result."
            )
            seen.add(key)

    if not prompts:
        skill = _skill_anchor(packet)
        prompts.append(f"Prepare one concrete example showing {skill} with project, action, and measurable result.")
    return prompts[:limit]


def _context_for_prompt(packet: SavedJobPacket, session: InterviewSimulationSession | None = None) -> dict[str, Any]:
    return {
        "role_title": packet.job_requirements.role_title,
        "company_name": packet.job_requirements.company_name,
        "role_summary": packet.job_requirements.role_summary[:500],
        "required_skills": packet.job_requirements.required_skills[:8],
        "soft_requirements": packet.job_requirements.soft_requirements[:5],
        "matched_skills": packet.report.matched_skills[:8],
        "missing_skills": packet.report.missing_skills[:6],
        "under_evidenced_skills": packet.report.under_evidenced_skills[:6],
        "project_titles": _safe_project_titles(packet),
        "safe_evidence_prompts": _safe_evidence_prompts(packet),
        "top_gap_titles": [gap.title for gap in packet.report.top_gaps[:4]],
        "asked_questions": session.questions if session else [],
    }


def _valid_question(question: Any, *, asked_questions: list[str]) -> str:
    text = _normalize_spaces(question)
    key = _normalize_for_compare(text)
    asked_keys = {_normalize_for_compare(item) for item in asked_questions}
    if (
        len(text) < _MIN_QUESTION_CHARS
        or len(text) > _MAX_QUESTION_CHARS
        or _unsafe_text(text)
        or key in asked_keys
        or any(pattern in key for pattern in _GENERIC_QUESTION_PATTERNS)
    ):
        return ""
    if "?" not in text and not text.lower().startswith(("describe ", "tell ", "explain ", "walk ")):
        return ""
    return text


def _gemini_question(
    packet: SavedJobPacket,
    *,
    settings: Settings | None,
    session: InterviewSimulationSession | None = None,
    previous_answer: str = "",
    feedback: InterviewFeedback | None = None,
) -> str:
    context = _context_for_prompt(packet, session)
    previous_feedback = {
        "evidence_score": feedback.evidence_score if feedback else None,
        "clarity_score": feedback.clarity_score if feedback else None,
        "relevance_score": feedback.relevance_score if feedback else None,
        "gap_score": feedback.gap_score if feedback else None,
    }
    prompt = f"""
Return only JSON: {{"question": "..."}}
You are an interview coach for an early-career job seeker.
Ask exactly one interview question. Make it specific to the role, the candidate's safe project evidence, and any remaining gaps.
Do not ask generic questions. Do not repeat asked questions. Do not include contact info, links, education dates, or diagnostic wording.

Context:
{json.dumps(context, ensure_ascii=True)}

Previous answer:
{previous_answer[:1200]}

Previous feedback scores:
{json.dumps(previous_feedback)}
"""
    payload = _call_gemini_json(settings, prompt)
    if not payload:
        return ""
    return _valid_question(payload.get("question"), asked_questions=context["asked_questions"])


def _fallback_questions(packet: SavedJobPacket, feedback: InterviewFeedback | None = None) -> list[str]:
    role = packet.job_requirements.role_title or "this role"
    company = packet.job_requirements.company_name or "the company"
    project = _project_anchor(packet)
    skill = _skill_anchor(packet)
    hard_requirement = (packet.report.missing_skills or packet.job_requirements.required_skills or ["the hardest requirement"])[0]
    candidates = [
        f"For {role}, describe {project}: what problem did you solve, what did you personally own, and what result would you cite?",
        f"The role needs {skill}. Which project proves you can use it beyond coursework, and what technical decision did you make?",
        f"If an interviewer challenged your weak evidence around {skill}, what concrete example would you use to prove depth?",
        f"Pick one responsibility from {role} that is hardest to prove from your resume. What adjacent project evidence closes that concern?",
        f"Why are you ready for {company} now? Give a 60-second answer tied to project evidence rather than student labels.",
        f"Tell me about a tradeoff you made while building {project}. What option did you reject, and why was your choice better?",
        f"If you lack direct {hard_requirement} experience, what adjacent evidence would you offer and how would you make it credible?",
    ]
    if feedback:
        if feedback.evidence_score < 7:
            candidates.insert(
                0,
                f"Your last answer needed stronger evidence. Which specific project, action, and measurable result would you use to prove {skill}?",
            )
        elif feedback.clarity_score < 7:
            candidates.insert(
                0,
                f"Re-answer the same story in a tighter structure: situation, your action, result, and why it matters for {role}.",
            )
        elif feedback.gap_score < 7:
            candidates.insert(
                0,
                f"Your last answer did not fully cover the gap around {skill}. What example directly addresses that concern?",
            )
    return candidates


def _next_interview_question(
    packet: SavedJobPacket,
    *,
    settings: Settings | None = None,
    session: InterviewSimulationSession | None = None,
    previous_answer: str = "",
    feedback: InterviewFeedback | None = None,
) -> str:
    asked_questions = session.questions if session else []
    gemini_question = _gemini_question(
        packet,
        settings=settings,
        session=session,
        previous_answer=previous_answer,
        feedback=feedback,
    )
    if gemini_question:
        return gemini_question
    for question in _fallback_questions(packet, feedback):
        valid = _valid_question(question, asked_questions=asked_questions)
        if valid:
            return valid
    return "Which concrete project best proves your fit for this role, and what measurable outcome would you use to defend it?"


def _contains_project_reference(packet: SavedJobPacket, answer_lower: str) -> bool:
    for project in packet.resume_facts.projects:
        project_lower = project.lower()
        tokens = [token for token in re.split(r"[^a-z0-9]+", project_lower) if len(token) > 3]
        if project_lower and project_lower in answer_lower:
            return True
        if len(tokens) >= 2 and sum(1 for token in tokens if token in answer_lower) >= 2:
            return True
    return False


def _base_feedback(packet: SavedJobPacket, question: str, answer: str) -> InterviewFeedback:
    normalized = _normalize_spaces(answer)
    answer_lower = normalized.lower()
    matched_skills = [skill.lower() for skill in packet.report.matched_skills[:5]]
    gap_skills = [skill.lower() for skill in packet.report.under_evidenced_skills[:4]]
    required_skills = [skill.lower() for skill in packet.job_requirements.required_skills[:6]]

    project_hit = _contains_project_reference(packet, answer_lower)
    skill_hits = sum(1 for skill in matched_skills if skill and skill in answer_lower)
    gap_hits = sum(1 for skill in gap_skills if skill and skill in answer_lower)
    required_hits = sum(1 for skill in required_skills if skill and skill in answer_lower)
    metric_hit = any(char.isdigit() for char in normalized)

    evidence_score = 5.4 + (1.8 if project_hit else 0.0) + min(skill_hits * 0.5, 1.2) + (1.4 if metric_hit else 0.0)
    evidence_score = min(evidence_score, 10.0)

    word_count = len(normalized.split())
    if word_count >= 90:
        clarity_score = 8.8
    elif word_count >= 55:
        clarity_score = 8.0
    elif word_count >= 30:
        clarity_score = 6.8
    elif word_count >= 15:
        clarity_score = 5.6
    else:
        clarity_score = 4.4

    role_words = [word for word in re.split(r"[^a-z0-9]+", packet.job_requirements.role_title.lower()) if len(word) > 3]
    question_words = [word for word in re.split(r"[^a-z0-9]+", question.lower()) if len(word) > 5][:8]
    relevance_hits = sum(1 for word in {*role_words, *question_words} if word in answer_lower)
    relevance_score = min(6.0 + (required_hits * 0.6) + min(relevance_hits * 0.4, 2.2), 10.0)

    gap_score = min(5.2 + min(gap_hits * 1.4, 2.8) + (1.0 if metric_hit else 0.0) + (0.6 if project_hit else 0.0), 10.0)

    overall_score = round(
        (evidence_score * 0.35) + (clarity_score * 0.2) + (relevance_score * 0.2) + (gap_score * 0.25),
        1,
    )
    if word_count < 12:
        evidence_score = min(evidence_score, 5.4 if project_hit or metric_hit else 4.8)
        relevance_score = min(relevance_score, 5.2)
        gap_score = min(gap_score, 5.0)
        overall_score = min(
            round((evidence_score * 0.35) + (clarity_score * 0.2) + (relevance_score * 0.2) + (gap_score * 0.25), 1),
            5.1,
        )

    strengths: list[str] = []
    if project_hit:
        strengths.append("You grounded the answer in a concrete project.")
    if metric_hit:
        strengths.append("You included a measurable result or scope signal.")
    if skill_hits or required_hits:
        strengths.append("You connected the answer to role-relevant technical skills.")
    if not strengths:
        strengths.append("You gave enough context for the coach to identify the next improvement.")

    improvements: list[str] = []
    if not project_hit:
        improvements.append("Name one specific project, internship, or coursework example instead of staying abstract.")
    if not metric_hit:
        improvements.append("Add a verified metric, scope number, or result so the claim feels defensible.")
    if gap_skills and gap_hits == 0:
        improvements.append(f"Directly address the weak-evidence area around {gap_skills[0]}.")
    if word_count < 40:
        improvements.append("Expand the answer into a fuller story: problem, action, result, and role fit.")
    if not improvements:
        improvements.append("Tighten the opening sentence and make the final role-fit takeaway explicit.")

    evidence_prompts = _safe_evidence_prompts(packet)
    answer_structure = (
        "Use STAR in 60-90 seconds: situation, your task, the action you personally took, "
        "the measurable result, then one sentence tying it back to the role."
    )
    feedback_text = " ".join(improvements[:2]) if improvements else "Good grounding. Keep the outcome explicit."

    return InterviewFeedback(
        overall_score=overall_score,
        evidence_score=round(evidence_score, 1),
        clarity_score=round(clarity_score, 1),
        relevance_score=round(relevance_score, 1),
        gap_score=round(gap_score, 1),
        feedback=feedback_text,
        suggested_evidence=evidence_prompts,
        strengths=strengths[:3],
        improvements=improvements[:3],
        evidence_prompts=evidence_prompts,
        answer_structure=answer_structure,
    )


def _gemini_feedback(
    packet: SavedJobPacket,
    question: str,
    answer: str,
    base: InterviewFeedback,
    *,
    settings: Settings | None,
) -> InterviewFeedback:
    context = _context_for_prompt(packet)
    prompt = f"""
Return only JSON with keys overall_score, evidence_score, clarity_score, relevance_score, gap_score, feedback, strengths, improvements, evidence_prompts, answer_structure.
You are an interview coach. Score this answer using evidence, clarity, relevance, and gap coverage.
Do not include contact info, links, education date ranges, or raw diagnostic sentences.
Use only safe evidence prompts from context. If evidence is missing, tell the candidate what example to prepare.
Very short or vague answers should score low even if they mention a keyword.

Question:
{question}

Answer:
{answer[:1800]}

Scores:
overall={base.overall_score}, evidence={base.evidence_score}, clarity={base.clarity_score}, relevance={base.relevance_score}, gap={base.gap_score}

Safe context:
{json.dumps(context, ensure_ascii=True)}
"""
    payload = _call_gemini_json(settings, prompt)
    if not payload:
        return base

    scores = _feedback_scores_from_payload(payload, base, answer)
    feedback_text = _clean_coaching_text(payload.get("feedback")) or base.feedback
    strengths = _safe_text_list(payload.get("strengths"), limit=3) or base.strengths
    improvements = _safe_text_list(payload.get("improvements"), limit=3) or base.improvements
    evidence_prompts = _safe_text_list(payload.get("evidence_prompts"), limit=3) or base.evidence_prompts
    answer_structure = _clean_coaching_text(payload.get("answer_structure")) or base.answer_structure
    return replace(
        base,
        overall_score=scores["overall_score"],
        evidence_score=scores["evidence_score"],
        clarity_score=scores["clarity_score"],
        relevance_score=scores["relevance_score"],
        gap_score=scores["gap_score"],
        feedback=feedback_text,
        suggested_evidence=evidence_prompts,
        strengths=strengths,
        improvements=improvements,
        evidence_prompts=evidence_prompts,
        answer_structure=answer_structure,
    )


def _payload_score(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    elif isinstance(value, str):
        try:
            score = float(value.strip().removesuffix("/10"))
        except ValueError:
            return None
    else:
        return None
    if score < 0 or score > 10:
        return None
    return round(score, 1)


def _feedback_scores_from_payload(payload: dict[str, Any], base: InterviewFeedback, answer: str) -> dict[str, float]:
    keys = ("overall_score", "evidence_score", "clarity_score", "relevance_score", "gap_score")
    parsed = {key: _payload_score(payload, key) for key in keys}
    if any(value is None for value in parsed.values()):
        return {
            "overall_score": base.overall_score,
            "evidence_score": base.evidence_score,
            "clarity_score": base.clarity_score,
            "relevance_score": base.relevance_score,
            "gap_score": base.gap_score,
        }
    scores = {key: float(value) for key, value in parsed.items() if value is not None}
    if len(_normalize_spaces(answer).split()) < 12 and scores["overall_score"] > 5.5:
        return {
            "overall_score": base.overall_score,
            "evidence_score": base.evidence_score,
            "clarity_score": base.clarity_score,
            "relevance_score": base.relevance_score,
            "gap_score": base.gap_score,
        }
    return scores


def _score_answer(
    packet: SavedJobPacket,
    question: str,
    answer: str,
    *,
    settings: Settings | None = None,
) -> InterviewFeedback:
    base = _base_feedback(packet, question, answer)
    return _gemini_feedback(packet, question, answer, base, settings=settings)


def submit_interview_answer(
    packet: SavedJobPacket,
    session: InterviewSimulationSession,
    answer: str,
    settings: Settings | None = None,
) -> InterviewSimulationSession:
    if session.status == "completed":
        return session
    if not session.questions:
        session = replace(session, questions=[_next_interview_question(packet, settings=settings)])

    current_index = min(session.current_index, len(session.questions) - 1)
    question = session.questions[current_index]
    feedback = _score_answer(packet, question, answer, settings=settings)
    turn = InterviewTurn(question=question, answer=answer, feedback=feedback)
    turns = [*session.turns, turn]
    answered_count = len(turns)
    completed = answered_count >= MAX_INTERVIEW_TURNS

    questions = list(session.questions)
    next_index = current_index
    final_summary = session.final_summary
    if completed:
        average = round(sum(item.feedback.overall_score for item in turns) / len(turns), 1)
        final_summary = (
            f"Interview practice complete. Average score: {average}/10. "
            "Keep using one project, one action, and one measurable result in each core answer."
        )
    else:
        next_question = _next_interview_question(
            packet,
            settings=settings,
            session=replace(session, questions=questions, turns=turns),
            previous_answer=answer,
            feedback=feedback,
        )
        questions.append(next_question)
        next_index = len(questions) - 1

    return replace(
        session,
        questions=questions,
        turns=turns,
        current_index=next_index,
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
            "weak_skill": report.under_evidenced_skills[0] if report.under_evidenced_skills else "",
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
