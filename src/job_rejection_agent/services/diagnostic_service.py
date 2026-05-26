"""Core orchestration service for diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
import uuid

from job_rejection_agent.analysis import extract_job_requirements, extract_resume_facts, score_resume_match
from job_rejection_agent.coaching import (
    generate_action_plan,
    generate_interview_questions,
    generate_rewrite_package,
    generate_rewritten_resume,
    start_interview_session,
    submit_interview_answer,
)
from job_rejection_agent.config import Settings, get_settings
from job_rejection_agent.domain import (
    DiagnosticReport,
    InterviewSimulationSession,
    JobRequirements,
    MultiJDComparison,
    MultiJDRow,
    ResumeFacts,
    SavedJobPacket,
)
from job_rejection_agent.google_models import build_google_genai_client, is_resource_exhausted_error
from job_rejection_agent.ingestion import parse_job_description, parse_rejection_notes, parse_resume_file
from job_rejection_agent.persistence import JobTracker, build_packet_repository


@dataclass(slots=True)
class DiagnosticSessionResult:
    packet: SavedJobPacket
    report_markdown: str
    eval_scores: dict[str, Any] = field(default_factory=dict)
    used_llm_augmentation: bool = False


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class GeminiAugmenter:
    """Optional Gemini refinements layered on top of deterministic extraction."""

    _REPRESENTATIVE_KEYS = (
        "name",
        "title",
        "project",
        "skill",
        "summary",
        "role_summary",
        "text",
        "content",
        "value",
        "label",
        "experience_level",
    )

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def available(self) -> bool:
        return self.settings.google_genai_enabled

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    def _call(self, prompt: str) -> dict[str, Any] | None:
        if not self.available:
            return None
        try:
            client = build_google_genai_client(self.settings)
        except ImportError:
            return None
        for model_id in self.settings.generation_model_candidates:
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                )
            except Exception as exc:
                if is_resource_exhausted_error(exc):
                    continue
                continue
            text = getattr(response, "text", None)
            if not text:
                continue
            payload = self._extract_json(text)
            if payload:
                return payload
        return None

    def _normalize_text(self, value: Any) -> str | None:
        if isinstance(value, str):
            normalized = " ".join(value.split()).strip()
            return normalized or None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, dict):
            for key in self._REPRESENTATIVE_KEYS:
                if key not in value:
                    continue
                normalized = self._normalize_text(value[key])
                if normalized:
                    return normalized
            return None
        if isinstance(value, (list, tuple, set)):
            for item in value:
                normalized = self._normalize_text(item)
                if normalized:
                    return normalized
        return None

    def _normalize_text_list(self, value: Any, *, lowercase: bool = False) -> list[str]:
        if value is None:
            return []

        if isinstance(value, dict):
            direct_item = self._normalize_text(value)
            candidates: list[Any] = [direct_item] if direct_item else list(value.values())
        elif isinstance(value, (list, tuple, set)):
            candidates = list(value)
        else:
            candidates = [value]

        normalized_items: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            iterable = candidate if isinstance(candidate, (list, tuple, set)) else [candidate]
            for item in iterable:
                normalized = self._normalize_text(item)
                if not normalized:
                    continue
                if lowercase:
                    normalized = normalized.lower()
                if normalized in seen:
                    continue
                seen.add(normalized)
                normalized_items.append(normalized)
        return normalized_items

    def refine_resume_facts(self, resume_facts: ResumeFacts) -> ResumeFacts:
        prompt = f"""
Return only JSON with keys summary, skills, projects.
Use this resume text to improve extraction without inventing evidence.

Resume:
{resume_facts.normalized_text[:5000]}
"""
        payload = self._call(prompt)
        if not payload:
            return resume_facts
        base_skills = self._normalize_text_list(resume_facts.skills, lowercase=True)
        merged_skills = sorted(set(base_skills) | set(self._normalize_text_list(payload.get("skills"), lowercase=True)))
        merged_projects = list(
            dict.fromkeys(
                self._normalize_text_list(resume_facts.projects) + self._normalize_text_list(payload.get("projects"))
            )
        )[:6]
        summary = self._normalize_text(payload.get("summary")) or resume_facts.summary
        return ResumeFacts(
            raw_text=resume_facts.raw_text,
            normalized_text=resume_facts.normalized_text,
            summary=summary,
            skills=merged_skills,
            projects=merged_projects,
            experiences=resume_facts.experiences,
            education=resume_facts.education,
            metrics=resume_facts.metrics,
            inferred_level=resume_facts.inferred_level,
            evidence_by_skill=resume_facts.evidence_by_skill,
            ats_findings=resume_facts.ats_findings,
            contact_signals=resume_facts.contact_signals,
            header_lines=resume_facts.header_lines,
            section_map=resume_facts.section_map,
            source_file_type=resume_facts.source_file_type,
        )

    def refine_job_requirements(self, requirements: JobRequirements, jd_text: str) -> JobRequirements:
        prompt = f"""
Return only JSON with keys role_summary, required_skills, preferred_skills, keywords, experience_level.
Do not invent tools not present in the job description.

Job description:
{jd_text[:5000]}
"""
        payload = self._call(prompt)
        if not payload:
            return requirements
        base_required_skills = self._normalize_text_list(requirements.required_skills, lowercase=True)
        base_preferred_skills = self._normalize_text_list(requirements.preferred_skills, lowercase=True)
        base_keywords = self._normalize_text_list(requirements.keywords, lowercase=True)
        return JobRequirements(
            role_title=requirements.role_title,
            company_name=requirements.company_name,
            role_summary=self._normalize_text(payload.get("role_summary")) or requirements.role_summary,
            required_skills=sorted(
                set(base_required_skills)
                | set(self._normalize_text_list(payload.get("required_skills"), lowercase=True))
            ),
            preferred_skills=sorted(
                set(base_preferred_skills)
                | set(self._normalize_text_list(payload.get("preferred_skills"), lowercase=True))
            ),
            keywords=sorted(
                set(base_keywords) | set(self._normalize_text_list(payload.get("keywords"), lowercase=True))
            ),
            responsibilities=requirements.responsibilities,
            experience_level=self._normalize_text(payload.get("experience_level")) or requirements.experience_level,
            ats_checks=requirements.ats_checks,
        )


class DiagnosticService:
    def __init__(self, settings: Settings | None = None, tracker: JobTracker | None = None) -> None:
        self.settings = settings or get_settings()
        self.tracker = tracker or JobTracker(repository=build_packet_repository(self.settings))
        self.augmenter = GeminiAugmenter(self.settings)

    def diagnose(
        self,
        *,
        resume_path: str | Path,
        jd_text: str,
        rejection_notes: str = "",
        user_id: str = "anonymous",
        session_id: str | None = None,
        persist: bool = True,
    ) -> DiagnosticSessionResult:
        parsed_resume = parse_resume_file(resume_path)
        parsed_job_description = parse_job_description(jd_text)
        rejection_signals = parse_rejection_notes(rejection_notes)
        resume_facts = extract_resume_facts(parsed_resume)
        requirements = extract_job_requirements(parsed_job_description)
        used_llm = False
        if self.augmenter.available:
            resume_facts = self.augmenter.refine_resume_facts(resume_facts)
            requirements = self.augmenter.refine_job_requirements(requirements, parsed_job_description.normalized_text)
            used_llm = True

        bundle = score_resume_match(resume_facts, requirements, rejection_signals)
        draft_report = DiagnosticReport(
            score_overall=bundle.score_overall,
            score_ats=bundle.score_ats,
            score_evidence=bundle.score_evidence,
            score_level_fit=bundle.score_level_fit,
            matched_skills=bundle.matched_skills,
            missing_skills=bundle.missing_skills,
            under_evidenced_skills=bundle.under_evidenced_skills,
            ats_findings=bundle.ats_findings,
            ats_checks=bundle.ats_checks,
            top_gaps=bundle.top_gaps,
            exact_edits=[],
            rewritten_resume=None,
            project_reframes=[],
            action_plan=[],
            interview_questions=[],
            provenance=bundle.provenance,
            recommended_decision=bundle.recommended_decision,
            narrative_summary=bundle.narrative_summary,
        )
        rewrite_package = generate_rewrite_package(resume_facts, requirements, draft_report)
        report_without_resume = DiagnosticReport(
            score_overall=bundle.score_overall,
            score_ats=bundle.score_ats,
            score_evidence=bundle.score_evidence,
            score_level_fit=bundle.score_level_fit,
            matched_skills=bundle.matched_skills,
            missing_skills=bundle.missing_skills,
            under_evidenced_skills=bundle.under_evidenced_skills,
            ats_findings=bundle.ats_findings,
            ats_checks=bundle.ats_checks,
            top_gaps=bundle.top_gaps,
            exact_edits=rewrite_package.exact_edits,
            rewritten_resume=None,
            project_reframes=rewrite_package.project_reframes,
            action_plan=generate_action_plan(draft_report, resume_facts, requirements, rejection_signals),
            interview_questions=generate_interview_questions(draft_report, resume_facts, requirements),
            provenance=bundle.provenance,
            recommended_decision=bundle.recommended_decision,
            narrative_summary=bundle.narrative_summary,
        )
        rewritten_resume = generate_rewritten_resume(
            resume_facts,
            requirements,
            report_without_resume,
            rewrite_package,
        )
        report = DiagnosticReport(
            score_overall=report_without_resume.score_overall,
            score_ats=report_without_resume.score_ats,
            score_evidence=report_without_resume.score_evidence,
            score_level_fit=report_without_resume.score_level_fit,
            matched_skills=report_without_resume.matched_skills,
            missing_skills=report_without_resume.missing_skills,
            under_evidenced_skills=report_without_resume.under_evidenced_skills,
            ats_findings=report_without_resume.ats_findings,
            ats_checks=report_without_resume.ats_checks,
            top_gaps=report_without_resume.top_gaps,
            exact_edits=report_without_resume.exact_edits,
            rewritten_resume=rewritten_resume,
            project_reframes=report_without_resume.project_reframes,
            action_plan=report_without_resume.action_plan,
            interview_questions=report_without_resume.interview_questions,
            provenance=report_without_resume.provenance,
            recommended_decision=report_without_resume.recommended_decision,
            narrative_summary=report_without_resume.narrative_summary,
        )
        packet = SavedJobPacket.new(
            user_id=user_id,
            session_id=session_id or str(uuid.uuid4()),
            resume_name=Path(resume_path).name,
            job_requirements=requirements,
            resume_facts=resume_facts,
            report=report,
            rejection_notes=rejection_notes,
        )
        if persist:
            packet = self.tracker.save(packet)
        return DiagnosticSessionResult(
            packet=packet,
            report_markdown=report.to_markdown(),
            used_llm_augmentation=used_llm,
        )

    def create_interview_session(self, *, packet_id: str, user_id: str) -> tuple[SavedJobPacket, InterviewSimulationSession] | None:
        packet = self.tracker.get(packet_id)
        if packet is None or packet.user_id != user_id:
            return None
        session = start_interview_session(packet)
        packet.interview_sessions.append(session)
        packet.updated_at = _utc_now_iso()
        self.tracker.save(packet)
        return packet, session

    def submit_interview_answer(
        self,
        *,
        packet_id: str,
        session_id: str,
        user_id: str,
        answer: str,
    ) -> tuple[SavedJobPacket, InterviewSimulationSession] | None:
        packet = self.tracker.get(packet_id)
        if packet is None or packet.user_id != user_id:
            return None
        for index, session in enumerate(packet.interview_sessions):
            if session.session_id != session_id:
                continue
            updated = submit_interview_answer(packet, session, answer)
            updated.updated_at = _utc_now_iso()
            packet.interview_sessions[index] = updated
            packet.updated_at = _utc_now_iso()
            self.tracker.save(packet)
            return packet, updated
        return None

    def compare_job_descriptions(
        self,
        *,
        resume_path: str | Path,
        jd_texts: list[str],
        rejection_notes: str = "",
        user_id: str = "anonymous",
    ) -> MultiJDComparison:
        rows: list[MultiJDRow] = []
        packets: list[SavedJobPacket] = []
        for jd_text in [item.strip() for item in jd_texts if item.strip()][:5]:
            result = self.diagnose(
                resume_path=resume_path,
                jd_text=jd_text,
                rejection_notes=rejection_notes,
                user_id=user_id,
                session_id=str(uuid.uuid4()),
                persist=True,
            )
            packet = result.packet
            packets.append(packet)
            rows.append(
                MultiJDRow(
                    packet_id=packet.packet_id,
                    role_title=packet.job_requirements.role_title,
                    company_name=packet.job_requirements.company_name,
                    score_overall=packet.report.score_overall,
                    score_ats=packet.report.score_ats,
                    score_evidence=packet.report.score_evidence,
                    score_level_fit=packet.report.score_level_fit,
                    recommended_decision=packet.report.recommended_decision,
                    top_gap_title=packet.report.top_gaps[0].title if packet.report.top_gaps else "No major gap detected",
                )
            )
        rows.sort(key=lambda item: item.score_overall, reverse=True)
        comparison = MultiJDComparison(
            comparison_id=str(uuid.uuid4()),
            user_id=user_id,
            resume_name=packets[0].resume_name if packets else Path(resume_path).name,
            rows=rows,
        )
        return self.tracker.save_comparison(comparison)
