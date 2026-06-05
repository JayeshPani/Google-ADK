"""Core orchestration service for diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
import time
from typing import Any, Callable
import uuid

from job_rejection_agent.analysis import ScoreBundle, extract_job_requirements, extract_resume_facts, score_resume_match
from job_rejection_agent.analysis.jd_requirements import split_hard_and_soft_requirements
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
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class CachedResumeParse:
    cache_key: str
    file_name: str
    parsed_resume: Any
    resume_facts: ResumeFacts
    created_at: float = field(default_factory=time.time)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


_DECISIONS = {"apply_now", "apply_after_patch", "defer", "not_fit"}
_COMPARE_DECISION_PRIORITY = {
    "apply_now": 4,
    "apply_after_patch": 3,
    "defer": 2,
    "not_fit": 1,
}
_RESUME_CACHE_TTL_SECONDS = 60 * 20


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        normalized = item.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            result.append(normalized)
            seen.add(key)
    return result


class ResumeParseCache:
    def __init__(self) -> None:
        self._items: dict[str, CachedResumeParse] = {}

    def _prune(self) -> None:
        now = time.time()
        self._items = {
            key: item
            for key, item in self._items.items()
            if now - item.created_at <= _RESUME_CACHE_TTL_SECONDS
        }

    def put(self, *, file_name: str, content: bytes) -> CachedResumeParse:
        self._prune()
        suffix = Path(file_name).suffix or ".txt"
        cache_key = hashlib.sha256(content).hexdigest()
        if cache_key in self._items:
            return self._items[cache_key]
        temp_path: Path | None = None
        try:
            with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(content)
                temp_path = Path(handle.name)
            parsed_resume = parse_resume_file(temp_path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        resume_facts = extract_resume_facts(parsed_resume)
        entry = CachedResumeParse(
            cache_key=cache_key,
            file_name=file_name,
            parsed_resume=parsed_resume,
            resume_facts=resume_facts,
        )
        self._items[cache_key] = entry
        return entry

    def get(self, cache_key: str) -> CachedResumeParse | None:
        self._prune()
        return self._items.get(cache_key)


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

    def _skill_evidence(self, resume_text: str, skills: list[str], existing: dict[str, list[str]]) -> dict[str, list[str]]:
        evidence = {skill: list(snippets) for skill, snippets in existing.items()}
        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+|\n", resume_text) if item.strip()]
        for skill in skills:
            if evidence.get(skill):
                continue
            variants = {skill, skill.replace("ml", "machine learning"), skill.replace("dl", "deep learning")}
            snippets = [
                sentence
                for sentence in sentences
                if any(variant and variant in sentence.lower() for variant in variants)
            ]
            if snippets:
                evidence[skill] = snippets[:3]
        return evidence

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
            evidence_by_skill=self._skill_evidence(resume_facts.normalized_text, merged_skills, resume_facts.evidence_by_skill),
            ats_findings=resume_facts.ats_findings,
            contact_signals=resume_facts.contact_signals,
            header_lines=resume_facts.header_lines,
            section_map=resume_facts.section_map,
            source_file_type=resume_facts.source_file_type,
        )

    def refine_job_requirements(self, requirements: JobRequirements, jd_text: str) -> JobRequirements:
        prompt = f"""
Return only JSON with keys role_summary, required_skills, preferred_skills, keywords, soft_requirements, experience_level.
Put hard technical tools, languages, frameworks, and platforms in required_skills/preferred_skills.
Put generic phrases like "ability to manage tasks", communication, coordination, presentations, and stakeholder skills in soft_requirements.
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
        payload_required, required_soft = split_hard_and_soft_requirements(
            self._normalize_text_list(payload.get("required_skills"), lowercase=True)
        )
        payload_preferred, preferred_soft = split_hard_and_soft_requirements(
            self._normalize_text_list(payload.get("preferred_skills"), lowercase=True)
        )
        _, keyword_soft = split_hard_and_soft_requirements(
            self._normalize_text_list(payload.get("keywords"), lowercase=True)
        )
        _, explicit_soft = split_hard_and_soft_requirements(
            self._normalize_text_list(payload.get("soft_requirements"), lowercase=True)
        )
        safe_keywords = [
            item
            for item in self._normalize_text_list(payload.get("keywords"), lowercase=True)
            if item not in {soft.lower() for soft in [*required_soft, *preferred_soft, *keyword_soft, *explicit_soft]}
        ]
        required_skills = sorted(set(base_required_skills) | set(payload_required))
        preferred_skills = sorted((set(base_preferred_skills) | set(payload_preferred)) - set(required_skills))
        soft_requirements = list(
            dict.fromkeys(
                [
                    *requirements.soft_requirements,
                    *required_soft,
                    *preferred_soft,
                    *keyword_soft,
                    *explicit_soft,
                ]
            )
        )[:8]
        return JobRequirements(
            role_title=requirements.role_title,
            company_name=requirements.company_name,
            role_summary=self._normalize_text(payload.get("role_summary")) or requirements.role_summary,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            keywords=sorted(set(base_keywords) | set(safe_keywords)),
            responsibilities=requirements.responsibilities,
            soft_requirements=soft_requirements,
            experience_level=self._normalize_text(payload.get("experience_level")) or requirements.experience_level,
            ats_checks=requirements.ats_checks,
        )

    def refine_profile(
        self,
        resume_facts: ResumeFacts,
        requirements: JobRequirements,
        jd_text: str,
    ) -> tuple[ResumeFacts, JobRequirements]:
        prompt = f"""
Return only JSON with keys resume_summary, resume_skills, resume_projects, role_summary, required_skills, preferred_skills, keywords, soft_requirements, experience_level.
Use this single request to improve both resume extraction and job requirements without inventing evidence.
Put hard technical tools, languages, frameworks, and platforms in required_skills/preferred_skills.
Put generic phrases like communication, coordination, presentations, stakeholders, and task management in soft_requirements.

Resume:
{resume_facts.normalized_text[:4200]}

Job description:
{jd_text[:4200]}
"""
        payload = self._call(prompt)
        if not payload:
            return resume_facts, requirements

        base_skills = self._normalize_text_list(resume_facts.skills, lowercase=True)
        merged_skills = sorted(set(base_skills) | set(self._normalize_text_list(payload.get("resume_skills"), lowercase=True)))
        merged_projects = list(
            dict.fromkeys(
                self._normalize_text_list(resume_facts.projects) + self._normalize_text_list(payload.get("resume_projects"))
            )
        )[:6]
        refined_resume = ResumeFacts(
            raw_text=resume_facts.raw_text,
            normalized_text=resume_facts.normalized_text,
            summary=self._normalize_text(payload.get("resume_summary")) or resume_facts.summary,
            skills=merged_skills,
            projects=merged_projects,
            experiences=resume_facts.experiences,
            education=resume_facts.education,
            metrics=resume_facts.metrics,
            inferred_level=resume_facts.inferred_level,
            evidence_by_skill=self._skill_evidence(resume_facts.normalized_text, merged_skills, resume_facts.evidence_by_skill),
            ats_findings=resume_facts.ats_findings,
            contact_signals=resume_facts.contact_signals,
            header_lines=resume_facts.header_lines,
            section_map=resume_facts.section_map,
            source_file_type=resume_facts.source_file_type,
        )

        base_required_skills = self._normalize_text_list(requirements.required_skills, lowercase=True)
        base_preferred_skills = self._normalize_text_list(requirements.preferred_skills, lowercase=True)
        base_keywords = self._normalize_text_list(requirements.keywords, lowercase=True)
        payload_required, required_soft = split_hard_and_soft_requirements(
            self._normalize_text_list(payload.get("required_skills"), lowercase=True)
        )
        payload_preferred, preferred_soft = split_hard_and_soft_requirements(
            self._normalize_text_list(payload.get("preferred_skills"), lowercase=True)
        )
        _, keyword_soft = split_hard_and_soft_requirements(
            self._normalize_text_list(payload.get("keywords"), lowercase=True)
        )
        _, explicit_soft = split_hard_and_soft_requirements(
            self._normalize_text_list(payload.get("soft_requirements"), lowercase=True)
        )
        soft_keys = {soft.lower() for soft in [*required_soft, *preferred_soft, *keyword_soft, *explicit_soft]}
        safe_keywords = [
            item
            for item in self._normalize_text_list(payload.get("keywords"), lowercase=True)
            if item not in soft_keys
        ]
        required_skills = sorted(set(base_required_skills) | set(payload_required))
        preferred_skills = sorted((set(base_preferred_skills) | set(payload_preferred)) - set(required_skills))
        soft_requirements = list(
            dict.fromkeys(
                [
                    *requirements.soft_requirements,
                    *required_soft,
                    *preferred_soft,
                    *keyword_soft,
                    *explicit_soft,
                ]
            )
        )[:8]
        refined_requirements = JobRequirements(
            role_title=requirements.role_title,
            company_name=requirements.company_name,
            role_summary=self._normalize_text(payload.get("role_summary")) or requirements.role_summary,
            required_skills=required_skills,
            preferred_skills=preferred_skills,
            keywords=sorted(set(base_keywords) | set(safe_keywords)),
            responsibilities=requirements.responsibilities,
            soft_requirements=soft_requirements,
            experience_level=self._normalize_text(payload.get("experience_level")) or requirements.experience_level,
            ats_checks=requirements.ats_checks,
        )
        return refined_resume, refined_requirements

    def _score_value(self, payload: dict[str, Any], key: str) -> float | None:
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

    def _score_rationale(self, payload: dict[str, Any], baseline: ScoreBundle) -> dict[str, str]:
        source = payload.get("rationale") or payload.get("score_rationale")
        if not isinstance(source, dict):
            return baseline.score_rationale or {}
        rationale: dict[str, str] = {}
        for key in ("overall", "ats", "evidence", "level_fit", "decision"):
            value = self._normalize_text(source.get(key))
            if value:
                rationale[key] = value[:260]
        return rationale or (baseline.score_rationale or {})

    def _score_review_is_valid(
        self,
        *,
        scores: dict[str, float],
        decision: str,
        requirements: JobRequirements,
        baseline: ScoreBundle,
    ) -> bool:
        required_count = len(requirements.required_skills)
        missing_ratio = len(baseline.missing_skills) / max(1, required_count)
        mostly_matched = missing_ratio <= 0.25 and baseline.score_level_fit >= 6.0 and baseline.score_evidence >= 6.0
        if mostly_matched and (decision in {"defer", "not_fit"} or scores["overall"] < 6.5):
            return False
        if decision == "apply_now" and (missing_ratio > 0.35 or scores["level_fit"] < 7.0 or scores["evidence"] < 6.2):
            return False
        if missing_ratio >= 0.55 and decision == "apply_now":
            return False
        if abs(scores["ats"] - baseline.score_ats) > 3.5:
            return False
        return True

    def review_scores(self, resume_facts: ResumeFacts, requirements: JobRequirements, baseline: ScoreBundle) -> ScoreBundle:
        prompt = f"""
Return only JSON with keys overall, ats, evidence, level_fit, recommended_decision, rationale.
You are the Match Analyst scoring reviewer. Score the candidate for this specific role.
Use hard required skills for missing-skill penalties. Treat soft requirements only as coaching context.
Recommended decision must be one of apply_now, apply_after_patch, defer, not_fit.

Baseline:
{json.dumps({
    "overall": baseline.score_overall,
    "ats": baseline.score_ats,
    "evidence": baseline.score_evidence,
    "level_fit": baseline.score_level_fit,
    "matched_skills": baseline.matched_skills,
    "missing_skills": baseline.missing_skills,
    "under_evidenced_skills": baseline.under_evidenced_skills,
    "decision": baseline.recommended_decision,
}, ensure_ascii=True)}

Job:
{json.dumps({
    "role_title": requirements.role_title,
    "company_name": requirements.company_name,
    "hard_required_skills": requirements.required_skills,
    "preferred_skills": requirements.preferred_skills,
    "soft_requirements": requirements.soft_requirements,
    "responsibilities": requirements.responsibilities[:6],
    "experience_level": requirements.experience_level,
}, ensure_ascii=True)}

Resume evidence:
{json.dumps({
    "skills": resume_facts.skills,
    "projects": resume_facts.projects[:6],
    "experiences": resume_facts.experiences[:6],
    "metrics": resume_facts.metrics[:8],
    "level": resume_facts.inferred_level,
}, ensure_ascii=True)}
"""
        payload = self._call(prompt)
        if not payload:
            return baseline
        scores = {
            "overall": self._score_value(payload, "overall"),
            "ats": self._score_value(payload, "ats"),
            "evidence": self._score_value(payload, "evidence"),
            "level_fit": self._score_value(payload, "level_fit"),
        }
        if any(value is None for value in scores.values()):
            return baseline
        typed_scores = {key: float(value) for key, value in scores.items() if value is not None}
        decision = self._normalize_text(payload.get("recommended_decision")) or ""
        if decision not in _DECISIONS:
            return baseline
        if not self._score_review_is_valid(
            scores=typed_scores,
            decision=decision,
            requirements=requirements,
            baseline=baseline,
        ):
            return baseline
        rationale = self._score_rationale(payload, baseline)
        summary = rationale.get(
            "overall",
            f"Overall fit is {typed_scores['overall']}/10 after AI scoring review.",
        )
        if not summary.lower().startswith("overall fit"):
            summary = f"Overall fit is {typed_scores['overall']}/10. {summary}"
        return replace(
            baseline,
            score_overall=typed_scores["overall"],
            score_ats=typed_scores["ats"],
            score_evidence=typed_scores["evidence"],
            score_level_fit=typed_scores["level_fit"],
            recommended_decision=decision,
            narrative_summary=summary,
            scoring_source="gemini",
            score_rationale=rationale,
        )


class DiagnosticService:
    def __init__(self, settings: Settings | None = None, tracker: JobTracker | None = None) -> None:
        self.settings = settings or get_settings()
        self.tracker = tracker or JobTracker(repository=build_packet_repository(self.settings))
        self.augmenter = GeminiAugmenter(self.settings)
        self.resume_cache = ResumeParseCache()

    def cache_resume_upload(self, *, file_name: str, content: bytes) -> CachedResumeParse:
        return self.resume_cache.put(file_name=file_name, content=content)

    def _cache_preview(self, entry: CachedResumeParse, *, max_chars: int = 2200) -> dict[str, Any]:
        preview_text = entry.parsed_resume.normalized_text[:max_chars].strip()
        return {
            "resume_cache_key": entry.cache_key,
            "display_name": entry.file_name,
            "file_type": entry.parsed_resume.file_type.upper(),
            "text": preview_text,
            "is_truncated": len(entry.parsed_resume.normalized_text) > len(preview_text),
            "line_count": len([line for line in preview_text.splitlines() if line.strip()]),
        }

    def preview_cached_resume(self, *, file_name: str, content: bytes, max_chars: int = 2200) -> dict[str, Any]:
        return self._cache_preview(self.cache_resume_upload(file_name=file_name, content=content), max_chars=max_chars)

    def _resume_from_cache_or_path(
        self,
        *,
        resume_path: str | Path,
        resume_cache_key: str = "",
        timings: dict[str, float],
    ) -> tuple[Any, ResumeFacts, str]:
        started = time.perf_counter()
        cache_entry = self.resume_cache.get(resume_cache_key) if resume_cache_key else None
        if cache_entry is not None:
            timings["resume_parse"] = round(time.perf_counter() - started, 4)
            timings["resume_cache_hit"] = 1.0
            return cache_entry.parsed_resume, cache_entry.resume_facts, cache_entry.file_name
        parsed_resume = parse_resume_file(resume_path)
        resume_facts = extract_resume_facts(parsed_resume)
        timings["resume_parse"] = round(time.perf_counter() - started, 4)
        timings["resume_cache_hit"] = 0.0
        return parsed_resume, resume_facts, parsed_resume.file_name

    def _quick_report(self, bundle: ScoreBundle) -> DiagnosticReport:
        return DiagnosticReport(
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
            scoring_source=bundle.scoring_source,
            score_rationale=bundle.score_rationale or {},
        )

    def _complete_report(
        self,
        *,
        packet: SavedJobPacket,
        rejection_signals: Any,
        timings: dict[str, float],
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> DiagnosticReport:
        progress_callback = progress_callback or (lambda _step, _percent: None)
        progress_callback("Generating resume patch", 78)
        started = time.perf_counter()
        rewrite_package = generate_rewrite_package(packet.resume_facts, packet.job_requirements, packet.report)
        timings["rewrite_package"] = round(time.perf_counter() - started, 4)

        progress_callback("Building roadmap and interview prep", 86)
        started = time.perf_counter()
        action_plan = generate_action_plan(packet.report, packet.resume_facts, packet.job_requirements, rejection_signals)
        interview_questions = generate_interview_questions(packet.report, packet.resume_facts, packet.job_requirements)
        timings["coaching_assets"] = round(time.perf_counter() - started, 4)

        progress_callback("Drafting evidence-backed resume", 93)
        started = time.perf_counter()
        rewritten_resume = generate_rewritten_resume(
            packet.resume_facts,
            packet.job_requirements,
            packet.report,
            rewrite_package,
        )
        timings["rewritten_resume"] = round(time.perf_counter() - started, 4)
        return replace(
            packet.report,
            exact_edits=rewrite_package.exact_edits,
            rewritten_resume=rewritten_resume,
            project_reframes=rewrite_package.project_reframes,
            action_plan=action_plan,
            interview_questions=interview_questions,
        )

    def diagnose(
        self,
        *,
        resume_path: str | Path,
        jd_text: str,
        rejection_notes: str = "",
        user_id: str = "anonymous",
        session_id: str | None = None,
        persist: bool = True,
        resume_cache_key: str = "",
        resume_name: str | None = None,
    ) -> DiagnosticSessionResult:
        quick = self.diagnose_quick(
            resume_path=resume_path,
            jd_text=jd_text,
            rejection_notes=rejection_notes,
            user_id=user_id,
            session_id=session_id,
            persist=False,
            resume_cache_key=resume_cache_key,
            resume_name=resume_name,
        )
        timings = dict(quick.timings)
        rejection_signals = parse_rejection_notes(rejection_notes)
        report = self._complete_report(
            packet=quick.packet,
            rejection_signals=rejection_signals,
            timings=timings,
        )
        packet = replace(
            quick.packet,
            report=report,
            updated_at=_utc_now_iso(),
        )
        if persist:
            packet = self.tracker.save(packet)
        return DiagnosticSessionResult(
            packet=packet,
            report_markdown=packet.report.to_markdown(),
            used_llm_augmentation=quick.used_llm_augmentation,
            timings=timings,
        )

    def diagnose_quick(
        self,
        *,
        resume_path: str | Path,
        jd_text: str,
        rejection_notes: str = "",
        user_id: str = "anonymous",
        session_id: str | None = None,
        persist: bool = True,
        resume_cache_key: str = "",
        resume_name: str | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> DiagnosticSessionResult:
        progress_callback = progress_callback or (lambda _step, _percent: None)
        timings: dict[str, float] = {}
        progress_callback("Upload received", 8)
        rejection_signals = parse_rejection_notes(rejection_notes)
        progress_callback("Parsing resume and job description", 18)
        with ThreadPoolExecutor(max_workers=2) as executor:
            resume_future = executor.submit(
                self._resume_from_cache_or_path,
                resume_path=resume_path,
                resume_cache_key=resume_cache_key,
                timings=timings,
            )
            jd_started = time.perf_counter()
            jd_future = executor.submit(parse_job_description, jd_text)
            parsed_resume, resume_facts, detected_resume_name = resume_future.result()
            parsed_job_description = jd_future.result()
            timings["jd_parse"] = round(time.perf_counter() - jd_started, 4)
        progress_callback("Extracting role requirements", 32)
        started = time.perf_counter()
        requirements = extract_job_requirements(parsed_job_description)
        timings["requirement_extract"] = round(time.perf_counter() - started, 4)

        used_llm = False
        if self.augmenter.available:
            progress_callback("Refining resume and JD with Gemini", 45)
            started = time.perf_counter()
            resume_facts, requirements = self.augmenter.refine_profile(
                resume_facts,
                requirements,
                parsed_job_description.normalized_text,
            )
            timings["gemini_profile_refine"] = round(time.perf_counter() - started, 4)
            used_llm = True

        progress_callback("Scoring fit", 58)
        started = time.perf_counter()
        bundle = score_resume_match(resume_facts, requirements, rejection_signals)
        timings["deterministic_scoring"] = round(time.perf_counter() - started, 4)
        if self.augmenter.available:
            progress_callback("Reviewing score with Gemini", 66)
            started = time.perf_counter()
            reviewed_bundle = self.augmenter.review_scores(resume_facts, requirements, bundle)
            timings["gemini_score_review"] = round(time.perf_counter() - started, 4)
            if reviewed_bundle.scoring_source != bundle.scoring_source:
                used_llm = True
            bundle = reviewed_bundle

        progress_callback("Quick score ready", 72)
        packet = SavedJobPacket.new(
            user_id=user_id,
            session_id=session_id or str(uuid.uuid4()),
            resume_name=resume_name or detected_resume_name or Path(resume_path).name,
            job_requirements=requirements,
            resume_facts=resume_facts,
            report=self._quick_report(bundle),
            rejection_notes=rejection_notes,
        )
        if persist:
            packet = self.tracker.save(packet)
        return DiagnosticSessionResult(
            packet=packet,
            report_markdown=packet.report.to_markdown(),
            used_llm_augmentation=used_llm,
            timings=timings,
        )

    def complete_diagnosis(
        self,
        *,
        packet_id: str,
        user_id: str | None = None,
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> DiagnosticSessionResult | None:
        packet = self.tracker.get(packet_id)
        if packet is None or (user_id is not None and packet.user_id != user_id):
            return None
        timings: dict[str, float] = {}
        if packet.report.rewritten_resume is not None and packet.report.exact_edits:
            return DiagnosticSessionResult(packet=packet, report_markdown=packet.report.to_markdown(), timings=timings)
        rejection_signals = parse_rejection_notes(packet.rejection_notes)
        report = self._complete_report(
            packet=packet,
            rejection_signals=rejection_signals,
            timings=timings,
            progress_callback=progress_callback,
        )
        packet.report = report
        packet.updated_at = _utc_now_iso()
        packet = self.tracker.save(packet)
        return DiagnosticSessionResult(packet=packet, report_markdown=packet.report.to_markdown(), timings=timings)

    def create_interview_session(self, *, packet_id: str, user_id: str) -> tuple[SavedJobPacket, InterviewSimulationSession] | None:
        packet = self.tracker.get(packet_id)
        if packet is None or packet.user_id != user_id:
            return None
        session = start_interview_session(packet, settings=self.settings)
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
            updated = submit_interview_answer(packet, session, answer, settings=self.settings)
            updated.updated_at = _utc_now_iso()
            packet.interview_sessions[index] = updated
            packet.updated_at = _utc_now_iso()
            self.tracker.save(packet)
            return packet, updated
        return None

    def _comparison_strengths(self, packet: SavedJobPacket) -> list[str]:
        strengths: list[str] = []
        for skill in packet.report.matched_skills[:4]:
            evidence = packet.resume_facts.evidence_by_skill.get(skill, [])
            if evidence:
                strengths.append(f"{skill}: {evidence[0][:140]}")
            else:
                strengths.append(f"Matches hard requirement: {skill}")
        for metric in packet.resume_facts.metrics[:2]:
            strengths.append(f"Metric evidence: {metric[:140]}")
        for project in packet.resume_facts.projects[:2]:
            strengths.append(f"Project evidence: {project[:140]}")
        return _dedupe_preserve_order(strengths)[:2] or ["General resume evidence is relevant to this role."]

    def _comparison_risks(self, packet: SavedJobPacket) -> list[str]:
        risks: list[str] = []
        risks.extend([f"Missing hard skill: {skill}" for skill in packet.report.missing_skills[:3]])
        risks.extend([f"Weak evidence for: {skill}" for skill in packet.report.under_evidenced_skills[:3]])
        for gap in packet.report.top_gaps[:2]:
            if gap.title not in risks:
                risks.append(gap.title)
        return _dedupe_preserve_order(risks)[:2] or ["No major hard-skill blocker detected."]

    def _comparison_next_action(self, packet: SavedJobPacket) -> str:
        missing = packet.report.missing_skills
        weak = packet.report.under_evidenced_skills
        decision = packet.report.recommended_decision
        if decision == "apply_now":
            return "Apply now; keep the strongest evidence bullets unchanged."
        if decision == "apply_after_patch":
            if weak:
                return f"Patch {weak[0]} evidence first, then apply."
            if missing:
                return f"Add truthful {missing[0]} evidence first, then apply."
            return "Make the recommended resume patch, then apply."
        if decision == "defer":
            if missing:
                return f"Defer until you can show real {missing[0]} evidence."
            return "Defer until the top evidence gap is stronger."
        return "Do not prioritize this role right now."

    def _comparison_rank_reason(self, row: MultiJDRow, packet: SavedJobPacket) -> str:
        coverage = int(round(row.hard_skill_coverage * 100))
        if row.recommended_decision == "apply_now":
            return f"Ranked here because it is ready to apply with {coverage}% hard-skill coverage and strong level fit."
        if row.recommended_decision == "apply_after_patch":
            return f"Ranked here because it is viable after a focused patch, with {coverage}% hard-skill coverage."
        if row.recommended_decision == "defer":
            return f"Ranked lower because hard-skill or evidence gaps still create application risk at {coverage}% coverage."
        return "Ranked lowest because the role is not a strong fit for the current resume evidence."

    def _comparison_row_from_packet(self, packet: SavedJobPacket) -> MultiJDRow:
        matched = packet.report.matched_skills
        missing = packet.report.missing_skills
        required_count = len(matched) + len(missing)
        hard_skill_coverage = round(len(matched) / max(1, required_count), 2)
        row = MultiJDRow(
            packet_id=packet.packet_id,
            role_title=packet.job_requirements.role_title,
            company_name=packet.job_requirements.company_name,
            score_overall=packet.report.score_overall,
            score_ats=packet.report.score_ats,
            score_evidence=packet.report.score_evidence,
            score_level_fit=packet.report.score_level_fit,
            recommended_decision=packet.report.recommended_decision,
            top_gap_title=packet.report.top_gaps[0].title if packet.report.top_gaps else "No major gap detected",
            strengths=self._comparison_strengths(packet),
            risks=self._comparison_risks(packet),
            next_action=self._comparison_next_action(packet),
            matched_skills=matched,
            missing_hard_skills=missing,
            under_evidenced_skills=packet.report.under_evidenced_skills,
            hard_skill_coverage=hard_skill_coverage,
            scoring_source=packet.report.scoring_source,
            score_rationale=packet.report.score_rationale,
        )
        row.rank_reason = self._comparison_rank_reason(row, packet)
        return row

    @staticmethod
    def _comparison_priority_key(row: MultiJDRow) -> tuple[int, float, float, float, float]:
        return (
            _COMPARE_DECISION_PRIORITY.get(row.recommended_decision, 0),
            row.score_overall,
            row.hard_skill_coverage,
            row.score_evidence,
            row.score_level_fit,
        )

    def _finalize_comparison_rows(self, rows: list[MultiJDRow]) -> list[MultiJDRow]:
        ranked_rows = sorted(rows, key=self._comparison_priority_key, reverse=True)
        for index, row in enumerate(ranked_rows, start=1):
            row.rank = index
        return ranked_rows

    def _comparison_common_missing(self, rows: list[MultiJDRow]) -> list[str]:
        counts: dict[str, int] = {}
        labels: dict[str, str] = {}
        for row in rows:
            for skill in row.missing_hard_skills:
                key = skill.lower()
                counts[key] = counts.get(key, 0) + 1
                labels[key] = skill
        common = sorted(counts, key=lambda key: (-counts[key], labels[key].lower()))
        repeated = [labels[key] for key in common if counts[key] > 1]
        return repeated[:4] or [labels[key] for key in common[:3]]

    def _comparison_strategy(self, rows: list[MultiJDRow]) -> str:
        common_missing = self._comparison_common_missing(rows)
        weak_skills = _dedupe_preserve_order(
            [skill for row in rows for skill in row.under_evidenced_skills]
        )
        if common_missing:
            return f"Prioritize truthful evidence for {', '.join(common_missing[:3])}; these gaps affect multiple roles."
        if weak_skills:
            return f"Strengthen evidence for {', '.join(weak_skills[:3])}; these skills are present but need clearer project outcomes."
        return "Preserve the current strongest evidence and tailor only the summary and top bullets per role."

    def compare_job_descriptions(
        self,
        *,
        resume_path: str | Path,
        jd_texts: list[str],
        rejection_notes: str = "",
        user_id: str = "anonymous",
        resume_cache_key: str = "",
        resume_name: str | None = None,
    ) -> MultiJDComparison:
        rows: list[MultiJDRow] = []
        packets: list[SavedJobPacket] = []
        cleaned_jds = [item.strip() for item in jd_texts if item.strip()][:5]
        cache_key = resume_cache_key
        display_name = resume_name or Path(resume_path).name
        if not cache_key:
            try:
                entry = self.cache_resume_upload(file_name=display_name, content=Path(resume_path).read_bytes())
                cache_key = entry.cache_key
                display_name = entry.file_name
            except Exception:
                cache_key = ""

        def run_quick(jd_text: str) -> DiagnosticSessionResult:
            return self.diagnose_quick(
                resume_path=resume_path,
                jd_text=jd_text,
                rejection_notes=rejection_notes,
                user_id=user_id,
                session_id=str(uuid.uuid4()),
                persist=False,
                resume_cache_key=cache_key,
                resume_name=display_name,
            )

        with ThreadPoolExecutor(max_workers=min(2, max(1, len(cleaned_jds)))) as executor:
            results = list(executor.map(run_quick, cleaned_jds))

        for result in results:
            packet = result.packet
            packet = self.tracker.save(packet)
            packets.append(packet)
            rows.append(self._comparison_row_from_packet(packet))
        rows = self._finalize_comparison_rows(rows)
        common_missing = self._comparison_common_missing(rows)
        best = rows[0] if rows else None
        comparison = MultiJDComparison(
            comparison_id=str(uuid.uuid4()),
            user_id=user_id,
            resume_name=packets[0].resume_name if packets else display_name,
            rows=rows,
            best_packet_id=best.packet_id if best else "",
            summary=(
                f"Best first application: {best.role_title} at {best.company_name} "
                f"({best.score_overall:.1f}/10, {best.next_action})"
                if best
                else "No roles were compared."
            ),
            common_missing_skills=common_missing,
            shared_resume_strategy=self._comparison_strategy(rows),
        )
        return self.tracker.save_comparison(comparison)
