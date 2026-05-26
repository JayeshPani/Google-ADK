"""Core domain schemas for the diagnostic pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from typing import Any, Literal
import json
import uuid


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


@dataclass(slots=True)
class ProvenanceNote:
    label: str
    evidence: str
    source: Literal["resume", "job_description", "rejection_note", "heuristic", "llm"]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProvenanceNote":
        return cls(**payload)


@dataclass(slots=True)
class ATSCheckResult:
    category: Literal[
        "section_structure",
        "contact_info",
        "readability",
        "formatting_risk",
        "keyword_coverage",
        "file_hygiene",
    ]
    status: Literal["pass", "warn", "fail"]
    title: str
    details: str
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ATSCheckResult":
        return cls(**payload)


@dataclass(slots=True)
class ResumeFacts:
    raw_text: str
    normalized_text: str
    summary: str
    skills: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    experiences: list[str] = field(default_factory=list)
    education: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    inferred_level: str = "new_grad"
    evidence_by_skill: dict[str, list[str]] = field(default_factory=dict)
    ats_findings: list[str] = field(default_factory=list)
    contact_signals: dict[str, bool] = field(default_factory=dict)
    header_lines: list[str] = field(default_factory=list)
    section_map: dict[str, list[str]] = field(default_factory=dict)
    source_file_type: str = "txt"

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResumeFacts":
        return cls(**payload)


@dataclass(slots=True)
class JobRequirements:
    role_title: str
    company_name: str
    role_summary: str
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    experience_level: str = "entry"
    ats_checks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JobRequirements":
        return cls(**payload)


@dataclass(slots=True)
class EvidenceGap:
    category: Literal["ats", "missing_skill", "under_evidenced_skill", "level_fit", "rejection_signal"]
    title: str
    severity: Literal["high", "medium", "low"]
    details: str
    recommended_fix: str
    supporting_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvidenceGap":
        return cls(**payload)


@dataclass(slots=True)
class RewritePatch:
    section: Literal["summary", "project", "experience", "skills", "education"]
    original_text: str
    rewritten_text: str
    reason: str
    confidence: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RewritePatch":
        return cls(**payload)


@dataclass(slots=True)
class ResumeSectionBlock:
    title: str
    items: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResumeSectionBlock":
        return cls(**payload)


@dataclass(slots=True)
class RewrittenResume:
    header: ResumeSectionBlock
    summary: ResumeSectionBlock
    experience: ResumeSectionBlock
    projects: ResumeSectionBlock
    skills: ResumeSectionBlock
    education: ResumeSectionBlock
    ats_notes: list[str] = field(default_factory=list)
    provenance_map: dict[str, list[ProvenanceNote]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RewrittenResume":
        provenance_map = {
            key: [ProvenanceNote.from_dict(item) for item in items]
            for key, items in payload.get("provenance_map", {}).items()
        }
        return cls(
            header=ResumeSectionBlock.from_dict(payload["header"]),
            summary=ResumeSectionBlock.from_dict(payload["summary"]),
            experience=ResumeSectionBlock.from_dict(payload["experience"]),
            projects=ResumeSectionBlock.from_dict(payload["projects"]),
            skills=ResumeSectionBlock.from_dict(payload["skills"]),
            education=ResumeSectionBlock.from_dict(payload["education"]),
            ats_notes=payload.get("ats_notes", []),
            provenance_map=provenance_map,
        )


@dataclass(slots=True)
class InterviewFeedback:
    overall_score: float
    evidence_score: float
    clarity_score: float
    relevance_score: float
    gap_score: float
    feedback: str
    suggested_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InterviewFeedback":
        return cls(**payload)


@dataclass(slots=True)
class InterviewTurn:
    question: str
    answer: str
    feedback: InterviewFeedback
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InterviewTurn":
        return cls(
            question=payload["question"],
            answer=payload["answer"],
            feedback=InterviewFeedback.from_dict(payload["feedback"]),
            created_at=payload.get("created_at", utc_now_iso()),
        )


@dataclass(slots=True)
class InterviewSimulationSession:
    session_id: str
    packet_id: str
    user_id: str
    questions: list[str]
    current_index: int = 0
    status: Literal["in_progress", "completed"] = "in_progress"
    turns: list[InterviewTurn] = field(default_factory=list)
    final_summary: str = ""
    started_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InterviewSimulationSession":
        return cls(
            session_id=payload["session_id"],
            packet_id=payload["packet_id"],
            user_id=payload["user_id"],
            questions=payload.get("questions", []),
            current_index=payload.get("current_index", 0),
            status=payload.get("status", "in_progress"),
            turns=[InterviewTurn.from_dict(item) for item in payload.get("turns", [])],
            final_summary=payload.get("final_summary", ""),
            started_at=payload.get("started_at", utc_now_iso()),
            updated_at=payload.get("updated_at", utc_now_iso()),
        )


@dataclass(slots=True)
class MultiJDRow:
    packet_id: str
    role_title: str
    company_name: str
    score_overall: float
    score_ats: float
    score_evidence: float
    score_level_fit: float
    recommended_decision: Literal["apply_now", "apply_after_patch", "defer", "not_fit"]
    top_gap_title: str

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MultiJDRow":
        return cls(**payload)


@dataclass(slots=True)
class MultiJDComparison:
    comparison_id: str
    user_id: str
    resume_name: str
    rows: list[MultiJDRow]
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MultiJDComparison":
        return cls(
            comparison_id=payload["comparison_id"],
            user_id=payload["user_id"],
            resume_name=payload["resume_name"],
            rows=[MultiJDRow.from_dict(item) for item in payload.get("rows", [])],
            created_at=payload.get("created_at", utc_now_iso()),
            updated_at=payload.get("updated_at", utc_now_iso()),
        )


@dataclass(slots=True)
class DiagnosticReport:
    score_overall: float
    score_ats: float
    score_evidence: float
    score_level_fit: float
    matched_skills: list[str]
    missing_skills: list[str]
    under_evidenced_skills: list[str]
    ats_findings: list[str]
    ats_checks: list[ATSCheckResult]
    top_gaps: list[EvidenceGap]
    exact_edits: list[RewritePatch]
    rewritten_resume: RewrittenResume | None
    project_reframes: list[str]
    action_plan: list[str]
    interview_questions: list[str]
    provenance: list[ProvenanceNote]
    recommended_decision: Literal["apply_now", "apply_after_patch", "defer", "not_fit"]
    narrative_summary: str

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiagnosticReport":
        return cls(
            score_overall=payload["score_overall"],
            score_ats=payload["score_ats"],
            score_evidence=payload["score_evidence"],
            score_level_fit=payload["score_level_fit"],
            matched_skills=payload["matched_skills"],
            missing_skills=payload["missing_skills"],
            under_evidenced_skills=payload["under_evidenced_skills"],
            ats_findings=payload.get("ats_findings", []),
            ats_checks=[ATSCheckResult.from_dict(item) for item in payload.get("ats_checks", [])],
            top_gaps=[EvidenceGap.from_dict(item) for item in payload["top_gaps"]],
            exact_edits=[RewritePatch.from_dict(item) for item in payload["exact_edits"]],
            rewritten_resume=RewrittenResume.from_dict(payload["rewritten_resume"]) if payload.get("rewritten_resume") else None,
            project_reframes=payload["project_reframes"],
            action_plan=payload["action_plan"],
            interview_questions=payload["interview_questions"],
            provenance=[ProvenanceNote.from_dict(item) for item in payload["provenance"]],
            recommended_decision=payload["recommended_decision"],
            narrative_summary=payload["narrative_summary"],
        )

    def to_markdown(self) -> str:
        top_gap_lines = [f"- {gap.title}: {gap.details}" for gap in self.top_gaps[:3]]
        action_lines = [f"- {item}" for item in self.action_plan]
        edit_lines = [f"- {patch.rewritten_text} ({patch.reason})" for patch in self.exact_edits[:3]]
        return "\n".join(
            [
                f"## Match Score: {self.score_overall:.1f}/10",
                "",
                f"**Recommendation:** `{self.recommended_decision}`",
                "",
                "### Why you're getting screened out",
                *top_gap_lines,
                "",
                "### Resume edits to make now",
                *edit_lines,
                "",
                "### This week's action plan",
                *action_lines,
            ]
        )


@dataclass(slots=True)
class TrackerEntry:
    packet_id: str
    user_id: str
    status: Literal["draft", "apply_now", "apply_after_patch", "defer", "not_fit"]
    role_title: str
    company_name: str
    score_overall: float
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrackerEntry":
        return cls(**payload)


@dataclass(slots=True)
class SavedJobPacket:
    packet_id: str
    user_id: str
    session_id: str
    resume_name: str
    job_requirements: JobRequirements
    resume_facts: ResumeFacts
    report: DiagnosticReport
    rejection_notes: str = ""
    interview_sessions: list[InterviewSimulationSession] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def new(
        cls,
        *,
        user_id: str,
        session_id: str,
        resume_name: str,
        job_requirements: JobRequirements,
        resume_facts: ResumeFacts,
        report: DiagnosticReport,
        rejection_notes: str = "",
        interview_sessions: list[InterviewSimulationSession] | None = None,
    ) -> "SavedJobPacket":
        return cls(
            packet_id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            resume_name=resume_name,
            job_requirements=job_requirements,
            resume_facts=resume_facts,
            report=report,
            rejection_notes=rejection_notes,
            interview_sessions=interview_sessions or [],
        )

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SavedJobPacket":
        return cls(
            packet_id=payload["packet_id"],
            user_id=payload["user_id"],
            session_id=payload["session_id"],
            resume_name=payload["resume_name"],
            job_requirements=JobRequirements.from_dict(payload["job_requirements"]),
            resume_facts=ResumeFacts.from_dict(payload["resume_facts"]),
            report=DiagnosticReport.from_dict(payload["report"]),
            rejection_notes=payload.get("rejection_notes", ""),
            interview_sessions=[
                InterviewSimulationSession.from_dict(item)
                for item in payload.get("interview_sessions", [])
            ],
            created_at=payload.get("created_at", utc_now_iso()),
            updated_at=payload.get("updated_at", utc_now_iso()),
        )


@dataclass(slots=True)
class ImprovementRun:
    run_id: str
    baseline_prompt_version: str
    candidate_prompt_version: str
    source_span_ids: list[str]
    baseline_scores: dict[str, float]
    candidate_scores: dict[str, float]
    promoted: bool
    analysis: str
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImprovementRun":
        return cls(**payload)
