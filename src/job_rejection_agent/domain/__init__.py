"""Typed domain contracts."""

from .auth import UserAccount
from .schemas import (
    ATSCheckResult,
    DiagnosticReport,
    EvidenceGap,
    InterviewFeedback,
    InterviewSimulationSession,
    InterviewTurn,
    ImprovementRun,
    JobRequirements,
    MultiJDComparison,
    MultiJDRow,
    ProvenanceNote,
    ResumeSectionBlock,
    ResumeFacts,
    RewrittenResume,
    RewritePatch,
    SavedJobPacket,
    TrackerEntry,
)

__all__ = [
    "ATSCheckResult",
    "DiagnosticReport",
    "EvidenceGap",
    "InterviewFeedback",
    "InterviewSimulationSession",
    "InterviewTurn",
    "ImprovementRun",
    "JobRequirements",
    "MultiJDComparison",
    "MultiJDRow",
    "ProvenanceNote",
    "ResumeSectionBlock",
    "ResumeFacts",
    "RewrittenResume",
    "RewritePatch",
    "SavedJobPacket",
    "TrackerEntry",
    "UserAccount",
]
