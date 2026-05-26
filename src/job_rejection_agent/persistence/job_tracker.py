"""Saved job packet tracker."""

from __future__ import annotations

from dataclasses import dataclass

from job_rejection_agent.domain import SavedJobPacket, TrackerEntry

from .firestore import PacketRepository


@dataclass(slots=True)
class JobTracker:
    repository: PacketRepository

    def save(self, packet: SavedJobPacket) -> SavedJobPacket:
        return self.repository.save_packet(packet)

    def get(self, packet_id: str) -> SavedJobPacket | None:
        return self.repository.load_packet(packet_id)

    def find_by_session(self, user_id: str, session_id: str) -> SavedJobPacket | None:
        for packet in self.repository.list_packets(user_id):
            if packet.session_id == session_id:
                return packet
        return None

    def list_entries(self, user_id: str) -> list[TrackerEntry]:
        return [
            TrackerEntry(
                packet_id=packet.packet_id,
                user_id=packet.user_id,
                status=packet.report.recommended_decision,
                role_title=packet.job_requirements.role_title,
                company_name=packet.job_requirements.company_name,
                score_overall=packet.report.score_overall,
                updated_at=packet.updated_at,
            )
            for packet in self.repository.list_packets(user_id)
        ]
