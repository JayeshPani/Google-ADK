"""Firestore-backed and local packet repositories."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

from job_rejection_agent.config import Settings, get_settings
from job_rejection_agent.domain import MultiJDComparison, SavedJobPacket


class PacketRepository(Protocol):
    def save_packet(self, packet: SavedJobPacket) -> SavedJobPacket: ...
    def load_packet(self, packet_id: str) -> SavedJobPacket | None: ...
    def list_packets(self, user_id: str) -> list[SavedJobPacket]: ...
    def save_comparison(self, comparison: MultiJDComparison) -> MultiJDComparison: ...
    def load_comparison(self, comparison_id: str) -> MultiJDComparison | None: ...
    def list_comparisons(self, user_id: str) -> list[MultiJDComparison]: ...


@dataclass(slots=True)
class LocalJsonPacketRepository:
    storage_path: Path

    @property
    def comparison_storage_path(self) -> Path:
        return self.storage_path.with_name(f"{self.storage_path.stem}_comparisons.json")

    def _read_all(self) -> dict[str, dict]:
        if not self.storage_path.exists():
            return {}
        return json.loads(self.storage_path.read_text(encoding="utf-8"))

    def _write_all(self, payload: dict[str, dict]) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _read_comparisons(self) -> dict[str, dict]:
        if not self.comparison_storage_path.exists():
            return {}
        return json.loads(self.comparison_storage_path.read_text(encoding="utf-8"))

    def _write_comparisons(self, payload: dict[str, dict]) -> None:
        self.comparison_storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.comparison_storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save_packet(self, packet: SavedJobPacket) -> SavedJobPacket:
        payload = self._read_all()
        payload[packet.packet_id] = packet.to_dict()
        self._write_all(payload)
        return packet

    def load_packet(self, packet_id: str) -> SavedJobPacket | None:
        payload = self._read_all()
        document = payload.get(packet_id)
        return SavedJobPacket.from_dict(document) if document else None

    def list_packets(self, user_id: str) -> list[SavedJobPacket]:
        payload = self._read_all()
        packets = [
            SavedJobPacket.from_dict(document)
            for document in payload.values()
            if document.get("user_id") == user_id
        ]
        return sorted(packets, key=lambda item: item.updated_at, reverse=True)

    def save_comparison(self, comparison: MultiJDComparison) -> MultiJDComparison:
        payload = self._read_comparisons()
        payload[comparison.comparison_id] = comparison.to_dict()
        self._write_comparisons(payload)
        return comparison

    def load_comparison(self, comparison_id: str) -> MultiJDComparison | None:
        payload = self._read_comparisons()
        document = payload.get(comparison_id)
        return MultiJDComparison.from_dict(document) if document else None

    def list_comparisons(self, user_id: str) -> list[MultiJDComparison]:
        payload = self._read_comparisons()
        comparisons = [
            MultiJDComparison.from_dict(document)
            for document in payload.values()
            if document.get("user_id") == user_id
        ]
        return sorted(comparisons, key=lambda item: item.updated_at, reverse=True)


@dataclass(slots=True)
class FirestorePacketRepository:
    project_id: str
    collection_name: str

    def _collection(self):
        from google.cloud import firestore

        client = firestore.Client(project=self.project_id)
        return client.collection(self.collection_name)

    def _comparison_collection(self):
        from google.cloud import firestore

        client = firestore.Client(project=self.project_id)
        return client.collection(f"{self.collection_name}_comparisons")

    def save_packet(self, packet: SavedJobPacket) -> SavedJobPacket:
        self._collection().document(packet.packet_id).set(packet.to_dict())
        return packet

    def load_packet(self, packet_id: str) -> SavedJobPacket | None:
        document = self._collection().document(packet_id).get()
        if not document.exists:
            return None
        return SavedJobPacket.from_dict(document.to_dict())

    def list_packets(self, user_id: str) -> list[SavedJobPacket]:
        query = self._collection().where("user_id", "==", user_id).stream()
        packets = [SavedJobPacket.from_dict(document.to_dict()) for document in query]
        return sorted(packets, key=lambda item: item.updated_at, reverse=True)

    def save_comparison(self, comparison: MultiJDComparison) -> MultiJDComparison:
        self._comparison_collection().document(comparison.comparison_id).set(comparison.to_dict())
        return comparison

    def load_comparison(self, comparison_id: str) -> MultiJDComparison | None:
        document = self._comparison_collection().document(comparison_id).get()
        if not document.exists:
            return None
        return MultiJDComparison.from_dict(document.to_dict())

    def list_comparisons(self, user_id: str) -> list[MultiJDComparison]:
        query = self._comparison_collection().where("user_id", "==", user_id).stream()
        comparisons = [MultiJDComparison.from_dict(document.to_dict()) for document in query]
        return sorted(comparisons, key=lambda item: item.updated_at, reverse=True)


def build_packet_repository(settings: Settings | None = None) -> PacketRepository:
    settings = settings or get_settings()
    if settings.firestore_project_id:
        try:
            return FirestorePacketRepository(
                project_id=settings.firestore_project_id,
                collection_name=settings.firestore_collection,
            )
        except Exception:
            pass
    return LocalJsonPacketRepository(storage_path=settings.local_storage_path)
