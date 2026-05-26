"""Persistent storage for user accounts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

from job_rejection_agent.config import Settings, get_settings
from job_rejection_agent.domain import UserAccount


class UserRepository(Protocol):
    def save_user(self, user: UserAccount) -> UserAccount: ...
    def load_user(self, user_id: str) -> UserAccount | None: ...
    def load_by_email(self, email: str) -> UserAccount | None: ...


@dataclass(slots=True)
class LocalJsonUserRepository:
    storage_path: Path

    def _read_all(self) -> dict[str, dict]:
        if not self.storage_path.exists():
            return {}
        return json.loads(self.storage_path.read_text(encoding="utf-8"))

    def _write_all(self, payload: dict[str, dict]) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def save_user(self, user: UserAccount) -> UserAccount:
        payload = self._read_all()
        payload[user.user_id] = user.to_dict()
        self._write_all(payload)
        return user

    def load_user(self, user_id: str) -> UserAccount | None:
        payload = self._read_all()
        document = payload.get(user_id)
        return UserAccount.from_dict(document) if document else None

    def load_by_email(self, email: str) -> UserAccount | None:
        normalized = email.strip().lower()
        payload = self._read_all()
        for document in payload.values():
            if str(document.get("email", "")).strip().lower() == normalized:
                return UserAccount.from_dict(document)
        return None


@dataclass(slots=True)
class FirestoreUserRepository:
    project_id: str
    collection_name: str

    def _collection(self):
        from google.cloud import firestore

        client = firestore.Client(project=self.project_id)
        return client.collection(self.collection_name)

    def save_user(self, user: UserAccount) -> UserAccount:
        self._collection().document(user.user_id).set(user.to_dict())
        return user

    def load_user(self, user_id: str) -> UserAccount | None:
        document = self._collection().document(user_id).get()
        if not document.exists:
            return None
        return UserAccount.from_dict(document.to_dict())

    def load_by_email(self, email: str) -> UserAccount | None:
        normalized = email.strip().lower()
        query = self._collection().where("email", "==", normalized).limit(1).stream()
        document = next(iter(query), None)
        return UserAccount.from_dict(document.to_dict()) if document else None


def build_user_repository(settings: Settings | None = None) -> UserRepository:
    settings = settings or get_settings()
    if settings.firestore_project_id:
        try:
            return FirestoreUserRepository(
                project_id=settings.firestore_project_id,
                collection_name=settings.firestore_user_collection,
            )
        except Exception:
            pass
    return LocalJsonUserRepository(storage_path=settings.local_user_storage_path)
