"""Authentication domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import uuid

from .schemas import utc_now_iso


@dataclass(slots=True)
class UserAccount:
    user_id: str
    email: str
    password_hash: str = ""
    password_salt: str = ""
    google_sub: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def new(
        cls,
        *,
        email: str,
        password_hash: str,
        password_salt: str,
    ) -> "UserAccount":
        return cls(
            user_id=f"user-{uuid.uuid4().hex[:16]}",
            email=email,
            password_hash=password_hash,
            password_salt=password_salt,
        )

    @classmethod
    def new_google(
        cls,
        *,
        email: str,
        google_sub: str,
    ) -> "UserAccount":
        return cls(
            user_id=f"user-{uuid.uuid4().hex[:16]}",
            email=email,
            google_sub=google_sub,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "password_hash": self.password_hash,
            "password_salt": self.password_salt,
            "google_sub": self.google_sub,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UserAccount":
        return cls(
            user_id=payload["user_id"],
            email=payload["email"],
            password_hash=payload.get("password_hash", ""),
            password_salt=payload.get("password_salt", ""),
            google_sub=payload.get("google_sub"),
            created_at=payload.get("created_at", utc_now_iso()),
            updated_at=payload.get("updated_at", utc_now_iso()),
        )
