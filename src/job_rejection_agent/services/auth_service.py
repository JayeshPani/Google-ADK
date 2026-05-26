"""Authentication and session helpers."""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import hmac
import json
import secrets
import time

from job_rejection_agent.config import Settings, get_settings
from job_rejection_agent.domain import UserAccount
from job_rejection_agent.persistence import JobTracker, build_packet_repository, build_user_repository


class AuthError(ValueError):
    """Raised for expected authentication failures."""


@dataclass(slots=True)
class AuthSession:
    user_id: str
    email: str


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _b64url_decode(raw: str) -> bytes:
    padding = "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(f"{raw}{padding}")


class AuthService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        tracker: JobTracker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.user_repository = build_user_repository(self.settings)
        self.tracker = tracker or JobTracker(repository=build_packet_repository(self.settings))

    @staticmethod
    def normalize_email(email: str) -> str:
        return email.strip().lower()

    def _hash_password(self, password: str, salt: bytes) -> str:
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 390000)
        return digest.hex()

    def _validate_password(self, password: str) -> None:
        if len(password) < 8:
            raise AuthError("Password must be at least 8 characters.")

    def register(self, *, email: str, password: str, guest_user_id: str | None = None) -> UserAccount:
        normalized_email = self.normalize_email(email)
        if "@" not in normalized_email or "." not in normalized_email.split("@")[-1]:
            raise AuthError("Enter a valid email address.")
        self._validate_password(password)
        if self.user_repository.load_by_email(normalized_email):
            raise AuthError("An account with that email already exists.")

        salt = secrets.token_bytes(16)
        user = UserAccount.new(
            email=normalized_email,
            password_hash=self._hash_password(password, salt),
            password_salt=salt.hex(),
        )
        self.user_repository.save_user(user)
        if guest_user_id:
            self.tracker.reassign_packets(guest_user_id, user.user_id)
        return user

    def authenticate(self, *, email: str, password: str, guest_user_id: str | None = None) -> UserAccount:
        normalized_email = self.normalize_email(email)
        user = self.user_repository.load_by_email(normalized_email)
        if user is None:
            raise AuthError("No account exists for that email.")
        expected = self._hash_password(password, bytes.fromhex(user.password_salt))
        if not hmac.compare_digest(expected, user.password_hash):
            raise AuthError("Incorrect password.")
        if guest_user_id:
            self.tracker.reassign_packets(guest_user_id, user.user_id)
        return user

    def load_user(self, user_id: str) -> UserAccount | None:
        return self.user_repository.load_user(user_id)

    def create_session_token(self, user: UserAccount, *, ttl_seconds: int) -> str:
        payload = {
            "uid": user.user_id,
            "email": user.email,
            "exp": int(time.time()) + ttl_seconds,
        }
        encoded_payload = _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        signature = hmac.new(
            self.settings.app_secret_key.encode("utf-8"),
            encoded_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{encoded_payload}.{signature}"

    def verify_session_token(self, token: str | None) -> AuthSession | None:
        if not token or "." not in token:
            return None
        encoded_payload, signature = token.rsplit(".", 1)
        expected_signature = hmac.new(
            self.settings.app_secret_key.encode("utf-8"),
            encoded_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected_signature, signature):
            return None
        try:
            payload = json.loads(_b64url_decode(encoded_payload).decode("utf-8"))
        except Exception:
            return None
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        user = self.load_user(payload.get("uid", ""))
        if user is None or user.email != payload.get("email"):
            return None
        return AuthSession(user_id=user.user_id, email=user.email)
