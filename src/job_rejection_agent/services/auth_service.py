"""Authentication and session helpers."""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any
from urllib.parse import urlencode

import httpx

from job_rejection_agent.config import Settings, get_settings
from job_rejection_agent.domain import UserAccount
from job_rejection_agent.domain.schemas import utc_now_iso
from job_rejection_agent.persistence import JobTracker, build_packet_repository, build_user_repository

GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
GOOGLE_OAUTH_SCOPES = ("openid", "email", "profile")


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

    @property
    def google_oauth_enabled(self) -> bool:
        return self.settings.google_oauth_enabled

    def _sign_payload(self, payload: dict[str, Any]) -> str:
        encoded_payload = _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        signature = hmac.new(
            self.settings.app_secret_key.encode("utf-8"),
            encoded_payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{encoded_payload}.{signature}"

    def _verify_signed_payload(self, token: str | None) -> dict[str, Any] | None:
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
        return payload

    def register(self, *, email: str, password: str, guest_user_id: str | None = None) -> UserAccount:
        normalized_email = self.normalize_email(email)
        if "@" not in normalized_email or "." not in normalized_email.split("@")[-1]:
            raise AuthError("Enter a valid email address.")
        self._validate_password(password)
        existing = self.user_repository.load_by_email(normalized_email)
        if existing:
            if existing.password_hash and existing.password_salt:
                raise AuthError("An account with that email already exists.")
            salt = secrets.token_bytes(16)
            existing.password_hash = self._hash_password(password, salt)
            existing.password_salt = salt.hex()
            existing.updated_at = utc_now_iso()
            self.user_repository.save_user(existing)
            if guest_user_id:
                self.tracker.reassign_packets(guest_user_id, existing.user_id)
            return existing

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
        if not user.password_hash or not user.password_salt:
            raise AuthError("This account uses Google sign-in. Continue with Google instead.")
        expected = self._hash_password(password, bytes.fromhex(user.password_salt))
        if not hmac.compare_digest(expected, user.password_hash):
            raise AuthError("Incorrect password.")
        if guest_user_id:
            self.tracker.reassign_packets(guest_user_id, user.user_id)
        return user

    def create_google_oauth_state_token(
        self,
        *,
        next_path: str,
        guest_user_id: str | None,
        ttl_seconds: int = 600,
    ) -> str:
        nonce = secrets.token_urlsafe(24)
        return self._sign_payload(
            {
                "next_path": next_path,
                "guest_user_id": guest_user_id or "",
                "nonce": nonce,
                "exp": int(time.time()) + ttl_seconds,
            }
        )

    def verify_google_oauth_state_token(self, token: str | None) -> dict[str, Any] | None:
        return self._verify_signed_payload(token)

    def build_google_oauth_authorize_url(
        self,
        *,
        redirect_uri: str,
        state_token: str,
        login_hint: str | None = None,
    ) -> str:
        if not self.google_oauth_enabled:
            raise AuthError("Google sign-in is not configured.")
        query = {
            "client_id": self.settings.google_oauth_client_id or "",
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(GOOGLE_OAUTH_SCOPES),
            "state": state_token,
            "prompt": "select_account",
            "include_granted_scopes": "true",
        }
        if login_hint:
            query["login_hint"] = login_hint
        return f"{GOOGLE_AUTH_ENDPOINT}?{urlencode(query)}"

    async def authenticate_google_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        guest_user_id: str | None = None,
    ) -> UserAccount:
        if not self.google_oauth_enabled:
            raise AuthError("Google sign-in is not configured.")
        token_payload = {
            "code": code,
            "client_id": self.settings.google_oauth_client_id or "",
            "client_secret": self.settings.google_oauth_client_secret or "",
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                GOOGLE_TOKEN_ENDPOINT,
                data=token_payload,
                headers={"Accept": "application/json"},
            )
        if response.status_code >= 400:
            detail = response.text.strip() or "Google token exchange failed."
            raise AuthError(f"Google sign-in failed: {detail}")
        token_response = response.json()
        id_token_value = token_response.get("id_token")
        if not id_token_value:
            raise AuthError("Google sign-in did not return an ID token.")

        try:
            from google.auth.transport.requests import Request as GoogleRequest
            from google.oauth2.id_token import verify_oauth2_token

            claims = verify_oauth2_token(
                id_token_value,
                GoogleRequest(),
                self.settings.google_oauth_client_id,
            )
        except Exception as exc:  # pragma: no cover - exercised via mocks in tests
            raise AuthError(f"Google ID token verification failed: {exc}") from exc

        google_sub = str(claims.get("sub", "")).strip()
        email = self.normalize_email(str(claims.get("email", "")).strip())
        email_verified = bool(claims.get("email_verified"))
        if not google_sub or not email:
            raise AuthError("Google did not return a usable account identity.")
        if not email_verified:
            raise AuthError("Google account email is not verified.")

        user = self.user_repository.load_by_google_sub(google_sub)
        if user is None:
            existing = self.user_repository.load_by_email(email)
            if existing:
                if existing.google_sub and existing.google_sub != google_sub:
                    raise AuthError("That email is already linked to a different Google account.")
                existing.google_sub = google_sub
                existing.updated_at = utc_now_iso()
                user = self.user_repository.save_user(existing)
            else:
                user = self.user_repository.save_user(
                    UserAccount.new_google(
                        email=email,
                        google_sub=google_sub,
                    )
                )

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
        return self._sign_payload(payload)

    def verify_session_token(self, token: str | None) -> AuthSession | None:
        payload = self._verify_signed_payload(token)
        if payload is None:
            return None
        user = self.load_user(payload.get("uid", ""))
        if user is None or user.email != payload.get("email"):
            return None
        return AuthSession(user_id=user.user_id, email=user.email)
