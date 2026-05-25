"""ADK session-service creation."""

from __future__ import annotations

from job_rejection_agent.config import Settings, get_settings


def create_session_service(settings: Settings | None = None):
    settings = settings or get_settings()
    try:
        from google.adk.sessions import DatabaseSessionService

        return DatabaseSessionService(db_url=settings.session_db_url)
    except Exception:
        from google.adk.sessions import InMemorySessionService

        return InMemorySessionService()
