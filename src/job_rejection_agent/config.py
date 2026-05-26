"""Application configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from functools import lru_cache
from pathlib import Path
import shutil

from job_rejection_agent.google_models import dedupe_model_ids

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - allows local tests before dependencies are installed.
    def load_dotenv() -> bool:
        return False


ROOT_DIR = Path(__file__).resolve().parents[2]
PROMPTS_DIR = ROOT_DIR / "src" / "job_rejection_agent" / "agents" / "prompts"
LOCAL_DIR = ROOT_DIR / ".local"
PROMPT_HISTORY_DIR = ROOT_DIR / "prompt_history"


@dataclass(slots=True)
class Settings:
    app_name: str = "job-rejection-agent"
    model_id: str = "gemini-2.5-flash"
    eval_model_id: str = "gemini-2.5-flash"
    model_fallbacks: tuple[str, ...] = ("gemini-2.5-flash-lite",)
    eval_model_fallbacks: tuple[str, ...] = ("gemini-2.5-flash-lite",)
    google_api_key: str | None = None
    google_cloud_project: str | None = None
    google_cloud_location: str = "us-central1"
    google_genai_use_vertexai: bool = False
    phoenix_api_key: str | None = None
    phoenix_base_url: str = "https://app.phoenix.arize.com"
    phoenix_collector_endpoint: str = "https://app.phoenix.arize.com"
    phoenix_project_name: str = "job-rejection-agent"
    phoenix_mcp_enabled: bool = True
    phoenix_mcp_command: str = "npx"
    phoenix_mcp_args: tuple[str, ...] = (
        "-y",
        "@arizeai/phoenix-mcp@latest",
        "--baseUrl",
        "https://app.phoenix.arize.com",
    )
    firestore_project_id: str | None = None
    firestore_collection: str = "job_packets"
    firestore_user_collection: str = "users"
    session_db_url: str = "sqlite+aiosqlite:///./.local/adk_sessions.db"
    local_storage_path: Path = LOCAL_DIR / "job_packets.json"
    local_user_storage_path: Path = LOCAL_DIR / "users.json"
    prompt_version: str = "baseline-v1"
    prompt_path: Path = PROMPTS_DIR / "coaching_system_prompt.txt"
    prompt_candidate_path: Path = PROMPTS_DIR / "coaching_system_prompt_candidate.txt"
    prompt_history_dir: Path = PROMPT_HISTORY_DIR
    app_secret_key: str = "local-dev-secret"

    @property
    def phoenix_headers(self) -> dict[str, str]:
        if not self.phoenix_api_key:
            return {}
        return {"Authorization": f"Bearer {self.phoenix_api_key}"}

    @property
    def phoenix_query_base_url(self) -> str:
        if "/s/" in self.phoenix_base_url:
            return self.phoenix_base_url.rstrip("/")
        if "/s/" in self.phoenix_collector_endpoint:
            return self.phoenix_collector_endpoint.rstrip("/")
        return self.phoenix_base_url.rstrip("/")

    @property
    def generation_model_candidates(self) -> tuple[str, ...]:
        return dedupe_model_ids((self.model_id,), self.model_fallbacks)

    @property
    def evaluation_model_candidates(self) -> tuple[str, ...]:
        return dedupe_model_ids((self.eval_model_id,), self.eval_model_fallbacks, self.model_fallbacks)

    @property
    def google_genai_backend(self) -> str:
        return "vertexai" if self.google_genai_use_vertexai else "google_ai_studio"

    @property
    def google_genai_enabled(self) -> bool:
        if self.google_genai_use_vertexai:
            has_vertex_project = bool(self.google_cloud_project and self.google_cloud_location)
            has_express_key = bool(self.google_api_key and self.google_cloud_location)
            return has_vertex_project or has_express_key
        return bool(self.google_api_key)


def _read_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_mcp_command() -> str:
    return "phoenix-mcp" if shutil.which("phoenix-mcp") else "npx"


def _default_mcp_args(command: str) -> tuple[str, ...]:
    if command == "phoenix-mcp":
        return ("--baseUrl", "https://app.phoenix.arize.com")
    return (
        "-y",
        "@arizeai/phoenix-mcp@latest",
        "--baseUrl",
        "https://app.phoenix.arize.com",
    )


def _read_mcp_args(
    raw: str | None,
    *,
    command: str,
    substitutions: dict[str, str | None],
) -> tuple[str, ...]:
    if not raw:
        return _default_mcp_args(command)
    hydrated = raw
    for key, value in substitutions.items():
        hydrated = hydrated.replace(f"{{{key}}}", value or "")
    return tuple(part.strip() for part in hydrated.split(",") if part.strip())


def _read_model_list(raw: str | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if not raw:
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    LOCAL_DIR.mkdir(exist_ok=True)
    PROMPT_HISTORY_DIR.mkdir(exist_ok=True)
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
    phoenix_api_key = os.getenv("PHOENIX_API_KEY")
    phoenix_base_url = os.getenv("PHOENIX_BASE_URL", "https://app.phoenix.arize.com")
    phoenix_collector_endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com")
    phoenix_mcp_command = os.getenv("PHOENIX_MCP_COMMAND", _default_mcp_command())
    return Settings(
        model_id=os.getenv("MODEL_ID", "gemini-2.5-flash"),
        eval_model_id=os.getenv("EVAL_MODEL_ID", os.getenv("MODEL_ID", "gemini-2.5-flash")),
        model_fallbacks=_read_model_list(
            os.getenv("MODEL_FALLBACKS"),
            default=("gemini-2.5-flash-lite",),
        ),
        eval_model_fallbacks=_read_model_list(
            os.getenv("EVAL_MODEL_FALLBACKS"),
            default=("gemini-2.5-flash-lite",),
        ),
        google_api_key=google_api_key or None,
        google_cloud_project=os.getenv("GOOGLE_CLOUD_PROJECT") or None,
        google_cloud_location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        google_genai_use_vertexai=_read_bool("GOOGLE_GENAI_USE_VERTEXAI", default=False),
        phoenix_api_key=phoenix_api_key or None,
        phoenix_base_url=phoenix_base_url,
        phoenix_collector_endpoint=phoenix_collector_endpoint,
        phoenix_project_name=os.getenv("PHOENIX_PROJECT_NAME", "job-rejection-agent"),
        phoenix_mcp_enabled=_read_bool("PHOENIX_MCP_ENABLED", default=True),
        phoenix_mcp_command=phoenix_mcp_command,
        phoenix_mcp_args=_read_mcp_args(
            os.getenv("PHOENIX_MCP_ARGS"),
            command=phoenix_mcp_command,
            substitutions={
                "PHOENIX_API_KEY": phoenix_api_key,
                "PHOENIX_BASE_URL": phoenix_base_url,
                "PHOENIX_COLLECTOR_ENDPOINT": phoenix_collector_endpoint,
            },
        ),
        firestore_project_id=os.getenv("FIRESTORE_PROJECT_ID") or None,
        firestore_collection=os.getenv("FIRESTORE_COLLECTION", "job_packets"),
        firestore_user_collection=os.getenv("FIRESTORE_USER_COLLECTION", "users"),
        session_db_url=os.getenv("SESSION_DB_URL", "sqlite+aiosqlite:///./.local/adk_sessions.db"),
        local_storage_path=Path(os.getenv("LOCAL_STORAGE_PATH", ".local/job_packets.json")),
        local_user_storage_path=Path(os.getenv("LOCAL_USER_STORAGE_PATH", ".local/users.json")),
        prompt_version=os.getenv("PROMPT_VERSION", "baseline-v1"),
        app_secret_key=(
            os.getenv("APP_SECRET_KEY")
            or phoenix_api_key
            or google_api_key
            or "local-dev-secret"
        ),
    )
