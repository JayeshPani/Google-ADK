"""Application configuration."""

from __future__ import annotations

from dataclasses import dataclass
import os
from functools import lru_cache
from pathlib import Path

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
    session_db_url: str = "sqlite+aiosqlite:///./.local/adk_sessions.db"
    local_storage_path: Path = LOCAL_DIR / "job_packets.json"
    prompt_version: str = "baseline-v1"
    prompt_path: Path = PROMPTS_DIR / "coaching_system_prompt.txt"
    prompt_candidate_path: Path = PROMPTS_DIR / "coaching_system_prompt_candidate.txt"
    prompt_history_dir: Path = PROMPT_HISTORY_DIR

    @property
    def phoenix_headers(self) -> dict[str, str]:
        if not self.phoenix_api_key:
            return {}
        return {"Authorization": f"Bearer {self.phoenix_api_key}"}


def _read_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_mcp_args(raw: str | None, api_key: str | None) -> tuple[str, ...]:
    if not raw:
        return (
            "-y",
            "@arizeai/phoenix-mcp@latest",
            "--baseUrl",
            "https://app.phoenix.arize.com",
        )
    hydrated = raw.replace("{PHOENIX_API_KEY}", api_key or "")
    return tuple(part.strip() for part in hydrated.split(",") if part.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    LOCAL_DIR.mkdir(exist_ok=True)
    PROMPT_HISTORY_DIR.mkdir(exist_ok=True)
    google_api_key = os.getenv("GOOGLE_API_KEY")
    phoenix_api_key = os.getenv("PHOENIX_API_KEY")
    return Settings(
        model_id=os.getenv("MODEL_ID", "gemini-2.5-flash"),
        eval_model_id=os.getenv("EVAL_MODEL_ID", os.getenv("MODEL_ID", "gemini-2.5-flash")),
        google_api_key=google_api_key or None,
        google_cloud_project=os.getenv("GOOGLE_CLOUD_PROJECT") or None,
        google_cloud_location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        google_genai_use_vertexai=_read_bool("GOOGLE_GENAI_USE_VERTEXAI", default=False),
        phoenix_api_key=phoenix_api_key or None,
        phoenix_base_url=os.getenv("PHOENIX_BASE_URL", "https://app.phoenix.arize.com"),
        phoenix_collector_endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "https://app.phoenix.arize.com"),
        phoenix_project_name=os.getenv("PHOENIX_PROJECT_NAME", "job-rejection-agent"),
        phoenix_mcp_enabled=_read_bool("PHOENIX_MCP_ENABLED", default=True),
        phoenix_mcp_command=os.getenv("PHOENIX_MCP_COMMAND", "npx"),
        phoenix_mcp_args=_read_mcp_args(os.getenv("PHOENIX_MCP_ARGS"), phoenix_api_key),
        firestore_project_id=os.getenv("FIRESTORE_PROJECT_ID") or None,
        firestore_collection=os.getenv("FIRESTORE_COLLECTION", "job_packets"),
        session_db_url=os.getenv("SESSION_DB_URL", "sqlite+aiosqlite:///./.local/adk_sessions.db"),
        local_storage_path=Path(os.getenv("LOCAL_STORAGE_PATH", ".local/job_packets.json")),
        prompt_version=os.getenv("PROMPT_VERSION", "baseline-v1"),
    )
