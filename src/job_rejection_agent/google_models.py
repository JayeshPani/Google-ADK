"""Google model selection and failover helpers."""

from __future__ import annotations

from collections.abc import Iterable
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from job_rejection_agent.config import Settings


def dedupe_model_ids(*groups: Iterable[str]) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            model_id = str(item).strip()
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            ordered.append(model_id)
    return tuple(ordered)


def apply_google_genai_environment(settings: "Settings") -> None:
    if settings.google_genai_use_vertexai:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        if settings.google_cloud_project:
            os.environ["GOOGLE_CLOUD_PROJECT"] = settings.google_cloud_project
        if settings.google_cloud_location:
            os.environ["GOOGLE_CLOUD_LOCATION"] = settings.google_cloud_location
        if settings.google_api_key:
            os.environ["GOOGLE_GENAI_API_KEY"] = settings.google_api_key
        return

    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"
    if settings.google_api_key:
        os.environ["GOOGLE_API_KEY"] = settings.google_api_key


def build_google_genai_client(settings: "Settings") -> Any:
    from google import genai

    apply_google_genai_environment(settings)
    if settings.google_genai_use_vertexai:
        kwargs: dict[str, Any] = {
            "vertexai": True,
            "location": settings.google_cloud_location,
        }
        if settings.google_cloud_project:
            kwargs["project"] = settings.google_cloud_project
        elif settings.google_api_key:
            kwargs["api_key"] = settings.google_api_key
        return genai.Client(**kwargs)
    return genai.Client(api_key=settings.google_api_key)


def is_resource_exhausted_error(exc: BaseException | None) -> bool:
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        code = getattr(current, "code", None)
        if code in {429, 503}:
            return True
        text = " ".join(
            [
                current.__class__.__name__,
                str(current),
                str(getattr(current, "details", "")),
                str(getattr(current, "response_json", "")),
            ]
        ).upper()
        if (
            "RESOURCE_EXHAUSTED" in text
            or "503 UNAVAILABLE" in text
            or "CURRENTLY EXPERIENCING HIGH DEMAND" in text
        ):
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return False
