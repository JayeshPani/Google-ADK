"""Google model selection and failover helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


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


def is_resource_exhausted_error(exc: BaseException | None) -> bool:
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        code = getattr(current, "code", None)
        if code == 429:
            return True
        text = " ".join(
            [
                current.__class__.__name__,
                str(current),
                str(getattr(current, "details", "")),
                str(getattr(current, "response_json", "")),
            ]
        ).upper()
        if "RESOURCE_EXHAUSTED" in text:
            return True
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)
    return False
