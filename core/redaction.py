"""Shared helpers for redacting credentials from state, logs, and traces."""

from __future__ import annotations

import re
from typing import Any

REDACTION = "[REDACTED]"

_SECRET_QUERY_RE = re.compile(
    r"(?P<prefix>[?&](?:key|api_key|token|access_token|auth|authorization|password|secret)=)"
    r"(?P<value>[^&\s'\"]+)",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"(?P<prefix>\bBearer\s+)(?P<value>[A-Za-z0-9._~+/=-]+)", re.IGNORECASE)
_HEADER_SECRET_RE = re.compile(
    r"(?P<prefix>\b(?:x-api-key|api-key|authorization|hf_token|llm_api_key|serper_api_key)\s*[:=]\s*)"
    r"(?P<quote>['\"]?)"
    r"(?P<value>[^,'\"\s}]+)"
    r"(?P=quote)",
    re.IGNORECASE,
)


def is_secret_key(key: str) -> bool:
    normalized = str(key or "").lower()
    if not normalized:
        return False
    if normalized in {"api_key", "token", "secret", "password"}:
        return True
    if normalized.endswith("_api_key") or normalized.endswith("_token"):
        return True
    return "password" in normalized or "secret" in normalized


def sanitize_secret_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    redacted = _SECRET_QUERY_RE.sub(lambda match: f"{match.group('prefix')}{REDACTION}", value)
    redacted = _BEARER_RE.sub(lambda match: f"{match.group('prefix')}{REDACTION}", redacted)
    redacted = _HEADER_SECRET_RE.sub(
        lambda match: f"{match.group('prefix')}{match.group('quote')}{REDACTION}{match.group('quote')}",
        redacted,
    )
    return redacted


def redact_secrets(value: Any, key: str = "") -> Any:
    if is_secret_key(key):
        return REDACTION if value not in (None, "") else value
    if isinstance(value, dict):
        return {str(item_key): redact_secrets(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    return sanitize_secret_text(value)
