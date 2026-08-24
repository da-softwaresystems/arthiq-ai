"""Scrubbing of anything that must never reach a log line or an API response.

Provider adapters build their error messages from vendor output, and vendor
output has been known to echo a request URL back - query string included. Every
message that leaves the provider layer passes through :func:`redact` first.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

MASK = "***"

#: Patterns that describe a credential by shape, so an unknown secret is caught
#: as well as a configured one.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\b(authorization|x-api-key|x-goog-api-key)\b\s*[:=]\s*.+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]+"),
    re.compile(r"(?i)([?&](?:key|api_key|apikey|access_token|token)=)[^&\s\"']+"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{10,}"),
)


def redact(text: str, extra_secrets: Iterable[str] = ()) -> str:
    """Return ``text`` with credentials replaced by :data:`MASK`.

    ``extra_secrets`` are literal values (the configured API keys); the
    patterns catch credentials this process was never told about.
    """
    if not text:
        return text
    cleaned = text
    for secret in extra_secrets:
        # A short "secret" would mask ordinary words; treat it as unusable.
        if secret and len(secret) >= 6:
            cleaned = cleaned.replace(secret, MASK)
    for pattern in _PATTERNS:
        if pattern.groups:
            cleaned = pattern.sub(lambda m: f"{m.group(1)}{MASK}", cleaned)
        else:
            cleaned = pattern.sub(MASK, cleaned)
    return cleaned


def truncate(text: str, limit: int = 500) -> str:
    """Shorten provider output before it is logged, never at full length."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [{len(text) - limit} more characters]"
