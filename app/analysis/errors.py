"""Translating provider failures into client-facing errors.

The provider layer speaks in neutral failures; this module decides what each one
means over HTTP. It is the last place a message can leak, so every message is
scrubbed of configured secrets on the way out - the adapter already removed
credentials by shape, this removes them by value.

The status codes carry intent for the backend:

* 504 and 503 mean *try again later*;
* 502 means *the provider answered, and its answer was unusable* - retrying the
  identical prompt is unlikely to help;
* 429 means *back off*, with ``Retry-After`` when the provider said how long.
"""

from __future__ import annotations

from fastapi import status

from app.core.config import Settings
from app.core.exceptions import AppError
from app.core.redaction import redact
from app.providers.exceptions import (
    AIProviderError,
    ProviderAuthError,
    ProviderModelUnavailableError,
    ProviderNotConfiguredError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    UnsupportedProviderError,
)

#: Provider failure -> (HTTP status, client-facing message).
#:
#: The messages are deliberately generic. An operator gets the specific reason
#: from the logs; a client gets a stable code and enough to decide whether to
#: retry.
_TRANSLATIONS: dict[type[AIProviderError], tuple[int, str]] = {
    ProviderTimeoutError: (
        status.HTTP_504_GATEWAY_TIMEOUT,
        "The AI provider did not respond in time",
    ),
    ProviderUnavailableError: (
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "The AI provider is unavailable",
    ),
    ProviderRateLimitError: (
        status.HTTP_429_TOO_MANY_REQUESTS,
        "The AI provider is rate limiting requests",
    ),
    ProviderAuthError: (
        status.HTTP_502_BAD_GATEWAY,
        "The AI service could not authenticate with its provider",
    ),
    ProviderResponseError: (
        status.HTTP_502_BAD_GATEWAY,
        "The AI provider returned an unusable response",
    ),
    ProviderModelUnavailableError: (
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "The configured model is not available",
    ),
    ProviderNotConfiguredError: (
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "The AI provider is not configured",
    ),
    UnsupportedProviderError: (
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "The configured AI provider is not supported by this build",
    ),
}

_FALLBACK = (status.HTTP_502_BAD_GATEWAY, "The AI provider call failed")


def translate_provider_error(exc: AIProviderError) -> AppError:
    """Map a provider failure onto the service's error envelope."""
    status_code, message = _TRANSLATIONS.get(type(exc), _FALLBACK)

    headers: dict[str, str] | None = None
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        headers = {"Retry-After": str(int(retry_after))}

    return AppError(
        message,
        code=exc.code,
        status_code=status_code,
        # The provider name is safe and useful; the model id is not exposed,
        # because it is deployment configuration rather than client business.
        details={"provider": exc.provider},
        headers=headers,
    )


def safe_reason(exc: AIProviderError, settings: Settings) -> str:
    """The failure detail, scrubbed, for logs only. Never returned to a client."""
    return redact(str(exc), settings.secret_values())
