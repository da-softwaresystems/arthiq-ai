"""Provider failures, in provider-neutral terms.

An adapter never raises an HTTP concept and never raises a vendor exception
type. It raises one of these, and the analysis layer decides what each one
means for a client.

Every message passes through :func:`~app.core.redaction.redact` on the way in.
Vendors do echo request URLs back in error bodies, and a URL can carry an API
key; a message built from vendor output must be scrubbed before it exists at
all, not before it is logged.
"""

from __future__ import annotations

from app.core.redaction import redact


class AIProviderError(Exception):
    """Base class for any AI provider failure."""

    #: Stable, provider-neutral code. Mirrored into the API error envelope.
    code = "PROVIDER_ERROR"

    def __init__(
        self, message: str, *, provider: str = "unknown", model: str | None = None
    ) -> None:
        self.provider = provider
        self.model = model
        super().__init__(redact(message))


class ProviderTimeoutError(AIProviderError):
    """The provider did not answer inside the configured budget."""

    code = "PROVIDER_TIMEOUT"


class ProviderUnavailableError(AIProviderError):
    """Not reachable, or answered 5xx. Worth retrying later."""

    code = "PROVIDER_UNAVAILABLE"


class ProviderAuthError(AIProviderError):
    """The provider rejected our credentials.

    Ours, not the caller's - a client never supplies a provider credential.
    """

    code = "PROVIDER_AUTH_FAILED"


class ProviderRateLimitError(AIProviderError):
    """The provider throttled us."""

    code = "PROVIDER_RATE_LIMITED"

    def __init__(
        self,
        message: str,
        *,
        provider: str = "unknown",
        model: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, provider=provider, model=model)


class ProviderResponseError(AIProviderError):
    """The provider answered, but the answer was unusable.

    Covers a truncated completion, a body that is not JSON, and output that is
    JSON but does not validate into a decision. Not retryable on its own: the
    same prompt would produce the same nonsense.
    """

    code = "PROVIDER_INVALID_RESPONSE"


class ProviderModelUnavailableError(AIProviderError):
    """The configured model does not exist or is not pulled locally."""

    code = "PROVIDER_MODEL_UNAVAILABLE"


class ProviderNotConfiguredError(AIProviderError):
    """The provider is selected but cannot be used - no key, no model."""

    code = "PROVIDER_NOT_CONFIGURED"


class UnsupportedProviderError(AIProviderError):
    """``AI_PROVIDER`` names something this build does not implement."""

    code = "PROVIDER_UNSUPPORTED"
