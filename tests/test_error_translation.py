"""Provider failures become client errors that leak nothing."""

from __future__ import annotations

import pytest

from app.analysis.errors import safe_reason, translate_provider_error
from app.core.config import Settings
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

SERVICE_KEY = "service-key-value-123"
PROVIDER_KEY = "AIzaProviderKeyValue123"


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, ai_service_api_key=SERVICE_KEY, gemini_api_key=PROVIDER_KEY)


class TestTranslation:
    @pytest.mark.parametrize(
        ("error", "status_code", "code"),
        [
            (ProviderTimeoutError("slow"), 504, "PROVIDER_TIMEOUT"),
            (ProviderUnavailableError("down"), 503, "PROVIDER_UNAVAILABLE"),
            (ProviderRateLimitError("slow down"), 429, "PROVIDER_RATE_LIMITED"),
            (ProviderAuthError("bad key"), 502, "PROVIDER_AUTH_FAILED"),
            (ProviderResponseError("garbage"), 502, "PROVIDER_INVALID_RESPONSE"),
            (ProviderModelUnavailableError("no model"), 503, "PROVIDER_MODEL_UNAVAILABLE"),
            (ProviderNotConfiguredError("no key"), 503, "PROVIDER_NOT_CONFIGURED"),
            (UnsupportedProviderError("who?"), 503, "PROVIDER_UNSUPPORTED"),
            (AIProviderError("unknown"), 502, "PROVIDER_ERROR"),
        ],
    )
    def test_each_failure_maps_to_a_status_and_a_stable_code(
        self, error: AIProviderError, status_code: int, code: str
    ) -> None:
        app_error = translate_provider_error(error)
        assert app_error.status_code == status_code
        assert app_error.code == code

    def test_rate_limit_sets_retry_after(self) -> None:
        app_error = translate_provider_error(ProviderRateLimitError("slow down", retry_after=30.0))
        assert app_error.headers == {"Retry-After": "30"}

    def test_client_message_is_generic(self) -> None:
        app_error = translate_provider_error(
            ProviderAuthError("key AIzaSomething rejected by upstream", provider="gemini")
        )
        assert app_error.message == "The AI service could not authenticate with its provider"
        assert "AIza" not in app_error.message

    def test_provider_name_is_reported_but_not_the_model(self) -> None:
        app_error = translate_provider_error(
            ProviderTimeoutError("slow", provider="ollama", model="internal-model-name")
        )
        assert app_error.details == {"provider": "ollama"}
        assert "internal-model-name" not in str(app_error.details)


class TestRedaction:
    def test_provider_error_scrubs_credentials_at_construction(self) -> None:
        error = ProviderUnavailableError(
            "GET https://api.test/models?key=AIzaSecretKeyValue123 failed"
        )
        assert "AIzaSecretKeyValue123" not in str(error)
        assert "***" in str(error)

    def test_authorization_headers_are_scrubbed(self) -> None:
        assert "abc.def.ghi" not in redact("Authorization: Bearer abc.def.ghi")

    def test_x_api_key_is_scrubbed(self) -> None:
        assert "sk-secret-value" not in redact("x-api-key: sk-secret-value")

    def test_configured_secrets_are_scrubbed_by_value(self, settings: Settings) -> None:
        reason = safe_reason(
            ProviderAuthError(f"upstream said {PROVIDER_KEY} is invalid"), settings
        )
        assert PROVIDER_KEY not in reason

    def test_short_strings_are_not_treated_as_secrets(self) -> None:
        # Masking a three-character "secret" would redact ordinary words.
        assert redact("the model is up", ["up"]) == "the model is up"
