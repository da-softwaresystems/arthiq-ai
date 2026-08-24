"""Configuration rules that are load-bearing rather than cosmetic."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


class TestModelPinning:
    """A model that moves under a prompt version breaks reproducibility."""

    @pytest.mark.parametrize(
        "alias",
        ["gemini-2.5-flash-latest", "llama3:latest", "some-model-preview"],
    )
    def test_floating_aliases_are_rejected(self, alias: str) -> None:
        with pytest.raises(ValidationError, match="floating alias"):
            _settings(gemini_model=alias)

    @pytest.mark.parametrize(
        "model",
        ["gemini-2.5-flash-lite", "gemini-2.5-flash", "qwen2.5:7b-instruct"],
    )
    def test_pinned_ids_are_accepted(self, model: str) -> None:
        assert _settings(gemini_model=model).gemini_model == model

    def test_no_model_is_hard_coded(self) -> None:
        settings = _settings()
        assert settings.ollama_model is None
        assert settings.gemini_model is None


class TestServiceKeys:
    def test_missing_key_means_no_accepted_keys(self) -> None:
        settings = _settings()
        assert settings.service_api_keys == frozenset()
        assert settings.service_auth_configured is False

    def test_comma_separated_keys_support_rotation(self) -> None:
        settings = _settings(ai_service_api_key="old-key-value, new-key-value")
        assert settings.service_api_keys == {"old-key-value", "new-key-value"}

    def test_key_is_a_secret_and_stays_out_of_repr(self) -> None:
        settings = _settings(ai_service_api_key="super-secret-value")
        assert "super-secret-value" not in repr(settings)
        assert "super-secret-value" not in str(settings.ai_service_api_key)

    def test_gemini_key_stays_out_of_repr(self) -> None:
        settings = _settings(gemini_api_key="AIzaSecretKeyValue")
        assert "AIzaSecretKeyValue" not in repr(settings)
        assert settings.gemini_api_key_value == "AIzaSecretKeyValue"

    def test_secret_values_collects_everything_redaction_needs(self) -> None:
        settings = _settings(ai_service_api_key="service-key-1", gemini_api_key="AIzaProviderKey")
        assert set(settings.secret_values()) == {"service-key-1", "AIzaProviderKey"}


class TestBounds:
    def test_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _settings(ai_request_timeout_seconds=0)

    def test_output_tokens_are_capped(self) -> None:
        with pytest.raises(ValidationError):
            _settings(ai_max_output_tokens=100_000)

    def test_unknown_provider_is_rejected_at_load(self) -> None:
        with pytest.raises(ValidationError):
            _settings(ai_provider="anthropic")

    def test_prompts_and_responses_are_not_logged_by_default(self) -> None:
        settings = _settings()
        assert settings.log_prompts is False
        assert settings.log_provider_responses is False

    def test_base_urls_lose_a_trailing_slash(self) -> None:
        assert _settings(ollama_base_url="http://localhost:11434/").ollama_base_url == (
            "http://localhost:11434"
        )
