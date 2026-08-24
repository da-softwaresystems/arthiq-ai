"""The provider abstraction, the fake provider, and provider selection."""

from __future__ import annotations

import json

import pytest

from app.core.config import Settings
from app.domain.enums import AnalysisDepth, Decision
from app.providers.base import AIProvider, CompletionRequest
from app.providers.exceptions import (
    ProviderNotConfiguredError,
    ProviderTimeoutError,
    UnsupportedProviderError,
)
from app.providers.fake import FakeAIProvider
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider
from app.providers.registry import build_provider, get_ai_provider, set_ai_provider


def _completion(user: str = "symbol: RELIANCE\n  rsi_14: 54.20") -> CompletionRequest:
    return CompletionRequest(
        system="system",
        user=user,
        max_output_tokens=256,
        temperature=0.0,
        timeout_seconds=5.0,
    )


class TestFakeProvider:
    async def test_returns_valid_decision_json(self) -> None:
        result = await FakeAIProvider().complete(_completion())
        payload = json.loads(result.text)
        assert payload["decision"] in {d.value for d in Decision}
        assert 0.0 <= payload["confidence"] <= 1.0
        assert payload["invalidating_conditions"]

    async def test_is_deterministic_across_instances(self) -> None:
        first = await FakeAIProvider().complete(_completion())
        second = await FakeAIProvider().complete(_completion())
        assert first.text == second.text

    async def test_different_prompts_give_different_answers(self) -> None:
        overbought = await FakeAIProvider().complete(_completion("symbol: X\n  rsi_14: 82.0"))
        oversold = await FakeAIProvider().complete(_completion("symbol: X\n  rsi_14: 12.0"))
        assert json.loads(overbought.text)["decision"] == "SELL"
        assert json.loads(oversold.text)["decision"] == "BUY"

    async def test_canned_responses_are_served_in_order(self) -> None:
        provider = FakeAIProvider(responses=['{"first": 1}', '{"second": 2}'])
        assert (await provider.complete(_completion())).text == '{"first": 1}'
        assert (await provider.complete(_completion())).text == '{"second": 2}'
        # The last canned response repeats rather than running out.
        assert (await provider.complete(_completion())).text == '{"second": 2}'

    async def test_injected_error_is_raised(self) -> None:
        provider = FakeAIProvider(error=ProviderTimeoutError("boom", provider="fake"))
        with pytest.raises(ProviderTimeoutError):
            await provider.complete(_completion())

    async def test_records_the_requests_it_was_given(self) -> None:
        provider = FakeAIProvider()
        await provider.complete(_completion())
        assert len(provider.calls) == 1
        assert provider.calls[0].max_output_tokens == 256

    async def test_reports_token_usage(self) -> None:
        result = await FakeAIProvider().complete(_completion())
        assert result.usage is not None
        assert result.usage.total_tokens is not None

    async def test_depth_selects_the_model(self) -> None:
        provider = FakeAIProvider(model="routine-model", deep_model="deep-model")
        assert provider.model_for(AnalysisDepth.ROUTINE) == "routine-model"
        assert provider.model_for(AnalysisDepth.DEEP) == "deep-model"
        result = await provider.complete(
            CompletionRequest(
                system="s",
                user="u",
                max_output_tokens=64,
                temperature=0.0,
                timeout_seconds=5.0,
                depth=AnalysisDepth.DEEP,
            )
        )
        assert result.model == "deep-model"

    async def test_readiness_needs_no_network(self) -> None:
        readiness = await FakeAIProvider().check_readiness()
        assert readiness.ready is True
        assert readiness.provider == "fake"


class TestAbstraction:
    @pytest.mark.parametrize(
        ("settings_kwargs", "expected"),
        [
            ({"ai_provider": "fake"}, FakeAIProvider),
            ({"ai_provider": "ollama", "ollama_model": "test-model"}, OllamaProvider),
            (
                {
                    "ai_provider": "gemini",
                    "gemini_api_key": "test-key",
                    "gemini_model": "gemini-2.5-flash-lite",
                },
                GeminiProvider,
            ),
        ],
    )
    def test_every_provider_implements_the_interface(
        self, settings_kwargs: dict[str, object], expected: type
    ) -> None:
        provider = build_provider(Settings(_env_file=None, **settings_kwargs))
        assert isinstance(provider, AIProvider)
        assert isinstance(provider, expected)

    def test_unknown_provider_is_refused(self) -> None:
        settings = Settings(_env_file=None, ai_provider="fake")
        # Assignment skips the literal check, simulating a build whose registry
        # no longer has the configured provider.
        settings.ai_provider = "anthropic"  # type: ignore[assignment]
        with pytest.raises(UnsupportedProviderError, match="anthropic"):
            build_provider(settings)

    def test_ollama_without_a_model_fails_closed(self) -> None:
        with pytest.raises(ProviderNotConfiguredError, match="OLLAMA_MODEL"):
            build_provider(Settings(_env_file=None, ai_provider="ollama"))

    def test_gemini_without_a_key_fails_closed(self) -> None:
        with pytest.raises(ProviderNotConfiguredError, match="GEMINI_API_KEY"):
            build_provider(
                Settings(_env_file=None, ai_provider="gemini", gemini_model="gemini-2.5-flash")
            )

    def test_gemini_without_a_model_fails_closed(self) -> None:
        with pytest.raises(ProviderNotConfiguredError, match="GEMINI_MODEL"):
            build_provider(
                Settings(_env_file=None, ai_provider="gemini", gemini_api_key="test-key")
            )

    def test_registry_returns_the_installed_provider(self) -> None:
        provider = FakeAIProvider()
        set_ai_provider(provider)
        try:
            assert get_ai_provider() is provider
        finally:
            set_ai_provider(None)
