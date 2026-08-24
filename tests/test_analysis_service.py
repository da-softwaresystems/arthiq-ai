"""The analysis pipeline, end to end against the deterministic provider."""

from __future__ import annotations

import json
from datetime import UTC

import pytest

from app.analysis.service import AnalysisService
from app.core.config import Settings
from app.core.exceptions import AppError, ValidationError
from app.domain.analysis import AnalysisRequest
from app.domain.enums import AnalysisDepth, Decision, RiskLevel
from app.providers.exceptions import (
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from app.providers.fake import FakeAIProvider

BUY_RESPONSE = json.dumps(
    {
        "decision": "BUY",
        "confidence": 0.78,
        "risk_level": "MEDIUM",
        "reasoning": "Price holds above EMA20 and EMA50; MACD is positive.",
        "key_factors": ["price above EMA50"],
        "invalidating_conditions": ["Close below EMA50"],
    }
)
SELL_RESPONSE = json.dumps(
    {
        "decision": "SELL",
        "confidence": 0.61,
        "risk_level": "HIGH",
        "reasoning": "RSI-14 is overbought and price has stalled at the upper band.",
        "invalidating_conditions": ["Close above the upper Bollinger band"],
    }
)
HOLD_RESPONSE = json.dumps(
    {
        "decision": "HOLD",
        "confidence": 0.4,
        "risk_level": "LOW",
        "reasoning": "Indicators are mixed and no level has been reclaimed.",
        "invalidating_conditions": [],
    }
)


def _service(settings: Settings, provider: FakeAIProvider) -> AnalysisService:
    return AnalysisService(provider, settings)


class TestDecisions:
    @pytest.mark.parametrize(
        ("response", "expected", "risk"),
        [
            (BUY_RESPONSE, Decision.BUY, RiskLevel.MEDIUM),
            (SELL_RESPONSE, Decision.SELL, RiskLevel.HIGH),
            (HOLD_RESPONSE, Decision.HOLD, RiskLevel.LOW),
        ],
    )
    async def test_each_decision_survives_the_pipeline(
        self,
        settings: Settings,
        analysis_request: AnalysisRequest,
        response: str,
        expected: Decision,
        risk: RiskLevel,
    ) -> None:
        service = _service(settings, FakeAIProvider(responses=[response]))
        decision = await service.analyze(analysis_request)
        assert decision.decision is expected
        assert decision.risk_level is risk

    async def test_the_same_request_gives_the_same_decision(
        self, settings: Settings, analysis_request: AnalysisRequest
    ) -> None:
        first = await _service(settings, FakeAIProvider()).analyze(analysis_request)
        second = await _service(settings, FakeAIProvider()).analyze(analysis_request)
        assert first.decision is second.decision
        assert first.confidence == second.confidence
        assert first.reasoning == second.reasoning


class TestProvenance:
    async def test_metadata_is_stamped_by_the_service(
        self, settings: Settings, analysis_request: AnalysisRequest
    ) -> None:
        provider = FakeAIProvider(responses=[BUY_RESPONSE], model="fake-model-1")
        decision = await _service(settings, provider).analyze(
            analysis_request, request_id="req-123"
        )

        assert decision.metadata.provider == "fake"
        assert decision.metadata.model == "fake-model-1"
        assert decision.metadata.prompt_name == "technical_analysis"
        assert decision.metadata.prompt_version == "technical_analysis_v1"
        assert decision.metadata.request_id == "req-123"
        assert decision.metadata.depth is AnalysisDepth.ROUTINE
        assert decision.metadata.generated_at.tzinfo is UTC
        assert decision.metadata.latency_ms is not None
        assert decision.metadata.usage is not None

    async def test_a_forged_provider_in_model_output_is_rejected(
        self, settings: Settings, analysis_request: AnalysisRequest
    ) -> None:
        forged = json.dumps({**json.loads(BUY_RESPONSE), "provider": "totally-legit"})
        service = _service(settings, FakeAIProvider(responses=[forged]))
        with pytest.raises(AppError) as excinfo:
            await service.analyze(analysis_request)
        assert excinfo.value.code == "PROVIDER_INVALID_RESPONSE"

    async def test_deep_depth_selects_the_deep_model(
        self, settings: Settings, analysis_request: AnalysisRequest
    ) -> None:
        provider = FakeAIProvider(
            responses=[BUY_RESPONSE], model="routine-model", deep_model="deep-model"
        )
        deep_request = analysis_request.model_copy(update={"depth": AnalysisDepth.DEEP})
        decision = await _service(settings, provider).analyze(deep_request)
        assert decision.metadata.model == "deep-model"
        assert decision.metadata.depth is AnalysisDepth.DEEP


class TestPromptSelection:
    async def test_pinned_prompt_version_is_honoured(
        self, settings: Settings, analysis_request: AnalysisRequest
    ) -> None:
        pinned = analysis_request.model_copy(update={"prompt_version": "technical_analysis_v1"})
        decision = await _service(settings, FakeAIProvider()).analyze(pinned)
        assert decision.metadata.prompt_version == "technical_analysis_v1"

    async def test_unknown_prompt_version_fails_before_any_provider_call(
        self, settings: Settings, analysis_request: AnalysisRequest
    ) -> None:
        provider = FakeAIProvider()
        unknown = analysis_request.model_copy(update={"prompt_version": "technical_analysis_v99"})
        with pytest.raises(ValidationError):
            await _service(settings, provider).analyze(unknown)
        # No money is spent discovering a caller's typo.
        assert provider.calls == []


class TestCostControls:
    async def test_exactly_one_provider_call_per_request(
        self, settings: Settings, analysis_request: AnalysisRequest
    ) -> None:
        provider = FakeAIProvider(responses=[BUY_RESPONSE])
        await _service(settings, provider).analyze(analysis_request)
        assert len(provider.calls) == 1

    async def test_configured_bounds_reach_the_provider(
        self, settings: Settings, analysis_request: AnalysisRequest
    ) -> None:
        provider = FakeAIProvider(responses=[BUY_RESPONSE])
        await _service(settings, provider).analyze(analysis_request)
        call = provider.calls[0]
        assert call.max_output_tokens == settings.ai_max_output_tokens
        assert call.temperature == settings.ai_temperature
        assert call.timeout_seconds == settings.ai_request_timeout_seconds

    async def test_context_limits_bound_the_prompt(
        self, settings: Settings, analysis_request: AnalysisRequest
    ) -> None:
        from decimal import Decimal

        provider = FakeAIProvider(responses=[BUY_RESPONSE])
        wide = analysis_request.model_copy(
            update={"recent_closes": [Decimal(value) for value in range(1000, 1100)]}
        )
        await _service(settings, provider).analyze(wide)
        # settings.context_max_recent_closes is 5 in the test fixture.
        assert "1095" in provider.calls[0].user
        assert "1000," not in provider.calls[0].user


class TestFailures:
    async def test_timeout_is_translated_to_a_gateway_timeout(
        self, settings: Settings, analysis_request: AnalysisRequest
    ) -> None:
        fast = settings.model_copy(update={"ai_request_timeout_seconds": 0.05})
        service = _service(fast, FakeAIProvider(delay_seconds=0.5))
        with pytest.raises(AppError) as excinfo:
            await service.analyze(analysis_request)
        assert excinfo.value.status_code == 504
        assert excinfo.value.code == "PROVIDER_TIMEOUT"

    async def test_provider_timeout_error_is_translated(
        self, settings: Settings, analysis_request: AnalysisRequest
    ) -> None:
        provider = FakeAIProvider(error=ProviderTimeoutError("slow", provider="fake"))
        with pytest.raises(AppError) as excinfo:
            await _service(settings, provider).analyze(analysis_request)
        assert excinfo.value.status_code == 504

    async def test_unavailable_provider_is_translated(
        self, settings: Settings, analysis_request: AnalysisRequest
    ) -> None:
        provider = FakeAIProvider(error=ProviderUnavailableError("down", provider="fake"))
        with pytest.raises(AppError) as excinfo:
            await _service(settings, provider).analyze(analysis_request)
        assert excinfo.value.status_code == 503

    async def test_rate_limit_is_translated(
        self, settings: Settings, analysis_request: AnalysisRequest
    ) -> None:
        provider = FakeAIProvider(
            error=ProviderRateLimitError("slow down", provider="fake", retry_after=12.0)
        )
        with pytest.raises(AppError) as excinfo:
            await _service(settings, provider).analyze(analysis_request)
        assert excinfo.value.status_code == 429
        assert excinfo.value.headers == {"Retry-After": "12"}

    async def test_malformed_output_is_translated(
        self, settings: Settings, analysis_request: AnalysisRequest
    ) -> None:
        provider = FakeAIProvider(responses=["I think you should buy it."])
        with pytest.raises(AppError) as excinfo:
            await _service(settings, provider).analyze(analysis_request)
        assert excinfo.value.status_code == 502
        assert excinfo.value.code == "PROVIDER_INVALID_RESPONSE"
