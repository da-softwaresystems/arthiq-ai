"""Live Gemini verification. Opt-in, and deliberately tiny.

One request, one model, bounded output. Every run of this file costs money, so
it stays minimal by design and is never part of the normal suite or of CI.

Run it with::

    AI_PROVIDER=gemini GEMINI_API_KEY=... GEMINI_MODEL=gemini-2.5-flash-lite \
        pytest -m integration tests/integration/test_gemini_live.py

It skips unless all three are set.
"""

from __future__ import annotations

import os

import pytest

from app.analysis.service import AnalysisService
from app.core.config import Settings
from app.domain.analysis import AnalysisRequest
from app.domain.enums import Decision, RiskLevel
from app.providers.gemini import GeminiProvider

pytestmark = pytest.mark.integration


def _settings() -> Settings:
    if os.environ.get("AI_PROVIDER") != "gemini":
        pytest.skip("AI_PROVIDER is not gemini")
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not set")
    if not os.environ.get("GEMINI_MODEL"):
        pytest.skip("GEMINI_MODEL is not set")
    return Settings(
        ai_service_api_key="integration-test-key",
        # A small ceiling: this test proves the contract, not the model.
        ai_max_output_tokens=512,
    )


async def test_one_minimal_live_analysis(analysis_request: AnalysisRequest) -> None:
    """The single billed call in this file."""
    settings = _settings()
    provider = GeminiProvider(settings)
    try:
        decision = await AnalysisService(provider, settings).analyze(
            analysis_request, request_id="gemini-integration"
        )
    finally:
        await provider.aclose()

    assert decision.decision in set(Decision)
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.risk_level in set(RiskLevel)
    assert decision.metadata.provider == "gemini"
    assert decision.metadata.model == settings.gemini_model or decision.metadata.model
    assert decision.metadata.prompt_version == "technical_analysis_v1"
    if decision.decision is not Decision.HOLD:
        assert decision.invalidating_conditions


async def test_readiness_costs_nothing() -> None:
    """Readiness is configuration-only, so it can be polled without billing."""
    settings = _settings()
    provider = GeminiProvider(settings)
    try:
        readiness = await provider.check_readiness()
    finally:
        await provider.aclose()
    assert readiness.ready is True
    assert readiness.provider == "gemini"
    assert "not verified" in readiness.detail
