"""Live Ollama verification. Opt-in; excluded from the default run.

Run it with a local Ollama serving the configured model::

    ollama serve
    ollama pull <model>
    OLLAMA_MODEL=<model> pytest -m integration tests/integration/test_ollama_live.py

The test is skipped - not failed - when Ollama is not running or the model is
not pulled, so a developer without Ollama sees no red.
"""

from __future__ import annotations

import os

import httpx
import pytest

from app.analysis.output import parse_decision_draft
from app.analysis.service import AnalysisService
from app.context.builder import ContextLimits, build_context
from app.core.config import Settings
from app.domain.analysis import AnalysisRequest
from app.domain.enums import Decision, RiskLevel
from app.prompts import get_prompt
from app.providers.base import CompletionRequest
from app.providers.ollama import OllamaProvider

pytestmark = pytest.mark.integration


def _settings() -> Settings:
    # Resolved settings, not os.environ: OLLAMA_MODEL is normally set in .env,
    # which is how the service itself is configured.
    settings = Settings(ai_provider="ollama", ai_service_api_key="integration-test-key")
    if not settings.ollama_model:
        pytest.skip("OLLAMA_MODEL is not configured (environment or .env)")
    # A local model on CPU is slow. OLLAMA_TIMEOUT_SECONDS overrides the
    # configured budget for this run; otherwise whatever .env says applies.
    override = os.environ.get("OLLAMA_TIMEOUT_SECONDS")
    if override:
        settings = settings.model_copy(update={"ai_request_timeout_seconds": float(override)})
    return settings


@pytest.fixture
async def provider() -> OllamaProvider:
    settings = _settings()
    provider = OllamaProvider(settings)
    readiness = await provider.check_readiness()
    if not readiness.ready:
        await provider.aclose()
        pytest.skip(f"Ollama not ready: {readiness.detail}")
    return provider


async def test_ollama_is_reachable(provider: OllamaProvider) -> None:
    """1. Connect to the configured OLLAMA_BASE_URL."""
    readiness = await provider.check_readiness()
    assert readiness.ready is True
    assert readiness.provider == "ollama"
    await provider.aclose()


async def test_ollama_returns_a_valid_trading_decision(
    provider: OllamaProvider, analysis_request: AnalysisRequest
) -> None:
    """2-5. Configured model, small deterministic request, validated decision."""
    settings = _settings()
    try:
        decision = await AnalysisService(provider, settings).analyze(
            analysis_request, request_id="ollama-integration"
        )
    finally:
        await provider.aclose()

    assert decision.decision in set(Decision)
    assert 0.0 <= decision.confidence <= 1.0
    assert decision.risk_level in set(RiskLevel)
    assert decision.reasoning
    assert decision.metadata.provider == "ollama"
    assert decision.metadata.model
    assert decision.metadata.prompt_version == "technical_analysis_v1"
    if decision.decision is not Decision.HOLD:
        assert decision.invalidating_conditions


async def test_raw_completion_validates_into_a_draft(
    provider: OllamaProvider, analysis_request: AnalysisRequest
) -> None:
    """The provider's raw text passes our output validation unaided."""
    settings = _settings()
    prompt = get_prompt()
    context = build_context(analysis_request, ContextLimits.from_settings(settings))
    rendered = prompt.render(context.render())

    try:
        result = await provider.complete(
            CompletionRequest(
                system=rendered.system,
                user=rendered.user,
                max_output_tokens=settings.ai_max_output_tokens,
                temperature=settings.ai_temperature,
                timeout_seconds=settings.ai_request_timeout_seconds,
            )
        )
    finally:
        await provider.aclose()

    draft = parse_decision_draft(result.text, provider=result.provider, model=result.model)
    assert draft.decision in set(Decision)
    assert result.usage is not None


async def test_unpulled_model_is_reported_cleanly() -> None:
    """A missing model is a clean provider error, not a stack trace."""
    settings = _settings()
    base_url = settings.ollama_base_url
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.get(f"{base_url}/api/tags")
    except httpx.HTTPError:
        pytest.skip("Ollama is not running")

    missing = Settings(
        ai_provider="ollama",
        ollama_base_url=base_url,
        ollama_model="definitely-not-a-real-model-9999",
    )
    provider = OllamaProvider(missing)
    readiness = await provider.check_readiness()
    await provider.aclose()
    assert readiness.ready is False
