"""TradingDecision and DecisionDraft validation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.domain.decision import (
    MAX_REASONING_CHARS,
    DecisionDraft,
    DecisionMetadata,
    TokenUsage,
    TradingDecision,
)
from app.domain.enums import AnalysisDepth, Decision, RiskLevel

DRAFT: dict[str, object] = {
    "decision": "BUY",
    "confidence": 0.78,
    "risk_level": "MEDIUM",
    "reasoning": "Price holds above both EMAs and RSI-14 is neutral at 54.",
    "invalidating_conditions": ["Close below EMA50"],
}


def _draft(**overrides: object) -> DecisionDraft:
    return DecisionDraft.model_validate({**DRAFT, **overrides})


def _metadata(**overrides: object) -> DecisionMetadata:
    base: dict[str, object] = {
        "provider": "fake",
        "model": "fake-deterministic-v1",
        "prompt_name": "technical_analysis",
        "prompt_version": "technical_analysis_v1",
        "depth": AnalysisDepth.ROUTINE,
        "generated_at": datetime(2026, 8, 21, 10, 0, tzinfo=UTC),
    }
    return DecisionMetadata.model_validate({**base, **overrides})


class TestDecisionValues:
    def test_buy(self) -> None:
        assert _draft(decision="BUY").decision is Decision.BUY

    def test_sell(self) -> None:
        draft = _draft(decision="SELL", invalidating_conditions=["Close above EMA20"])
        assert draft.decision is Decision.SELL

    def test_hold_needs_no_invalidating_condition(self) -> None:
        draft = _draft(decision="HOLD", invalidating_conditions=[])
        assert draft.decision is Decision.HOLD

    @pytest.mark.parametrize("value", ["STRONG BUY", "buy!", "ACCUMULATE", "EXECUTE", ""])
    def test_arbitrary_decision_strings_are_rejected(self, value: str) -> None:
        with pytest.raises(ValidationError):
            _draft(decision=value)

    def test_there_is_no_execute_decision(self) -> None:
        assert set(Decision) == {Decision.BUY, Decision.SELL, Decision.HOLD}


class TestConfidence:
    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_bounds_are_inclusive(self, value: float) -> None:
        assert _draft(confidence=value).confidence == value

    @pytest.mark.parametrize("value", [-0.01, 1.01, 78, 100])
    def test_out_of_range_confidence_is_rejected(self, value: float) -> None:
        # 78 is the common model mistake (a percentage). It is rejected rather
        # than rescaled: guessing at intent would invent a confidence.
        with pytest.raises(ValidationError):
            _draft(confidence=value)

    def test_non_numeric_confidence_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _draft(confidence="high")


class TestRiskLevel:
    @pytest.mark.parametrize("value", ["LOW", "MEDIUM", "HIGH"])
    def test_accepted_levels(self, value: str) -> None:
        assert _draft(risk_level=value).risk_level is RiskLevel(value)

    @pytest.mark.parametrize("value", ["EXTREME", "medium-high", "3"])
    def test_other_levels_are_rejected(self, value: str) -> None:
        with pytest.raises(ValidationError):
            _draft(risk_level=value)


class TestControls:
    def test_actionable_call_requires_an_invalidating_condition(self) -> None:
        with pytest.raises(ValidationError, match="invalidating condition"):
            _draft(decision="BUY", invalidating_conditions=[])

    def test_blank_condition_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="blank"):
            _draft(invalidating_conditions=["   "])

    def test_reasoning_is_required(self) -> None:
        with pytest.raises(ValidationError):
            _draft(reasoning="")

    def test_reasoning_is_capped(self) -> None:
        with pytest.raises(ValidationError):
            _draft(reasoning="x" * (MAX_REASONING_CHARS + 1))

    def test_a_model_cannot_supply_its_own_provenance(self) -> None:
        # The draft is what the model said. Provider, model and prompt version
        # are stamped by the service, so they are unknown fields here.
        with pytest.raises(ValidationError, match="Extra inputs"):
            _draft(provider="gemini", model="gemini-2.5-pro")


class TestTradingDecision:
    def test_from_draft_carries_judgement_and_provenance(self) -> None:
        decision = TradingDecision.from_draft(_draft(), _metadata())
        assert decision.decision is Decision.BUY
        assert decision.confidence == 0.78
        assert decision.metadata.provider == "fake"
        assert decision.metadata.prompt_version == "technical_analysis_v1"

    def test_decision_is_immutable(self) -> None:
        decision = TradingDecision.from_draft(_draft(), _metadata())
        with pytest.raises(ValidationError):
            decision.decision = Decision.SELL  # type: ignore[misc]

    def test_metadata_carries_usage_and_correlation(self) -> None:
        metadata = _metadata(
            request_id="abc123",
            latency_ms=420,
            usage=TokenUsage(prompt_tokens=300, completion_tokens=120, total_tokens=420),
        )
        decision = TradingDecision.from_draft(_draft(), metadata)
        assert decision.metadata.request_id == "abc123"
        assert decision.metadata.latency_ms == 420
        assert decision.metadata.usage is not None
        assert decision.metadata.usage.total_tokens == 420

    def test_generated_at_is_timezone_aware(self) -> None:
        decision = TradingDecision.from_draft(_draft(), _metadata())
        assert decision.metadata.generated_at.tzinfo is not None

    def test_serialises_to_the_documented_shape(self) -> None:
        payload = TradingDecision.from_draft(_draft(), _metadata()).model_dump(mode="json")
        assert payload["decision"] == "BUY"
        assert payload["risk_level"] == "MEDIUM"
        assert payload["invalidating_conditions"] == ["Close below EMA50"]
        assert payload["metadata"]["prompt_version"] == "technical_analysis_v1"
