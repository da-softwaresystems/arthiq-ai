"""Output validation: nothing becomes a decision without passing this gate."""

from __future__ import annotations

import json

import pytest

from app.analysis.output import MAX_RESPONSE_CHARS, parse_decision_draft
from app.domain.enums import Decision, RiskLevel
from app.providers.exceptions import ProviderResponseError

VALID = {
    "decision": "BUY",
    "confidence": 0.72,
    "risk_level": "MEDIUM",
    "reasoning": "Price is above EMA20 and EMA50 with RSI-14 at 54.",
    "key_factors": ["price above EMA50"],
    "invalidating_conditions": ["Close below EMA50"],
}


def _parse(text: str):
    return parse_decision_draft(text, provider="fake", model="fake-model")


class TestAcceptedShapes:
    def test_plain_json(self) -> None:
        draft = _parse(json.dumps(VALID))
        assert draft.decision is Decision.BUY
        assert draft.risk_level is RiskLevel.MEDIUM

    def test_fenced_json(self) -> None:
        assert _parse(f"```json\n{json.dumps(VALID)}\n```").decision is Decision.BUY

    def test_json_wrapped_in_prose(self) -> None:
        text = f"Here is my analysis:\n{json.dumps(VALID)}\nHope that helps."
        assert _parse(text).decision is Decision.BUY

    def test_lower_case_enums_are_normalised(self) -> None:
        draft = _parse(json.dumps({**VALID, "decision": "buy", "risk_level": "medium"}))
        assert draft.decision is Decision.BUY
        assert draft.risk_level is RiskLevel.MEDIUM

    def test_nested_braces_do_not_break_extraction(self) -> None:
        payload = {**VALID, "reasoning": "It closed above {EMA50} today."}
        assert _parse(f"prefix {json.dumps(payload)} suffix").reasoning.endswith("today.")


class TestRejectedShapes:
    def test_empty_response(self) -> None:
        with pytest.raises(ProviderResponseError, match="empty"):
            _parse("   ")

    def test_prose_with_no_json(self) -> None:
        with pytest.raises(ProviderResponseError, match="no JSON object"):
            _parse("I would probably buy this one.")

    def test_broken_json(self) -> None:
        with pytest.raises(ProviderResponseError, match="not valid JSON"):
            _parse('{"decision": "BUY", ')

    def test_json_array(self) -> None:
        with pytest.raises(ProviderResponseError, match="not a JSON object"):
            _parse('["BUY"]')

    def test_oversized_response(self) -> None:
        with pytest.raises(ProviderResponseError, match="exceeds"):
            _parse("x" * (MAX_RESPONSE_CHARS + 1))

    def test_invalid_decision_value(self) -> None:
        with pytest.raises(ProviderResponseError, match="decision"):
            _parse(json.dumps({**VALID, "decision": "STRONG BUY"}))

    def test_confidence_as_a_percentage(self) -> None:
        with pytest.raises(ProviderResponseError, match="confidence"):
            _parse(json.dumps({**VALID, "confidence": 78}))

    def test_invalid_risk_level(self) -> None:
        with pytest.raises(ProviderResponseError, match="risk_level"):
            _parse(json.dumps({**VALID, "risk_level": "EXTREME"}))

    def test_missing_reasoning(self) -> None:
        payload = {key: value for key, value in VALID.items() if key != "reasoning"}
        with pytest.raises(ProviderResponseError, match="reasoning"):
            _parse(json.dumps(payload))

    def test_actionable_call_without_invalidating_conditions(self) -> None:
        with pytest.raises(ProviderResponseError, match="invalidating condition"):
            _parse(json.dumps({**VALID, "invalidating_conditions": []}))

    def test_model_supplied_metadata_is_refused(self) -> None:
        # A model claiming its own provider/model/prompt version must not be
        # able to write provenance. The extra keys are rejected outright.
        payload = {**VALID, "provider": "gemini", "prompt_version": "technical_analysis_v1"}
        with pytest.raises(ProviderResponseError, match="Extra inputs"):
            _parse(json.dumps(payload))

    def test_error_message_does_not_echo_model_output(self) -> None:
        secret_ish = "SENSITIVE-MODEL-TEXT"
        payload = {**VALID, "decision": secret_ish}
        with pytest.raises(ProviderResponseError) as excinfo:
            _parse(json.dumps(payload))
        assert secret_ish not in str(excinfo.value)
