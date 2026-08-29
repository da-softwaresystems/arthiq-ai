"""M5.6: honouring the caller's X-Deadline-Ms deadline.

M5.5 showed the failure this prevents: the backend gave up at ~30s while Ollama
kept generating for ~250s. The rule these tests pin down is that the layer doing
the work gives up first, and never runs a provider call nobody is waiting for.

No real inference here - everything runs against FakeAIProvider.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.analysis.service import AnalysisService
from app.core.config import Settings
from app.core.deadline import DEADLINE_HEADER, MAX_DEADLINE_MS, parse_deadline_ms, resolve_budget
from app.core.exceptions import AppError
from app.domain.analysis import AnalysisRequest
from app.providers.fake import FakeAIProvider
from app.providers.registry import set_ai_provider

ANALYZE = "/internal/v1/analyze"

BUY_RESPONSE = json.dumps(
    {
        "decision": "BUY",
        "confidence": 0.78,
        "risk_level": "MEDIUM",
        "reasoning": "Price holds above EMA20 and EMA50 with a positive MACD.",
        "key_factors": ["price above EMA50"],
        "invalidating_conditions": ["Close below EMA50"],
    }
)


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "app_env": "test",
        "ai_provider": "fake",
        "ai_service_api_key": "test-service-key-0123456789",
        "ai_request_timeout_seconds": 30.0,
        "ai_deadline_safety_margin_ms": 250,
    }
    return Settings(_env_file=None, **{**base, **overrides})


class TestHeaderParsing:
    """Validating the header safely: a bad value must not break the request."""

    def test_absent_header(self) -> None:
        assert parse_deadline_ms(None) is None

    def test_valid_value(self) -> None:
        assert parse_deadline_ms("25000") == 25_000

    def test_surrounding_whitespace_is_tolerated(self) -> None:
        assert parse_deadline_ms(" 25000 ") == 25_000

    @pytest.mark.parametrize(
        "raw",
        ["", "   ", "abc", "25.5", "25_000", "1e4", "NaN", "12,000", "30s"],
    )
    def test_malformed_values_fall_back_to_no_deadline(self, raw: str) -> None:
        # Lenient by design: refusing a valid analysis because an optional
        # header was malformed would trade a working request for a broken one.
        assert parse_deadline_ms(raw) is None

    def test_absurd_value_is_not_believed(self) -> None:
        assert parse_deadline_ms(str(MAX_DEADLINE_MS + 1)) is None

    def test_expired_values_are_parsed_not_discarded(self) -> None:
        # Zero and below are not malformed - they mean "already too late", and
        # the caller is told so rather than being silently given a full budget.
        assert parse_deadline_ms("0") == 0
        assert parse_deadline_ms("-500") == -500


class TestBudgetClamping:
    CONFIGURED = 30.0

    def test_no_deadline_keeps_the_configured_timeout(self) -> None:
        budget = resolve_budget(configured_timeout_seconds=self.CONFIGURED, deadline_ms=None)
        assert budget.seconds == self.CONFIGURED
        assert budget.expired is False
        assert budget.from_deadline is False

    def test_deadline_shorter_than_configured_wins(self) -> None:
        budget = resolve_budget(
            configured_timeout_seconds=self.CONFIGURED,
            deadline_ms=10_000,
            safety_margin_seconds=0.25,
        )
        assert budget.seconds == pytest.approx(9.75)
        assert budget.from_deadline is True

    def test_configured_shorter_than_deadline_wins(self) -> None:
        # The service never waits longer than its own configuration just
        # because the caller is patient.
        budget = resolve_budget(
            configured_timeout_seconds=self.CONFIGURED,
            deadline_ms=120_000,
            safety_margin_seconds=0.25,
        )
        assert budget.seconds == self.CONFIGURED
        assert budget.from_deadline is False

    def test_elapsed_time_comes_out_of_the_budget(self) -> None:
        budget = resolve_budget(
            configured_timeout_seconds=self.CONFIGURED,
            deadline_ms=10_000,
            elapsed_seconds=2.0,
            safety_margin_seconds=0.25,
        )
        assert budget.seconds == pytest.approx(7.75)

    @pytest.mark.parametrize(
        ("deadline_ms", "elapsed"),
        [(0, 0.0), (-1_000, 0.0), (200, 0.0), (5_000, 10.0), (250, 0.0)],
    )
    def test_expired_deadline_yields_no_budget(self, deadline_ms: int, elapsed: float) -> None:
        budget = resolve_budget(
            configured_timeout_seconds=self.CONFIGURED,
            deadline_ms=deadline_ms,
            elapsed_seconds=elapsed,
            safety_margin_seconds=0.25,
        )
        assert budget.expired is True
        assert budget.seconds == 0.0

    @pytest.mark.parametrize("deadline_ms", [-100_000, -1, 0, 1, 100, 249, 250, 251, 30_000])
    def test_effective_timeout_is_never_negative(self, deadline_ms: int) -> None:
        budget = resolve_budget(
            configured_timeout_seconds=self.CONFIGURED,
            deadline_ms=deadline_ms,
            elapsed_seconds=0.5,
            safety_margin_seconds=0.25,
        )
        assert budget.seconds >= 0.0
        assert budget.expired is (budget.seconds == 0.0)


class TestServiceHonoursTheDeadline:
    async def test_provider_receives_the_clamped_timeout(
        self, analysis_request: AnalysisRequest
    ) -> None:
        provider = FakeAIProvider(responses=[BUY_RESPONSE])
        settings = _settings(ai_request_timeout_seconds=30.0)

        await AnalysisService(provider, settings).analyze(analysis_request, deadline_ms=8_000)

        # 8s announced, 0.25s margin, minus the microseconds spent building the
        # context - well under the configured 30s.
        assert provider.calls[0].timeout_seconds == pytest.approx(7.75, abs=0.2)
        assert provider.calls[0].timeout_seconds < 30.0

    async def test_configured_timeout_still_caps_a_generous_deadline(
        self, analysis_request: AnalysisRequest
    ) -> None:
        provider = FakeAIProvider(responses=[BUY_RESPONSE])
        settings = _settings(ai_request_timeout_seconds=30.0)

        await AnalysisService(provider, settings).analyze(analysis_request, deadline_ms=300_000)

        assert provider.calls[0].timeout_seconds == 30.0

    async def test_no_deadline_preserves_existing_behaviour(
        self, analysis_request: AnalysisRequest
    ) -> None:
        provider = FakeAIProvider(responses=[BUY_RESPONSE])
        settings = _settings(ai_request_timeout_seconds=30.0)

        decision = await AnalysisService(provider, settings).analyze(analysis_request)

        assert provider.calls[0].timeout_seconds == 30.0
        assert decision.decision.value == "BUY"

    async def test_expired_deadline_makes_no_provider_call(
        self, analysis_request: AnalysisRequest
    ) -> None:
        provider = FakeAIProvider(responses=[BUY_RESPONSE])
        settings = _settings()

        with pytest.raises(AppError) as excinfo:
            await AnalysisService(provider, settings).analyze(analysis_request, deadline_ms=-1)

        # Existing normalised timeout semantics, and not one token spent.
        assert excinfo.value.status_code == 504
        assert excinfo.value.code == "PROVIDER_TIMEOUT"
        assert provider.calls == []


class TestApiContract:
    def test_valid_deadline_header_is_honoured(
        self, client: TestClient, auth_headers: dict[str, str], request_payload: dict
    ) -> None:
        provider = FakeAIProvider(responses=[BUY_RESPONSE])
        set_ai_provider(provider)

        response = client.post(
            ANALYZE,
            json=request_payload,
            headers={**auth_headers, DEADLINE_HEADER: "9000"},
        )

        assert response.status_code == 200
        assert provider.calls[0].timeout_seconds < 9.0

    def test_expired_deadline_header_returns_the_existing_timeout_error(
        self, client: TestClient, auth_headers: dict[str, str], request_payload: dict
    ) -> None:
        provider = FakeAIProvider(responses=[BUY_RESPONSE])
        set_ai_provider(provider)

        response = client.post(
            ANALYZE, json=request_payload, headers={**auth_headers, DEADLINE_HEADER: "0"}
        )

        assert response.status_code == 504
        assert response.json()["error"]["code"] == "PROVIDER_TIMEOUT"
        assert provider.calls == []

    def test_malformed_deadline_header_does_not_fail_the_request(
        self, client: TestClient, auth_headers: dict[str, str], request_payload: dict
    ) -> None:
        provider = FakeAIProvider(responses=[BUY_RESPONSE])
        set_ai_provider(provider)

        response = client.post(
            ANALYZE,
            json=request_payload,
            headers={**auth_headers, DEADLINE_HEADER: "not-a-number"},
        )

        # Falls back to the configured timeout: pre-M5.6 behaviour.
        assert response.status_code == 200
        assert provider.calls[0].timeout_seconds == 5.0  # the test fixture's configured value

    def test_deadline_header_still_requires_authentication(
        self, client: TestClient, request_payload: dict, fake_provider: FakeAIProvider
    ) -> None:
        # A deadline is not a credential: it must not open the endpoint.
        response = client.post(ANALYZE, json=request_payload, headers={DEADLINE_HEADER: "9000"})
        assert response.status_code == 401
        assert fake_provider.calls == []

    def test_response_contract_is_unchanged(
        self, client: TestClient, auth_headers: dict[str, str], request_payload: dict
    ) -> None:
        set_ai_provider(FakeAIProvider(responses=[BUY_RESPONSE]))

        response = client.post(
            ANALYZE,
            json=request_payload,
            headers={**auth_headers, DEADLINE_HEADER: "9000", "X-Request-ID": "deadline-corr-1"},
        )

        body = response.json()
        assert set(body) == {
            "decision",
            "confidence",
            "risk_level",
            "reasoning",
            "key_factors",
            "invalidating_conditions",
            "metadata",
        }
        assert set(body["metadata"]) == {
            "provider",
            "model",
            "prompt_name",
            "prompt_version",
            "depth",
            "generated_at",
            "request_id",
            "latency_ms",
            "usage",
        }
        # Correlation logging and echo survive the change.
        assert body["metadata"]["request_id"] == "deadline-corr-1"
        assert response.headers["X-Request-ID"] == "deadline-corr-1"

    def test_no_secret_leaks_when_a_deadline_expires(
        self, client: TestClient, auth_headers: dict[str, str], request_payload: dict
    ) -> None:
        set_ai_provider(FakeAIProvider(responses=[BUY_RESPONSE]))
        response = client.post(
            ANALYZE, json=request_payload, headers={**auth_headers, DEADLINE_HEADER: "-5"}
        )
        assert auth_headers["X-API-Key"] not in response.text
