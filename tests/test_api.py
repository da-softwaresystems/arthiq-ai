"""The HTTP surface: health, authentication, and POST /internal/v1/analyze."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.providers.exceptions import ProviderTimeoutError, ProviderUnavailableError
from app.providers.fake import FakeAIProvider
from app.providers.registry import set_ai_provider
from tests.conftest import TEST_SERVICE_KEY

ANALYZE = "/internal/v1/analyze"
READINESS = "/internal/v1/provider/readiness"

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


class TestHealth:
    def test_health_is_ok(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["service"] == "arthiq-ai"

    def test_health_needs_no_credentials(self, client: TestClient) -> None:
        assert client.get("/health").status_code == 200

    def test_health_does_not_touch_the_provider(
        self, client: TestClient, fake_provider: FakeAIProvider
    ) -> None:
        client.get("/health")
        assert fake_provider.calls == []

    def test_health_returns_a_request_id(self, client: TestClient) -> None:
        assert client.get("/health").headers["X-Request-ID"]


class TestAuthentication:
    def test_missing_key_is_rejected(self, client: TestClient, request_payload: dict) -> None:
        response = client.post(ANALYZE, json=request_payload)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"

    def test_wrong_key_is_rejected(self, client: TestClient, request_payload: dict) -> None:
        response = client.post(ANALYZE, json=request_payload, headers={"X-API-Key": "not-the-key"})
        assert response.status_code == 401

    def test_rejection_happens_before_the_provider_is_called(
        self, client: TestClient, request_payload: dict, fake_provider: FakeAIProvider
    ) -> None:
        client.post(ANALYZE, json=request_payload)
        assert fake_provider.calls == []

    def test_rejection_does_not_echo_the_presented_key(
        self, client: TestClient, request_payload: dict
    ) -> None:
        presented = "some-guessed-key-value"
        response = client.post(ANALYZE, json=request_payload, headers={"X-API-Key": presented})
        assert presented not in response.text

    def test_readiness_also_requires_the_key(self, client: TestClient) -> None:
        assert client.get(READINESS).status_code == 401

    def test_unconfigured_service_key_refuses_every_call(
        self, request_payload: dict, fake_provider: FakeAIProvider
    ) -> None:
        # Fails closed: no key configured means no caller is trusted, rather
        # than every caller being trusted.
        settings = Settings(_env_file=None, app_env="test", ai_provider="fake")
        with TestClient(create_app(settings)) as client:
            response = client.post(
                ANALYZE, json=request_payload, headers={"X-API-Key": TEST_SERVICE_KEY}
            )
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "CONFIGURATION_ERROR"

    def test_key_rotation_accepts_both_keys(self, request_payload: dict) -> None:
        settings = Settings(
            _env_file=None,
            app_env="test",
            ai_provider="fake",
            ai_service_api_key="old-key-value,new-key-value",
        )
        set_ai_provider(FakeAIProvider(responses=[BUY_RESPONSE, BUY_RESPONSE]))
        try:
            with TestClient(create_app(settings)) as client:
                for key in ("old-key-value", "new-key-value"):
                    response = client.post(
                        ANALYZE, json=request_payload, headers={"X-API-Key": key}
                    )
                    assert response.status_code == 200
        finally:
            set_ai_provider(None)


class TestAnalyze:
    def test_successful_analysis(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        request_payload: dict,
        fake_provider: FakeAIProvider,
    ) -> None:
        set_ai_provider(FakeAIProvider(responses=[BUY_RESPONSE]))
        response = client.post(ANALYZE, json=request_payload, headers=auth_headers)

        assert response.status_code == 200
        body = response.json()
        assert body["decision"] == "BUY"
        assert body["confidence"] == 0.78
        assert body["risk_level"] == "MEDIUM"
        assert body["invalidating_conditions"] == ["Close below EMA50"]
        assert body["metadata"]["provider"] == "fake"
        assert body["metadata"]["model"]
        assert body["metadata"]["prompt_version"] == "technical_analysis_v1"
        assert body["metadata"]["generated_at"]

    def test_response_carries_the_callers_correlation_id(
        self, client: TestClient, auth_headers: dict[str, str], request_payload: dict
    ) -> None:
        headers = {**auth_headers, "X-Request-ID": "backend-correlation-1"}
        response = client.post(ANALYZE, json=request_payload, headers=headers)
        assert response.status_code == 200
        assert response.json()["metadata"]["request_id"] == "backend-correlation-1"
        assert response.headers["X-Request-ID"] == "backend-correlation-1"

    def test_invalid_request_is_a_validation_error(
        self, client: TestClient, auth_headers: dict[str, str], request_payload: dict
    ) -> None:
        response = client.post(
            ANALYZE, json={**request_payload, "price": "-5"}, headers=auth_headers
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_unknown_field_is_a_validation_error(
        self, client: TestClient, auth_headers: dict[str, str], request_payload: dict
    ) -> None:
        response = client.post(
            ANALYZE, json={**request_payload, "user_id": "firebase-uid"}, headers=auth_headers
        )
        assert response.status_code == 422

    def test_unknown_prompt_version_is_a_validation_error(
        self, client: TestClient, auth_headers: dict[str, str], request_payload: dict
    ) -> None:
        response = client.post(
            ANALYZE,
            json={**request_payload, "prompt_version": "technical_analysis_v99"},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_provider_timeout_becomes_504(
        self, client: TestClient, auth_headers: dict[str, str], request_payload: dict
    ) -> None:
        set_ai_provider(FakeAIProvider(error=ProviderTimeoutError("slow", provider="fake")))
        response = client.post(ANALYZE, json=request_payload, headers=auth_headers)
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "PROVIDER_TIMEOUT"

    def test_provider_outage_becomes_503(
        self, client: TestClient, auth_headers: dict[str, str], request_payload: dict
    ) -> None:
        set_ai_provider(FakeAIProvider(error=ProviderUnavailableError("down", provider="fake")))
        response = client.post(ANALYZE, json=request_payload, headers=auth_headers)
        assert response.status_code == 503

    def test_malformed_provider_output_becomes_502(
        self, client: TestClient, auth_headers: dict[str, str], request_payload: dict
    ) -> None:
        set_ai_provider(FakeAIProvider(responses=["buy it, trust me"]))
        response = client.post(ANALYZE, json=request_payload, headers=auth_headers)
        assert response.status_code == 502
        assert response.json()["error"]["code"] == "PROVIDER_INVALID_RESPONSE"

    def test_errors_use_one_envelope(
        self, client: TestClient, auth_headers: dict[str, str], request_payload: dict
    ) -> None:
        set_ai_provider(FakeAIProvider(error=ProviderUnavailableError("down", provider="fake")))
        body = client.post(ANALYZE, json=request_payload, headers=auth_headers).json()
        assert set(body) == {"error"}
        assert set(body["error"]) >= {"code", "message"}

    def test_error_response_leaks_no_stack_trace(
        self, client: TestClient, auth_headers: dict[str, str], request_payload: dict
    ) -> None:
        set_ai_provider(
            FakeAIProvider(error=ProviderUnavailableError("connection to 10.0.0.5 refused"))
        )
        text = client.post(ANALYZE, json=request_payload, headers=auth_headers).text
        assert "Traceback" not in text
        assert "10.0.0.5" not in text


class TestSecrets:
    def test_no_secret_appears_in_a_successful_response(
        self, client: TestClient, auth_headers: dict[str, str], request_payload: dict
    ) -> None:
        set_ai_provider(FakeAIProvider(responses=[BUY_RESPONSE]))
        response = client.post(ANALYZE, json=request_payload, headers=auth_headers)
        assert TEST_SERVICE_KEY not in response.text

    def test_no_secret_appears_in_an_error_response(
        self, client: TestClient, auth_headers: dict[str, str], request_payload: dict
    ) -> None:
        set_ai_provider(FakeAIProvider(error=ProviderUnavailableError("down", provider="fake")))
        response = client.post(ANALYZE, json=request_payload, headers=auth_headers)
        assert TEST_SERVICE_KEY not in response.text

    def test_openapi_does_not_expose_secrets(self, client: TestClient) -> None:
        assert TEST_SERVICE_KEY not in client.get("/openapi.json").text


class TestReadiness:
    def test_readiness_reports_provider_and_prompts(
        self, client: TestClient, auth_headers: dict[str, str], fake_provider: FakeAIProvider
    ) -> None:
        response = client.get(READINESS, headers=auth_headers)
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "fake"
        assert body["ready"] is True
        assert body["default_prompt_version"] == "technical_analysis_v1"
        assert "technical_analysis_v1" in body["prompt_versions"]

    def test_readiness_runs_no_inference(
        self, client: TestClient, auth_headers: dict[str, str], fake_provider: FakeAIProvider
    ) -> None:
        client.get(READINESS, headers=auth_headers)
        assert fake_provider.calls == []
