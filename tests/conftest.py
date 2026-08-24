"""Shared fixtures.

Three rules hold for everything under ``tests/`` except ``tests/integration``:

* no network - the only provider used is
  :class:`~app.providers.fake.FakeAIProvider`;
* no secrets - the service key is a literal test value, and no vendor key is
  ever read from the environment; and
* no clock or randomness in an assertion - the same test gives the same answer
  on every machine.

The environment is cleared of ``ARTHIQ``-ish variables before settings are
built, so a developer's real ``.env`` cannot change a test result.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.domain.analysis import (
    AnalysisRequest,
    BollingerValues,
    MacdValues,
    MarketContext,
    Observation,
    TechnicalIndicators,
)
from app.domain.enums import CandleInterval, Exchange, MarketTrend
from app.main import create_app
from app.providers.fake import FakeAIProvider
from app.providers.registry import set_ai_provider

TEST_SERVICE_KEY = "test-service-key-0123456789"

#: Every environment variable the settings object reads. Cleared per test so
#: the suite is hermetic.
_ENV_KEYS = (
    "APP_ENV",
    "AI_PROVIDER",
    "AI_SERVICE_API_KEY",
    "AI_REQUEST_TIMEOUT_SECONDS",
    "AI_MAX_OUTPUT_TOKENS",
    "AI_TEMPERATURE",
    "CONTEXT_MAX_RECENT_CLOSES",
    "CONTEXT_MAX_OBSERVATIONS",
    "LOG_PROMPTS",
    "LOG_PROVIDER_RESPONSES",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OLLAMA_DEEP_MODEL",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_DEEP_MODEL",
    "GEMINI_BASE_URL",
)


@pytest.fixture(autouse=True)
def _clean_env(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic unit tests; integration tests keep their real configuration."""
    if request.node.get_closest_marker("integration"):
        return
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def settings() -> Settings:
    """Settings for a test run: fake provider, known key, tight bounds."""
    return Settings(
        app_env="test",
        # ``_env_file=None`` keeps a developer's local .env out of the suite.
        _env_file=None,
        ai_provider="fake",
        ai_service_api_key=TEST_SERVICE_KEY,
        ai_request_timeout_seconds=5.0,
        context_max_recent_closes=5,
        context_max_observations=3,
    )


@pytest.fixture
def fake_provider() -> Iterator[FakeAIProvider]:
    """A provider installed in the registry for the duration of one test."""
    provider = FakeAIProvider()
    set_ai_provider(provider)
    yield provider
    set_ai_provider(None)


@pytest.fixture
def client(settings: Settings, fake_provider: FakeAIProvider) -> Iterator[TestClient]:
    """A TestClient wired to the fake provider and the test settings."""
    app = create_app(settings)
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"X-API-Key": TEST_SERVICE_KEY}


@pytest.fixture
def analysis_request() -> AnalysisRequest:
    """A representative request: RELIANCE on a daily bar, mid-range RSI."""
    return AnalysisRequest(
        symbol="RELIANCE",
        exchange=Exchange.NSE,
        interval=CandleInterval.ONE_DAY,
        as_of=datetime(2026, 8, 21, 10, 0, tzinfo=UTC),
        trading_date=date(2026, 8, 21),
        price=Decimal("1316.0000"),
        technical=TechnicalIndicators(
            rsi_14=Decimal("54.20"),
            ema_20=Decimal("1308.40"),
            ema_50=Decimal("1297.80"),
            macd=MacdValues(line=Decimal("4.21"), signal=Decimal("3.10")),
            atr_14=Decimal("18.40"),
            bollinger=BollingerValues(
                upper=Decimal("1340.00"), middle=Decimal("1310.00"), lower=Decimal("1280.00")
            ),
        ),
        market=MarketContext(
            trend=MarketTrend.SIDEWAYS,
            index_symbol="NIFTY50",
            index_change_percent=Decimal("0.35"),
            sector="Energy",
        ),
        recent_closes=[Decimal("1300.00"), Decimal("1305.50"), Decimal("1316.00")],
        observations=[
            Observation(
                observed_at=datetime(2026, 8, 20, 10, 0, tzinfo=UTC),
                kind="volume",
                summary="Volume 1.4x the 20-day average",
            )
        ],
    )


@pytest.fixture
def request_payload() -> dict[str, object]:
    """The same request as JSON, exactly as the backend would send it."""
    return {
        "symbol": "RELIANCE",
        "exchange": "NSE",
        "interval": "ONE_DAY",
        "as_of": "2026-08-21T10:00:00Z",
        "trading_date": "2026-08-21",
        "price": "1316.0000",
        "technical": {
            "rsi_14": "54.20",
            "ema_20": "1308.40",
            "ema_50": "1297.80",
            "macd": {"line": "4.21", "signal": "3.10", "histogram": "1.11"},
            "atr_14": "18.40",
        },
    }
