"""AnalysisRequest validation.

The request is the service's entire view of the world, so what it refuses to
accept matters as much as what it accepts.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from app.domain.analysis import MAX_OBSERVATIONS, MAX_RECENT_CLOSES, AnalysisRequest
from app.domain.enums import AnalysisDepth, CandleInterval, Exchange

BASE: dict[str, object] = {
    "symbol": "RELIANCE",
    "exchange": "NSE",
    "interval": "ONE_DAY",
    "as_of": "2026-08-21T10:00:00Z",
    "trading_date": "2026-08-21",
    "price": "1316.0000",
}


def _request(**overrides: object) -> AnalysisRequest:
    return AnalysisRequest.model_validate({**BASE, **overrides})


class TestHappyPath:
    def test_minimal_request_validates(self) -> None:
        request = _request()
        assert request.symbol == "RELIANCE"
        assert request.exchange is Exchange.NSE
        assert request.interval is CandleInterval.ONE_DAY
        assert request.trading_date == date(2026, 8, 21)
        assert request.depth is AnalysisDepth.ROUTINE

    def test_price_keeps_full_precision(self) -> None:
        # A JSON string in, an exact Decimal out - no float ever touches it.
        assert _request(price="1316.0004").price == Decimal("1316.0004")

    def test_symbol_is_normalised(self) -> None:
        assert _request(symbol="  reliance  ").symbol == "RELIANCE"

    def test_timestamp_is_normalised_to_utc(self) -> None:
        request = _request(as_of="2026-08-21T15:30:00+05:30")
        assert request.as_of == datetime(2026, 8, 21, 10, 0, tzinfo=UTC)

    def test_indicators_are_optional_and_default_to_empty(self) -> None:
        assert _request().technical.present() == {}


class TestRejections:
    def test_unknown_field_is_rejected(self) -> None:
        # Contract drift should be loud. A silently ignored field would be a
        # decision made without an input the caller believed it had sent.
        with pytest.raises(ValidationError, match="Extra inputs"):
            _request(symbol_token="2885")

    def test_naive_timestamp_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            _request(as_of="2026-08-21T10:00:00")

    @pytest.mark.parametrize("price", ["0", "-10.5"])
    def test_non_positive_price_is_rejected(self, price: str) -> None:
        with pytest.raises(ValidationError):
            _request(price=price)

    def test_unknown_exchange_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _request(exchange="NASDAQ")

    def test_unknown_interval_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _request(interval="ONE_WEEK")

    def test_blank_symbol_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _request(symbol="   ")

    def test_symbol_with_unsupported_characters_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="unsupported characters"):
            _request(symbol="RELIANCE;DROP")

    @pytest.mark.parametrize("rsi", ["-1", "101"])
    def test_rsi_outside_its_range_is_rejected(self, rsi: str) -> None:
        with pytest.raises(ValidationError, match="rsi_14"):
            _request(technical={"rsi_14": rsi})

    def test_unknown_indicator_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs"):
            _request(technical={"stochastic_k": "20"})


class TestBoundedInput:
    """There is no candle list, and what history there is has a ceiling."""

    def test_no_candle_field_exists(self) -> None:
        assert "candles" not in AnalysisRequest.model_fields

    def test_recent_closes_are_capped(self) -> None:
        with pytest.raises(ValidationError):
            _request(recent_closes=["100"] * (MAX_RECENT_CLOSES + 1))

    def test_observations_are_capped(self) -> None:
        observation = {
            "observed_at": "2026-08-20T10:00:00Z",
            "kind": "volume",
            "summary": "spike",
        }
        with pytest.raises(ValidationError):
            _request(observations=[observation] * (MAX_OBSERVATIONS + 1))

    def test_observation_summary_is_capped(self) -> None:
        with pytest.raises(ValidationError):
            _request(
                observations=[
                    {
                        "observed_at": "2026-08-20T10:00:00Z",
                        "kind": "volume",
                        "summary": "x" * 500,
                    }
                ]
            )

    def test_non_positive_close_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="positive"):
            _request(recent_closes=["100", "0"])


class TestDepth:
    def test_depth_is_a_closed_set(self) -> None:
        assert _request(depth="DEEP").depth is AnalysisDepth.DEEP
        with pytest.raises(ValidationError):
            _request(depth="EXHAUSTIVE")
