"""Context builder: bounded, deterministic, and independently testable."""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from app.context.builder import ContextLimits, build_context
from app.core.config import Settings
from app.domain.analysis import AnalysisRequest, MacdValues, Observation, TechnicalIndicators
from app.domain.enums import CandleInterval, Exchange

LIMITS = ContextLimits(max_recent_closes=5, max_observations=3)


def _request(**overrides: object) -> AnalysisRequest:
    base: dict[str, object] = {
        "symbol": "RELIANCE",
        "exchange": Exchange.NSE,
        "interval": CandleInterval.ONE_DAY,
        "as_of": datetime(2026, 8, 21, 10, 0, tzinfo=UTC),
        "trading_date": date(2026, 8, 21),
        "price": Decimal("1316.0000"),
    }
    return AnalysisRequest.model_validate({**base, **overrides})


def _observation(day: int, kind: str = "volume") -> Observation:
    return Observation(
        observed_at=datetime(2026, 8, day, 10, 0, tzinfo=UTC),
        kind=kind,
        summary=f"note for day {day}",
    )


class TestBounds:
    def test_recent_closes_are_trimmed_to_the_newest(self) -> None:
        closes = [Decimal(value) for value in range(100, 112)]
        context = build_context(_request(recent_closes=closes), LIMITS)
        assert context.recent_closes == closes[-5:]
        assert context.closes_omitted == 7

    def test_observations_are_trimmed_to_the_newest(self) -> None:
        observations = [_observation(day) for day in range(10, 20)]
        context = build_context(_request(observations=observations), LIMITS)
        assert len(context.observations) == 3
        assert [obs.observed_at.day for obs in context.observations] == [17, 18, 19]
        assert context.observations_omitted == 7

    def test_zero_limits_drop_history_entirely(self) -> None:
        context = build_context(
            _request(recent_closes=[Decimal("1"), Decimal("2")], observations=[_observation(1)]),
            ContextLimits(max_recent_closes=0, max_observations=0),
        )
        assert context.recent_closes == []
        assert context.observations == []
        assert context.closes_omitted == 2

    def test_rendered_context_stays_small(self) -> None:
        context = build_context(
            _request(
                recent_closes=[Decimal(value) for value in range(100, 300)],
                observations=[_observation(day) for day in range(1, 29)],
            ),
            LIMITS,
        )
        # The ceiling is what matters, not the exact number: a caller cannot
        # grow the prompt by sending more history.
        assert len(context.render()) < 2000

    def test_limits_come_from_settings(self) -> None:
        settings = Settings(_env_file=None, context_max_recent_closes=7, context_max_observations=2)
        limits = ContextLimits.from_settings(settings)
        assert limits == ContextLimits(max_recent_closes=7, max_observations=2)


class TestDeterminism:
    def test_same_request_renders_identically(self) -> None:
        request = _request(recent_closes=[Decimal("100"), Decimal("101")])
        assert build_context(request, LIMITS).render() == build_context(request, LIMITS).render()

    def test_observation_order_does_not_depend_on_input_order(self) -> None:
        observations = [_observation(day) for day in (12, 10, 11)]
        shuffled = [_observation(day) for day in (11, 12, 10)]
        first = build_context(_request(observations=observations), LIMITS)
        second = build_context(_request(observations=shuffled), LIMITS)
        assert first.render() == second.render()

    def test_indicator_order_is_stable(self) -> None:
        technical = TechnicalIndicators(
            atr_14=Decimal("18.4"), rsi_14=Decimal("54.2"), ema_20=Decimal("1308.4")
        )
        rendered = build_context(_request(technical=technical), LIMITS).render()
        assert rendered.index("rsi_14") < rendered.index("ema_20") < rendered.index("atr_14")


class TestContent:
    def test_absent_indicators_are_omitted_not_zeroed(self) -> None:
        rendered = build_context(_request(technical=TechnicalIndicators()), LIMITS).render()
        assert "(none available)" in rendered
        assert "rsi_14" not in rendered

    def test_macd_components_are_flattened(self) -> None:
        technical = TechnicalIndicators(macd=MacdValues(line=Decimal("4.21"), signal=None))
        indicators = build_context(_request(technical=technical), LIMITS).indicators
        assert indicators == {"macd_line": Decimal("4.21")}

    def test_window_change_is_derived_from_the_retained_closes(self) -> None:
        context = build_context(
            _request(recent_closes=[Decimal("100.00"), Decimal("110.00")]), LIMITS
        )
        assert context.recent_change_percent == Decimal("10.00")

    def test_change_is_undefined_for_a_single_close(self) -> None:
        context = build_context(_request(recent_closes=[Decimal("100.00")]), LIMITS)
        assert context.recent_change_percent is None

    def test_identifying_facts_reach_the_prompt(self) -> None:
        rendered = build_context(_request(), LIMITS).render()
        assert "symbol: RELIANCE" in rendered
        assert "exchange: NSE" in rendered
        assert "interval: ONE_DAY" in rendered
        assert "trading_date: 2026-08-21" in rendered
        assert "price: 1316.0000" in rendered

    def test_context_is_frozen(self) -> None:
        context = build_context(_request(), LIMITS)
        assert context.model_config["frozen"] is True
