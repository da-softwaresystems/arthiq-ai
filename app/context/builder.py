"""The context builder.

One job: take a validated :class:`AnalysisRequest` and produce the exact text a
model will see - bounded, ordered and deterministic.

*Bounded* because prompt size is the main cost driver and the only one this
service fully controls. The request already caps what may arrive; the builder
caps it again, from configuration, and records what it dropped.

*Deterministic* because the same request must produce the same context, byte for
byte. Nothing here reads the clock, the environment or a random source, which is
what makes this layer testable on its own and what makes a decision
reproducible for a given prompt version and model.

What never enters a context: candle histories, database rows, user identities,
credentials, or anything not present on the request object.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal

from pydantic import BaseModel, ConfigDict

from app.core.config import Settings
from app.domain.analysis import AnalysisRequest, Observation
from app.domain.enums import CandleInterval, Exchange

#: Two decimals is enough for a percentage in a prompt, and fixing the scale
#: keeps the rendered text stable.
_PERCENT_SCALE = Decimal("0.01")


@dataclass(frozen=True, slots=True)
class ContextLimits:
    """Ceilings applied while building. Configuration, not magic numbers."""

    max_recent_closes: int = 20
    max_observations: int = 10

    @classmethod
    def from_settings(cls, settings: Settings) -> ContextLimits:
        return cls(
            max_recent_closes=settings.context_max_recent_closes,
            max_observations=settings.context_max_observations,
        )


class AnalysisContext(BaseModel):
    """The prompt-ready view of a request.

    Frozen: once built, the context that was sent to the model is exactly the
    context that can be logged or replayed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    exchange: Exchange
    interval: CandleInterval
    as_of: datetime
    trading_date: date
    price: Decimal

    indicators: dict[str, Decimal]
    market_lines: list[str]
    recent_closes: list[Decimal]
    recent_change_percent: Decimal | None
    observations: list[Observation]

    closes_omitted: int
    observations_omitted: int

    def render(self) -> str:
        """The deterministic text block handed to the prompt template."""
        lines: list[str] = [
            "INSTRUMENT",
            f"  symbol: {self.symbol}",
            f"  exchange: {self.exchange.value}",
            f"  interval: {self.interval.value}",
            f"  as_of_utc: {self.as_of.isoformat()}",
            f"  trading_date: {self.trading_date.isoformat()}",
            f"  price: {self.price}",
            "",
            "TECHNICAL INDICATORS",
        ]
        if self.indicators:
            lines += [f"  {name}: {value}" for name, value in self.indicators.items()]
        else:
            lines.append("  (none available)")

        lines += ["", "MARKET CONTEXT"]
        lines += [f"  {line}" for line in self.market_lines] or ["  (none provided)"]

        lines += ["", "RECENT CLOSES (oldest first)"]
        if self.recent_closes:
            lines.append("  " + ", ".join(str(close) for close in self.recent_closes))
            if self.recent_change_percent is not None:
                lines.append(f"  change_over_window_percent: {self.recent_change_percent}")
            if self.closes_omitted:
                lines.append(f"  (older {self.closes_omitted} closes omitted)")
        else:
            lines.append("  (none provided)")

        lines += ["", "OBSERVATIONS"]
        if self.observations:
            lines += [
                f"  [{obs.observed_at.isoformat()}] {obs.kind}: {obs.summary}"
                for obs in self.observations
            ]
            if self.observations_omitted:
                lines.append(f"  (older {self.observations_omitted} observations omitted)")
        else:
            lines.append("  (none provided)")

        return "\n".join(lines)


def _market_lines(request: AnalysisRequest) -> list[str]:
    market = request.market
    if market is None:
        return []
    lines: list[str] = []
    if market.trend is not None:
        lines.append(f"trend: {market.trend.value}")
    if market.index_symbol:
        lines.append(f"index: {market.index_symbol}")
    if market.index_change_percent is not None:
        lines.append(f"index_change_percent: {market.index_change_percent}")
    if market.sector:
        lines.append(f"sector: {market.sector}")
    if market.notes:
        lines.append(f"notes: {market.notes}")
    return lines


def _change_percent(closes: list[Decimal]) -> Decimal | None:
    """Percentage move across the retained window, or ``None`` if undefined."""
    if len(closes) < 2 or closes[0] <= 0:
        return None
    change = (closes[-1] - closes[0]) / closes[0] * Decimal(100)
    return change.quantize(_PERCENT_SCALE, rounding=ROUND_HALF_UP)


def build_context(request: AnalysisRequest, limits: ContextLimits | None = None) -> AnalysisContext:
    """Build the bounded context for ``request``.

    Trimming always keeps the *most recent* entries: the newest closes and the
    newest observations are the ones worth the tokens.
    """
    limits = limits or ContextLimits()

    closes = list(request.recent_closes)
    kept_closes = closes[-limits.max_recent_closes :] if limits.max_recent_closes else []

    ordered = sorted(request.observations, key=lambda obs: (obs.observed_at, obs.kind, obs.summary))
    kept_observations = ordered[-limits.max_observations :] if limits.max_observations else []

    return AnalysisContext(
        symbol=request.symbol,
        exchange=request.exchange,
        interval=request.interval,
        as_of=request.as_of,
        trading_date=request.trading_date,
        price=request.price,
        indicators=request.technical.present(),
        market_lines=_market_lines(request),
        recent_closes=kept_closes,
        recent_change_percent=_change_percent(kept_closes),
        observations=kept_observations,
        closes_omitted=len(closes) - len(kept_closes),
        observations_omitted=len(ordered) - len(kept_observations),
    )
