"""What the backend sends: a bounded, structured snapshot to reason about.

Design rules this file enforces:

* **No vendor fields.** Nothing here mentions Angel One, a symbol token, or how
  a candle is stored. The backend translates its own storage into this shape.
* **No raw history.** There is no candle list. A short, capped series of recent
  closes is the most history this contract can carry, and the context builder
  trims even that.
* **Strict.** Every model forbids unknown fields. An unrecognised key is a
  contract drift, and a 422 is a cheaper way to discover it than a silently
  ignored indicator.
* **Exact numbers.** Prices and indicators are ``Decimal``. The backend
  serialises them as JSON strings (``"1316.0000"``); a JSON float would round a
  price on the way in.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from app.domain.enums import AnalysisDepth, CandleInterval, Exchange, MarketTrend

#: Hard ceilings on the request itself. The context builder applies its own,
#: tighter, configurable limits; these stop an oversized *payload* before it is
#: ever parsed into a context.
MAX_RECENT_CLOSES = 200
MAX_OBSERVATIONS = 50
MAX_NOTE_CHARS = 500

_STRICT = ConfigDict(extra="forbid", frozen=True)


def _require_utc(value: datetime) -> datetime:
    """Reject a naive timestamp; normalise everything else to UTC."""
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class MacdValues(BaseModel):
    """MACD line, signal and histogram, as computed by the backend."""

    model_config = _STRICT

    line: Decimal | None = None
    signal: Decimal | None = None
    histogram: Decimal | None = None


class BollingerValues(BaseModel):
    """Bollinger Bands: middle SMA with the two bands around it."""

    model_config = _STRICT

    upper: Decimal | None = None
    middle: Decimal | None = None
    lower: Decimal | None = None


class TechnicalIndicators(BaseModel):
    """The indicator set for the bar being analysed.

    Every field is optional and ``None`` means *not enough history yet* - the
    same meaning the backend gives it. It never means zero, and the context
    builder omits it rather than inventing a value for the prompt.
    """

    model_config = _STRICT

    rsi_14: Decimal | None = None
    sma_20: Decimal | None = None
    sma_50: Decimal | None = None
    sma_200: Decimal | None = None
    ema_20: Decimal | None = None
    ema_50: Decimal | None = None
    macd: MacdValues | None = None
    atr_14: Decimal | None = None
    bollinger: BollingerValues | None = None
    volume_sma_20: Decimal | None = None

    @field_validator("rsi_14")
    @classmethod
    def _rsi_range(cls, value: Decimal | None) -> Decimal | None:
        if value is not None and not (Decimal(0) <= value <= Decimal(100)):
            raise ValueError("rsi_14 must be between 0 and 100")
        return value

    def present(self) -> dict[str, Decimal]:
        """Flattened scalar indicators that actually have a value.

        Insertion order follows the field order above, which makes the rendered
        context deterministic for a given input.
        """
        values: dict[str, Decimal] = {}
        for name in ("rsi_14", "sma_20", "sma_50", "sma_200", "ema_20", "ema_50", "atr_14"):
            value = getattr(self, name)
            if value is not None:
                values[name] = value
        if self.macd is not None:
            for suffix in ("line", "signal", "histogram"):
                value = getattr(self.macd, suffix)
                if value is not None:
                    values[f"macd_{suffix}"] = value
        if self.bollinger is not None:
            for suffix in ("upper", "middle", "lower"):
                value = getattr(self.bollinger, suffix)
                if value is not None:
                    values[f"bollinger_{suffix}"] = value
        if self.volume_sma_20 is not None:
            values["volume_sma_20"] = self.volume_sma_20
        return values


class MarketContext(BaseModel):
    """Conditions around the instrument, summarised by the backend."""

    model_config = _STRICT

    trend: MarketTrend | None = None
    index_symbol: str | None = Field(default=None, max_length=32)
    index_change_percent: Decimal | None = None
    sector: str | None = Field(default=None, max_length=64)
    notes: str | None = Field(default=None, max_length=MAX_NOTE_CHARS)


class Observation(BaseModel):
    """One structured note the backend wants the model to take into account.

    Free text is capped and the list is capped. This is the only place caller
    prose enters a prompt, so it is the only place that needs a ceiling.
    """

    model_config = _STRICT

    observed_at: datetime
    kind: str = Field(min_length=1, max_length=40)
    summary: str = Field(min_length=1, max_length=240)

    @field_validator("observed_at")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return _require_utc(value)


class AnalysisRequest(BaseModel):
    """A complete, self-contained request for one analytical opinion.

    Self-contained is the point: this service performs no lookups. If a fact is
    not in this object, it does not inform the decision.
    """

    model_config = _STRICT

    symbol: str = Field(min_length=1, max_length=32)
    exchange: Exchange
    interval: CandleInterval
    as_of: datetime = Field(description="UTC instant of the bar being analysed")
    trading_date: date = Field(description="The exchange session the bar belongs to")
    price: Decimal = Field(gt=0, description="Close (or last traded price) at as_of")

    technical: TechnicalIndicators = Field(default_factory=TechnicalIndicators)
    market: MarketContext | None = None
    recent_closes: list[Decimal] = Field(
        default_factory=list,
        max_length=MAX_RECENT_CLOSES,
        description="Oldest first. A short series, never a candle history.",
    )
    observations: list[Observation] = Field(default_factory=list, max_length=MAX_OBSERVATIONS)

    depth: AnalysisDepth = AnalysisDepth.ROUTINE
    prompt_version: str | None = Field(
        default=None,
        max_length=64,
        description="Pin a specific prompt version; the service default is used when omitted.",
    )

    @field_validator("symbol")
    @classmethod
    def _normalise_symbol(cls, value: str) -> str:
        symbol = value.strip().upper()
        if not symbol:
            raise ValueError("symbol must not be blank")
        if not all(char.isalnum() or char in "-.&_" for char in symbol):
            raise ValueError("symbol contains unsupported characters")
        return symbol

    @field_validator("as_of")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return _require_utc(value)

    @field_validator("recent_closes")
    @classmethod
    def _positive_closes(cls, values: list[Decimal]) -> list[Decimal]:
        if any(value <= 0 for value in values):
            raise ValueError("recent_closes must all be positive")
        return values

    @field_serializer("price", "recent_closes")
    def _decimals_as_strings(self, value: object) -> object:
        """Mirror the backend's convention: a number crosses the wire as text."""
        if isinstance(value, list):
            return [str(item) for item in value]
        return str(value)
