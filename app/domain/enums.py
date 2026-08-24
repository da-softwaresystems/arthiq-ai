"""The service's own vocabulary.

Deliberately not a vendor's wire format and deliberately closed: a model that
answers ``"STRONG BUY"`` produces a validation failure, not a fourth kind of
decision that the backend has never seen.
"""

from __future__ import annotations

from enum import StrEnum


class Exchange(StrEnum):
    """Exchanges supported in V1 - Indian equity cash market only."""

    NSE = "NSE"
    BSE = "BSE"


class CandleInterval(StrEnum):
    """Timeframes the backend can ask about.

    The member names match the backend's ``CandleInterval`` so the contract
    needs no translation table on either side.
    """

    FIVE_MINUTE = "FIVE_MINUTE"
    FIFTEEN_MINUTE = "FIFTEEN_MINUTE"
    ONE_HOUR = "ONE_HOUR"
    ONE_DAY = "ONE_DAY"


class Decision(StrEnum):
    """The only three answers this service can give.

    There is no ``EXECUTE``. A decision is an analytical opinion; acting on it
    is the backend's responsibility and the backend's alone.
    """

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class RiskLevel(StrEnum):
    """How exposed the position implied by the decision would be."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class MarketTrend(StrEnum):
    """Broad regime of the wider market, as assessed by the caller."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"


class AnalysisDepth(StrEnum):
    """How much the caller is willing to spend on this answer.

    ``ROUTINE`` is the default and maps to the cheaper configured model;
    ``DEEP`` maps to the stronger one. The mapping lives in provider
    configuration - no business rule anywhere names a model.
    """

    ROUTINE = "ROUTINE"
    DEEP = "DEEP"
