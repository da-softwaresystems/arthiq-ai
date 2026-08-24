"""Manual Ollama verification.

A one-shot check that the local provider is configured, reachable, serving the
configured model, and returning output that validates into a TradingDecision::

    OLLAMA_MODEL=<model> python scripts/verify_ollama.py

Exits 0 on success and 1 on failure, so it can be used as a pre-flight check.
It calls the model once - the same single call the API would make - and prints
the decision, never the prompt or the raw response.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, date, datetime
from decimal import Decimal

from app.analysis.service import AnalysisService
from app.core.config import Settings
from app.core.exceptions import AppError
from app.core.logging import configure_logging
from app.domain.analysis import AnalysisRequest, MacdValues, TechnicalIndicators
from app.domain.enums import CandleInterval, Exchange
from app.providers.ollama import OllamaProvider

SAMPLE = AnalysisRequest(
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
    ),
)


async def main() -> int:
    configure_logging("INFO")
    settings = Settings()

    if not settings.ollama_model:
        print("OLLAMA_MODEL is not set. Configure a pulled model and try again.")
        return 1

    provider = OllamaProvider(settings)
    try:
        readiness = await provider.check_readiness()
        print(f"provider : {readiness.provider}")
        print(f"model    : {readiness.model}")
        print(f"base_url : {settings.ollama_base_url}")
        print(f"ready    : {readiness.ready} ({readiness.detail})")
        if not readiness.ready:
            return 1

        print("\nRunning one analysis...")
        decision = await AnalysisService(provider, settings).analyze(
            SAMPLE, request_id="verify-ollama"
        )
    except AppError as exc:
        print(f"\nFAILED: [{exc.code}] {exc.message}")
        return 1
    finally:
        await provider.aclose()

    print(f"\ndecision   : {decision.decision.value}")
    print(f"confidence : {decision.confidence} (self-reported, not a probability)")
    print(f"risk_level : {decision.risk_level.value}")
    print(f"reasoning  : {decision.reasoning}")
    print(f"invalidated by: {decision.invalidating_conditions}")
    print(
        f"metadata   : provider={decision.metadata.provider} "
        f"model={decision.metadata.model} "
        f"prompt={decision.metadata.prompt_version} "
        f"latency_ms={decision.metadata.latency_ms}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
