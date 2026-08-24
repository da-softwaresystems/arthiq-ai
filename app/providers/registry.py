"""Selection and lifecycle of the active AI provider.

This is the only module that imports a vendor adapter. Everything else asks for
"a provider" and receives an :class:`~app.providers.base.AIProvider`, which is
what keeps the promise that adding Anthropic or OpenAI later means adding one
file here and touching nothing in the domain or analysis layers.

Selection fails closed: an unknown ``AI_PROVIDER``, a missing key or an
unconfigured model raises rather than returning something that quietly answers
with a stub.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from app.core.config import Settings, get_settings
from app.domain.enums import AnalysisDepth
from app.providers.base import AIProvider
from app.providers.exceptions import UnsupportedProviderError
from app.providers.fake import FakeAIProvider
from app.providers.gemini import GeminiProvider
from app.providers.ollama import OllamaProvider

logger = logging.getLogger(__name__)

#: Registering a provider here is what makes ``AI_PROVIDER`` accept its name.
PROVIDERS: dict[str, Callable[[Settings], AIProvider]] = {
    "fake": lambda _settings: FakeAIProvider(),
    "ollama": OllamaProvider,
    "gemini": GeminiProvider,
}

_provider: AIProvider | None = None


def build_provider(settings: Settings) -> AIProvider:
    """Construct the configured provider without caching it."""
    name = settings.ai_provider.lower()
    factory = PROVIDERS.get(name)
    if factory is None:
        raise UnsupportedProviderError(f"Unknown AI provider: {name}", provider=name)
    return factory(settings)


def get_ai_provider() -> AIProvider:
    """Process-wide provider, created on first use."""
    global _provider
    if _provider is None:
        settings = get_settings()
        _provider = build_provider(settings)
        logger.info(
            "AI provider ready",
            extra={
                "provider": _provider.name,
                "model": _provider.model_for(AnalysisDepth.ROUTINE),
            },
        )
    return _provider


def set_ai_provider(provider: AIProvider | None) -> None:
    """Install a provider explicitly. Used by tests and by the verify scripts."""
    global _provider
    _provider = provider


async def close_ai_provider() -> None:
    """Release provider resources on shutdown."""
    global _provider
    if _provider is not None:
        await _provider.aclose()
        logger.info("AI provider closed", extra={"provider": _provider.name})
    _provider = None
