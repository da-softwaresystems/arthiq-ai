"""FastAPI application factory.

Arthiq AI is a single small service with one job: turn a structured market
snapshot into a validated trading opinion. It is not a monolith and has no
internal services - the layering is context -> prompt -> provider -> output
validation, and that is the whole of it.

What this process cannot do, by construction rather than by policy: reach a
broker, read the backend's database, authenticate a user, or place an order.
None of those clients are imported anywhere in this package.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.analysis.errors import translate_provider_error
from app.analysis.router import router as analysis_router
from app.core.config import Settings, get_settings
from app.core.exceptions import error_body, register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware
from app.health.router import router as health_router
from app.providers.exceptions import AIProviderError
from app.providers.registry import close_ai_provider

logger = logging.getLogger(__name__)

DESCRIPTION = """
Internal AI service for the Arthiq platform.

* **Internal only.** Every `/internal/v1` route requires the shared service key
  in `X-API-Key`. There is no user authentication here: the backend
  authenticates the end user and never forwards that identity.
* **Advisory only.** The service returns a `TradingDecision`, which is an
  analytical opinion. It cannot place an order, hold a position or reach a
  broker, and the backend is free to ignore any decision it receives.
* **Provider-neutral.** Ollama, Gemini and the deterministic test provider sit
  behind one interface. Responses never contain vendor payloads, and every
  decision records which provider, model and prompt version produced it.
* **Confidence is not a probability.** It is the model's self-reported
  conviction on a 0-1 scale, uncalibrated and unvalidated against outcomes.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start and stop process-wide resources."""
    settings: Settings = app.state.settings
    configure_logging(settings.log_level, json_output=settings.json_logs)
    logger.info(
        "Starting %s (%s) with provider %s",
        settings.app_name,
        settings.app_env,
        settings.ai_provider,
    )
    if not settings.service_auth_configured:
        # Loud, but not fatal: the process still starts and stays healthy, and
        # every internal call is refused until a key is configured.
        logger.error("AI_SERVICE_API_KEY is not set - internal endpoints will refuse all calls")

    yield

    await close_ai_provider()
    logger.info("Shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the ASGI application."""
    settings = settings or get_settings()

    app = FastAPI(
        title="Arthiq AI Service",
        description=DESCRIPTION,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs" if settings.enable_docs else None,
        redoc_url="/redoc" if settings.enable_docs else None,
        openapi_url="/openapi.json" if settings.enable_docs else None,
    )
    app.state.settings = settings
    # The settings this app was built with are the settings its routes see,
    # even when a caller passes an explicit object instead of the cached one.
    app.dependency_overrides[get_settings] = lambda: settings

    app.add_middleware(RequestContextMiddleware)
    if settings.cors_origins:
        # Normally empty: the only intended caller is another server, and a
        # browser has no business here.
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["POST", "GET"],
            allow_headers=["*"],
        )

    register_exception_handlers(app)

    @app.exception_handler(AIProviderError)
    async def _provider_error(_: Request, exc: AIProviderError) -> JSONResponse:
        """Catch provider failures raised outside the analysis service.

        Provider construction happens in a dependency, so a missing key or an
        unknown ``AI_PROVIDER`` surfaces here rather than inside the service.
        """
        error = translate_provider_error(exc)
        logger.error(
            "Provider error outside the analysis path",
            extra={"provider": exc.provider, "error_code": exc.code},
        )
        return JSONResponse(
            status_code=error.status_code,
            content=error_body(error.code, error.message, error.details),
            headers=error.headers,
        )

    app.include_router(health_router)
    app.include_router(analysis_router, prefix=settings.internal_api_prefix)

    return app


app = create_app()
