"""Application errors and the handlers that render them.

Every error leaves this API in one shape::

    {"error": {"code": "PROVIDER_TIMEOUT", "message": "The AI provider did not respond in time"}}

Stack traces, provider payloads, credentials and prompt text never reach the
client. The code is stable enough for the backend to branch on; the message is
for a human reading a log.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

# Spelled out rather than taken from ``status``: Starlette renamed the
# constant, and the number is what actually matters.
HTTP_422_UNPROCESSABLE = 422


class AppError(Exception):
    """Base class for errors that are safe to show to a client."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "INTERNAL_ERROR"
    message: str = "An unexpected error occurred"

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.status_code = status_code or self.status_code
        self.details = details
        self.headers = headers
        super().__init__(self.message)


class AuthenticationError(AppError):
    """No service credential, or one that could not be verified."""

    status_code = status.HTTP_401_UNAUTHORIZED
    code = "UNAUTHORIZED"
    message = "Service authentication required"


class ValidationError(AppError):
    status_code = HTTP_422_UNPROCESSABLE
    code = "VALIDATION_ERROR"
    message = "Request validation failed"


class ServiceUnavailableError(AppError):
    """A dependency the request needs is unavailable or not configured."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "SERVICE_UNAVAILABLE"
    message = "Service temporarily unavailable"


class ConfigurationError(AppError):
    """Raised when the process is not safe to serve this request."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "CONFIGURATION_ERROR"
    message = "Service is misconfigured"


#: HTTP status -> stable error code, for errors raised outside :class:`AppError`.
_STATUS_CODES: dict[int, str] = {
    status.HTTP_400_BAD_REQUEST: "BAD_REQUEST",
    status.HTTP_401_UNAUTHORIZED: "UNAUTHORIZED",
    status.HTTP_403_FORBIDDEN: "FORBIDDEN",
    status.HTTP_404_NOT_FOUND: "NOT_FOUND",
    status.HTTP_405_METHOD_NOT_ALLOWED: "METHOD_NOT_ALLOWED",
    HTTP_422_UNPROCESSABLE: "VALIDATION_ERROR",
    status.HTTP_429_TOO_MANY_REQUESTS: "RATE_LIMITED",
    status.HTTP_503_SERVICE_UNAVAILABLE: "SERVICE_UNAVAILABLE",
    status.HTTP_504_GATEWAY_TIMEOUT: "TIMEOUT",
}


def error_body(
    code: str, message: str, details: dict[str, Any] | list[Any] | None = None
) -> dict[str, Any]:
    """Build the canonical error envelope."""
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    return {"error": error}


def register_exception_handlers(app: FastAPI) -> None:
    """Attach the handlers that guarantee a single error shape."""

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(exc.code, exc.message, exc.details),
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {
                "field": ".".join(str(part) for part in err.get("loc", ())),
                "message": err.get("msg", "invalid value"),
            }
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=HTTP_422_UNPROCESSABLE,
            content=error_body("VALIDATION_ERROR", "Request validation failed", details),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, "HTTP_ERROR")
        message = exc.detail if isinstance(exc.detail, str) else "Request failed"
        return JSONResponse(
            status_code=exc.status_code,
            content=error_body(code, message),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Full detail goes to the logs; the client only learns that it failed.
        logger.exception("Unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_body("INTERNAL_ERROR", "An unexpected error occurred"),
        )
