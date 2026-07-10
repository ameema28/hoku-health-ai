"""
Hoku Health Care - Global Error Handlers.

Centralized exception handling to ensure all API errors return
structured JSON responses instead of raw stack traces.
"""

import logging
import traceback

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


def add_error_handlers(app: FastAPI) -> None:
    """
    Register global exception handlers on the FastAPI application.

    Args:
        app: FastAPI application instance.
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """
        Handle FastAPI/Starlette HTTP exceptions.

        Args:
            request: Incoming request object.
            exc: HTTPException instance.

        Returns:
            JSONResponse: Structured error payload.
        """
        logger.warning("HTTP %s: %s | Path: %s", exc.status_code, exc.detail, request.url.path)
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "status_code": exc.status_code},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Handle unexpected exceptions to prevent stack trace leakage.

        Logs full traceback server-side but returns a generic message
        to the client for security.

        Args:
            request: Incoming request object.
            exc: Generic Exception instance.

        Returns:
            JSONResponse: Generic error payload with 500 status.
        """
        logger.error("Unhandled exception at %s: %s\n%s", request.url.path, exc, traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An internal server error occurred. Please contact support.",
                "status_code": 500,
            },
        )
