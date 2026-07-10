"""
Hoku Health Care - FastAPI Application Factory.

Entry point for the Hoku Health Care backend. Assembles routers,
middleware, and event handlers into a single ASGI application.
"""

import logging

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.v1.endpoints.ai import router as ai_router
from app.core.config import configure_logging, settings
from app.middleware.cors import add_cors_middleware
from app.middleware.error_handler import add_error_handlers

logger = logging.getLogger(__name__)


def create_application() -> FastAPI:
    """
    Factory function to create and configure the FastAPI app.

    Returns:
        FastAPI: Configured application instance.
    """
    configure_logging()

    app = FastAPI(
        title="Hoku Health Care",
        description="AI-powered home healthcare platform backend.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Register middleware
    add_cors_middleware(app)
    add_error_handlers(app)

    # Register API routers
    app.include_router(ai_router)

    # Root redirect to API documentation
    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        """Redirect root to Swagger UI."""
        return RedirectResponse(url="/docs")

    # Startup event
    @app.on_event("startup")
    async def startup_event() -> None:
        """Log service startup for observability."""
        logger.info("Hoku AI service starting | environment=%s", settings.ENVIRONMENT)

    return app


app = create_application()
