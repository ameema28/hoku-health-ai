"""
Hoku Health Care - CORS Middleware Configuration.

Cross-Origin Resource Sharing setup for the FastAPI application.
Allows the Vercel-hosted frontend to communicate with the backend.
"""

from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings


def add_cors_middleware(app) -> None:
    """
    Configure and attach CORS middleware to the FastAPI application.

    In production, origins should be restricted to the deployed frontend URL.
    Development allows localhost for local testing.

    Args:
        app: FastAPI application instance.
    """
    # Define allowed origins based on environment
    origins = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://localhost:3000",
    ]

    if settings.is_production:
        # Production frontend URLs (Vercel deployment)
        origins = [
            "https://hoku-health.vercel.app",
            "https://www.hoku-health.vercel.app",
        ]
    else:
        # Allow any localhost port in development for flexibility
        origins.append("http://localhost:8000")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
