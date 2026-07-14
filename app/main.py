"""
Hoku Health Care - FastAPI Application Entry Point.

Registers routers, middleware, and event handlers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import ai
from app.core.config import settings
from app.core.middleware import TimingMiddleware

app = FastAPI(
    title="Hoku Health Care API",
    description="AI-powered healthcare assistance platform",
    version="1.0.0",
)

# ------------------------------------------------------------------
# Middleware (order matters: Timing first to catch all requests)
# ------------------------------------------------------------------
app.add_middleware(TimingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not settings.is_production else ["https://hokuhealth.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Routers
# ------------------------------------------------------------------
# TEMPORARY: Bypass auth for Day 2 local testing only

app.include_router(ai.router)


@app.get("/")
async def root():
    """Root health check."""
    return {"message": "Hoku Health Care API is running"}