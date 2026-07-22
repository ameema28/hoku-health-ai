"""
Hoku Health Care - FastAPI Application Entry Point.

Registers routers, middleware, and event handlers.
"""

import asyncio
import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.endpoints import ai
from app.core.config import settings
from app.core.middleware import TimingMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup/shutdown events.

    Day 8: Eagerly initializes LLM clients and chains on startup to
    eliminate cold-start latency on the first patient request.
    """
    # ------------------------------------------------------------------
    # Startup: Warm up LLM clients
    # ------------------------------------------------------------------
    logger.info("Hoku Health Care API starting up...")
    
    # Import here to avoid circular dependencies at module level
    try:
        from app.ai.chatbot import HokuChatbot
        
        chatbot = HokuChatbot()
        # Run warm-up in a background thread so it doesn't block
        # the event loop startup (ChatGroq init is synchronous I/O)
        await asyncio.to_thread(chatbot.warm_up)
        logger.info("LLM warm-up completed successfully")
    except Exception as exc:
        # Log but don't fail startup — the chatbot will retry lazily
        logger.warning("LLM warm-up failed (will retry on first request): %s", exc)
    
    yield
    
    # ------------------------------------------------------------------
    # Shutdown: Cleanup
    # ------------------------------------------------------------------
    logger.info("Hoku Health Care API shutting down...")


app = FastAPI(
    title="Hoku Health Care API",
    description="AI-powered healthcare assistance platform",
    version="1.0.0",
    lifespan=lifespan,
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