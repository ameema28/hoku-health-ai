"""
Hoku Health Care - Authentication endpoints.

- ``POST /api/auth/login``: JSON ``{email, password}`` -> bearer token. Every
  failure (unknown email, wrong password, inactive user) returns the same
  401, and unknown emails still pay for one bcrypt check.
- ``POST /api/auth/demo``: only when ``DEMO_LOGIN_ENABLED``; otherwise 404.
  Creates a fresh demo patient per call (its own chat history), returns a
  short-lived token, and first purges demo users older than 24 hours.
  Rate limited per client IP. Behind a proxy that is not configured to pass
  the real client address, all callers share one IP and the limit is global.

Logs carry user ids only, never emails or passwords.
"""

import logging
import threading
from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.rate_limit import RateLimitExceeded, SlidingWindowRateLimiter
from app.core.security import burn_password_check, create_access_token, verify_password
from app.crud.crud_user import create_demo_user, get_user_by_email, purge_demo_users
from app.schemas.schemas_auth import LoginRequest, TokenResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["Auth"])

_INVALID_LOGIN = "Incorrect email or password"

_demo_limiter: Optional[SlidingWindowRateLimiter] = None
_demo_limiter_lock = threading.Lock()


def get_demo_rate_limiter() -> SlidingWindowRateLimiter:
    """Process-wide per-IP limiter for demo logins (window: one hour)."""
    global _demo_limiter
    with _demo_limiter_lock:
        if _demo_limiter is None:
            _demo_limiter = SlidingWindowRateLimiter(
                limit=settings.DEMO_RATE_LIMIT_PER_HOUR, window_seconds=3600.0
            )
    return _demo_limiter


def reset_demo_rate_limiter() -> None:
    """Drop the demo limiter so the next call rebuilds it (test helper)."""
    global _demo_limiter
    with _demo_limiter_lock:
        _demo_limiter = None


@router.post(
    "/login", response_model=TokenResponse, summary="Log in with email and password"
)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = get_user_by_email(db, body.email)
    if user is None:
        burn_password_check(body.password)
        logger.info("Login failed: unknown account")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=_INVALID_LOGIN
        )
    if not verify_password(body.password, user.hashed_password) or not user.is_active:
        logger.info("Login failed for user_id=%s", user.id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=_INVALID_LOGIN
        )

    minutes = settings.ACCESS_TOKEN_EXPIRE_MINUTES
    token = create_access_token(user.id, user.role, timedelta(minutes=minutes))
    logger.info("Login succeeded for user_id=%s", user.id)
    return TokenResponse(access_token=token, expires_in=minutes * 60)


@router.post(
    "/demo", response_model=TokenResponse, summary="Start a throwaway demo session"
)
def demo_login(request: Request, db: Session = Depends(get_db)) -> TokenResponse:
    if not settings.DEMO_LOGIN_ENABLED:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")

    client_ip = request.client.host if request.client else "unknown"
    try:
        get_demo_rate_limiter().check(client_ip)
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many demo sessions. Try again in {exc.retry_after_seconds} seconds.",
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc

    purged = purge_demo_users(db)
    user = create_demo_user(db)
    minutes = settings.DEMO_TOKEN_EXPIRE_MINUTES
    token = create_access_token(user.id, user.role, timedelta(minutes=minutes))
    logger.info(
        "Demo session created for user_id=%s (purged %d stale demo users)",
        user.id,
        purged,
    )
    return TokenResponse(access_token=token, expires_in=minutes * 60)
