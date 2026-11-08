"""
Hoku Health Care - Authentication and authorization primitives.

- Passwords: bcrypt (the ``bcrypt`` package directly; passlib 1.7.4 cannot
  hash with bcrypt 5). Passwords are 12+ characters and at most 72 UTF-8
  bytes, bcrypt's input limit, so nothing is silently truncated.
- Tokens: HS256 JWTs carrying ``sub`` (user id), ``role``, ``iat``, ``exp``,
  ``iss`` and ``aud``. The algorithm is fixed here, not configurable, so a
  token signed with another algorithm (including ``none``) is rejected.
- ``get_current_user`` re-reads the user on every request: deactivated or
  deleted users, and tokens whose role no longer matches the database,
  are rejected with 401.
- ``validate_security_settings`` is the startup guard for SECRET_KEY.
"""

import logging
import secrets
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Dict, Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from app.core.config import Settings, settings
from app.core.database import get_db
from app.models.user import ROLES, User

logger = logging.getLogger(__name__)

JWT_ALGORITHM = "HS256"
MIN_PASSWORD_LENGTH = 12
MAX_PASSWORD_BYTES = 72
MIN_SECRET_KEY_LENGTH = 32

# Fixed message: never interpolate the key (or its length) into it.
_SECRET_KEY_ERROR = (
    "SECRET_KEY must be at least 32 characters outside the test environment"
)

# HTTPBearer gives Swagger a simple "Value" field for Bearer tokens
security = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Startup guard
# ---------------------------------------------------------------------------
def validate_security_settings(cfg: Settings) -> None:
    """
    Refuse to run with a missing or short SECRET_KEY outside tests.

    A plain function rather than a pydantic validator on purpose: pydantic
    validation errors include ``input_value``, which would print the key.

    Raises:
        RuntimeError: with a fixed message that never contains the key.
    """
    if (
        cfg.ENVIRONMENT.lower() != "test"
        and len(cfg.SECRET_KEY) < MIN_SECRET_KEY_LENGTH
    ):
        raise RuntimeError(_SECRET_KEY_ERROR)


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------
def password_policy_error(password: str) -> Optional[str]:
    """Return why ``password`` is not acceptable as a new password, or None."""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return f"Password must be at most {MAX_PASSWORD_BYTES} bytes (UTF-8)."
    return None


def get_password_hash(password: str) -> str:
    """
    Hash a password with bcrypt.

    Raises:
        ValueError: if the password exceeds 72 UTF-8 bytes.
    """
    raw = password.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        raise ValueError("password exceeds 72 bytes")
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode("ascii")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Constant-time bcrypt check. Unusable or malformed hashes never match."""
    raw = plain_password.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(raw, hashed_password.encode("ascii"))
    except (ValueError, UnicodeEncodeError):
        return False


def unusable_password_hash() -> str:
    """A random value that is not a bcrypt hash, so no password matches it."""
    return "!" + secrets.token_urlsafe(32)


@lru_cache(maxsize=1)
def _dummy_hash() -> str:
    return get_password_hash(secrets.token_urlsafe(16))


def burn_password_check(plain_password: str) -> None:
    """Spend one bcrypt verify so unknown emails cost as much as known ones."""
    verify_password(plain_password, _dummy_hash())


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
def create_access_token(
    user_id: int, role: str, expires_delta: Optional[timedelta] = None
) -> str:
    """
    Sign an access token for ``user_id`` with ``role``.

    Args:
        user_id: The user's primary key (becomes ``sub``).
        role: One of ``ROLES``.
        expires_delta: Lifetime; defaults to ACCESS_TOKEN_EXPIRE_MINUTES.
    """
    if role not in ROLES:
        raise ValueError("unknown role")
    now = datetime.now(timezone.utc)
    lifetime = expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    claims = {
        "sub": str(user_id),
        "role": role,
        "iat": now,
        "exp": now + lifetime,
        "iss": settings.JWT_ISSUER,
        "aud": settings.JWT_AUDIENCE,
    }
    return jwt.encode(claims, settings.SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Verify signature, algorithm, exp, iss and aud; require sub and role.

    Raises:
        JWTError: on any validation failure.
    """
    payload = jwt.decode(
        token,
        settings.SECRET_KEY,
        algorithms=[JWT_ALGORITHM],
        audience=settings.JWT_AUDIENCE,
        issuer=settings.JWT_ISSUER,
        options={
            "require_exp": True,
            "require_iss": True,
            "require_aud": True,
            "require_sub": True,
        },
    )
    if payload.get("role") not in ROLES:
        raise JWTError("invalid role claim")
    return payload


def _unauthorized(detail: str = "Could not validate credentials") -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def user_to_principal(user: User) -> Dict[str, Any]:
    """The dict shape endpoints receive as ``current_user``."""
    return {
        "id": user.id,
        "role": user.role,
        "email": user.email,
        "is_demo": user.is_demo,
    }


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Resolve the Bearer token to an active user.

    Returns:
        Dict[str, Any]: ``{"id", "role", "email", "is_demo"}``.

    Raises:
        HTTPException: 401 for a missing/invalid/expired token, an unknown or
            inactive user, or a token role that differs from the database.
    """
    if credentials is None:
        raise _unauthorized("Not authenticated")

    try:
        payload = decode_access_token(credentials.credentials)
        user_id = int(payload["sub"])
    except (JWTError, ValueError, TypeError) as exc:
        logger.warning("JWT validation failed: %s", type(exc).__name__)
        raise _unauthorized() from exc

    user = db.get(User, user_id)
    if user is None or not user.is_active or user.role != payload["role"]:
        logger.warning(
            "JWT rejected for user_id=%s (unknown, inactive or role changed)", user_id
        )
        raise _unauthorized()
    return user_to_principal(user)
