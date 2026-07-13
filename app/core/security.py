"""
Hoku Health Care - Security Module (Stub).

Placeholder implementations for JWT authentication, password hashing,
and user retrieval. Uses HTTPBearer for cleaner Swagger UI token input.
Will be fully implemented by Backend Lead (Talha).
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTPBearer gives Swagger a simple "Value" field for Bearer tokens
security = HTTPBearer(auto_error=False)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Payload to encode in the token.
        expires_delta: Optional custom expiration time.

    Returns:
        str: Encoded JWT string.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.

    Args:
        plain_password: Raw password string.
        hashed_password: Bcrypt hashed password.

    Returns:
        bool: True if password matches.
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        password: Raw password string.

    Returns:
        str: Bcrypt hashed password.
    """
    return pwd_context.hash(password)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """
    Stub dependency to retrieve the current authenticated user.

    Uses HTTPBearer for a clean Swagger UI token input field.
    In production, this will validate the JWT token and query the
    users table. For now, returns a mock user to unblock AI route
    development.

    Args:
        credentials: HTTPAuthorizationCredentials from Bearer header.

    Returns:
        Dict[str, Any]: Mock user dictionary.

    Raises:
        HTTPException: If token is missing or invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        user_id: Optional[int] = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        # Stub: return mock user. Talha will replace with DB lookup.
        return {"id": int(user_id), "role": "patient", "email": "stub@hoku.health"}
    except JWTError as exc:
        logger.warning("JWT validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        ) from exc