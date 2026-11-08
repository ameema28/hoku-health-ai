"""Hoku Health Care - Auth request/response schemas."""

from pydantic import BaseModel, Field, field_validator

from app.core.security import MAX_PASSWORD_BYTES


class LoginRequest(BaseModel):
    """JSON login body."""

    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=1)

    @field_validator("password")
    @classmethod
    def _max_bytes(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_PASSWORD_BYTES:
            raise ValueError(f"password must be at most {MAX_PASSWORD_BYTES} bytes")
        return value


class TokenResponse(BaseModel):
    """Bearer token issued by login or demo."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Lifetime in seconds")
