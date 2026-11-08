"""Direct unit tests for JWT security helpers and input validators (Day 10 margin)."""

from __future__ import annotations

import datetime

import pytest
from jose import JWTError, jwt

from app.core.config import settings
from app.core.security import JWT_ALGORITHM, create_access_token, decode_access_token


class TestSecurityToken:
    """The JWT helpers behind get_current_user."""

    def test_valid_token_decodes(self) -> None:
        """A token from create_access_token decodes to its subject and role."""
        decoded = decode_access_token(create_access_token(1, "patient"))
        assert decoded["sub"] == "1"
        assert decoded["role"] == "patient"

    def test_expired_token_rejected(self) -> None:
        """An expired token raises during decode."""
        token = create_access_token(1, "patient", datetime.timedelta(seconds=-60))
        with pytest.raises(JWTError):
            decode_access_token(token)

    def test_wrong_signature_rejected(self) -> None:
        """A token signed with the wrong key is rejected."""
        now = datetime.datetime.now(datetime.timezone.utc)
        token = jwt.encode(
            {
                "sub": "1",
                "role": "patient",
                "exp": now + datetime.timedelta(hours=1),
                "iss": settings.JWT_ISSUER,
                "aud": settings.JWT_AUDIENCE,
            },
            "wrong-secret-key-wrong-secret-key-000",
            algorithm=JWT_ALGORITHM,
        )
        with pytest.raises(JWTError):
            decode_access_token(token)


class TestValidatorsExtra:
    """Edge cases for sanitize_message / validate_message_length."""

    def test_sanitize_empty_string(self) -> None:
        from app.utils.validators import sanitize_message

        assert sanitize_message("") == ""

    def test_sanitize_whitespace_only(self) -> None:
        from app.utils.validators import sanitize_message

        assert sanitize_message("   ").strip() == ""

    def test_validate_length_empty_is_false(self) -> None:
        from app.utils.validators import validate_message_length

        assert validate_message_length("") is False

    def test_validate_length_boundary(self) -> None:
        from app.utils.validators import validate_message_length

        assert validate_message_length("a valid short message") is True
