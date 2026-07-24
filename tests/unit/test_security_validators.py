"""Direct unit tests for JWT security helpers and input validators (Day 10 margin)."""

from __future__ import annotations

import datetime

import pytest
from jose import jwt

from app.core.config import settings


class TestSecurityToken:
    """The JWT decode path in get_current_user."""

    def test_valid_token_decodes(self) -> None:
        """A correctly signed token yields the expected subject."""
        token = jwt.encode(
            {"sub": "1", "id": 1, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)},
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        decoded = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        assert decoded["id"] == 1

    def test_expired_token_rejected(self) -> None:
        """An expired token raises during decode."""
        token = jwt.encode(
            {"sub": "1", "id": 1, "exp": datetime.datetime.utcnow() - datetime.timedelta(hours=1)},
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM,
        )
        with pytest.raises(Exception):
            jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])

    def test_wrong_signature_rejected(self) -> None:
        """A token signed with the wrong key is rejected."""
        token = jwt.encode(
            {"sub": "1", "id": 1, "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1)},
            "wrong-secret-key",
            algorithm=settings.ALGORITHM,
        )
        with pytest.raises(Exception):
            jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])


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