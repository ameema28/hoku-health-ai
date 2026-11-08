"""app/scripts/dev_token.py and the token_gen.py wrapper."""

import re
from pathlib import Path

import pytest
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.security import decode_access_token
from app.scripts import dev_token
from tests.auth_utils import add_user

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def factory(db):
    return sessionmaker(bind=db.get_bind())


def test_prints_only_a_valid_token_with_db_role(db, factory, capsys):
    add_user(db, user_id=5, role="admin", email="dev@example.com")
    assert dev_token.main(["--email", "DEV@example.com"], factory) == 0
    out = capsys.readouterr().out.strip()
    assert "\n" not in out
    claims = decode_access_token(out)
    assert claims["sub"] == "5"
    assert claims["role"] == "admin"
    assert (
        claims["iss"] == settings.JWT_ISSUER and claims["aud"] == settings.JWT_AUDIENCE
    )
    assert claims["exp"] - claims["iat"] == 15 * 60


def test_custom_lifetime(db, factory, capsys):
    add_user(db, user_id=5, email="dev@example.com")
    assert (
        dev_token.main(["--email", "dev@example.com", "--minutes", "5"], factory) == 0
    )
    claims = decode_access_token(capsys.readouterr().out.strip())
    assert claims["exp"] - claims["iat"] == 5 * 60


@pytest.mark.parametrize("minutes", ["0", "1441"])
def test_lifetime_bounds(db, factory, capsys, minutes):
    add_user(db, user_id=5, email="dev@example.com")
    assert (
        dev_token.main(["--email", "dev@example.com", "--minutes", minutes], factory)
        == 2
    )
    assert capsys.readouterr().out == ""


def test_refuses_in_production(db, factory, monkeypatch, capsys):
    add_user(db, user_id=5, email="dev@example.com")
    monkeypatch.setattr(settings, "ENVIRONMENT", "production")
    assert dev_token.main(["--email", "dev@example.com"], factory) == 1
    out = capsys.readouterr()
    assert out.out == ""
    assert "production" in out.err


def test_unknown_email(db, factory, capsys):
    assert dev_token.main(["--email", "nobody@example.com"], factory) == 1
    assert capsys.readouterr().out == ""


def test_inactive_user(db, factory, capsys):
    add_user(db, user_id=5, email="dev@example.com", is_active=False)
    assert dev_token.main(["--email", "dev@example.com"], factory) == 1
    assert capsys.readouterr().out == ""


def test_token_gen_wraps_dev_token():
    import token_gen

    assert token_gen.main is dev_token.main


@pytest.mark.parametrize("path", ["token_gen.py", "app/scripts/dev_token.py"])
def test_no_hard_coded_secret(path):
    source = (ROOT / path).read_text(encoding="utf-8")
    assert "jwt.encode" not in source
    assert not re.search(r"SECRET_KEY\s*=", source)
    # No long high-entropy literals (keys, tokens).
    for literal in re.findall(r"[\"']([^\"'\s]{24,})[\"']", source):
        assert not re.fullmatch(r"[A-Za-z0-9_\-+/=.]{24,}", literal), literal
