"""Engine arguments from settings, and Neon-style URLs with sslmode."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine

from app.core.config import Settings
from app.core.database import engine_kwargs

NEON_STYLE = (
    "postgresql://app_user:not-a-secret@ep-example-123.eu-central-1.aws.neon.tech"
    "/appdb?sslmode=require&channel_binding=require"
)


def _cfg(url, pool=2, overflow=3):
    return SimpleNamespace(
        DATABASE_URL=url,
        DB_POOL_SIZE=pool,
        DB_MAX_OVERFLOW=overflow,
        DEBUG=False,
        is_production=True,
    )


def test_postgres_pool_comes_from_settings():
    kwargs = engine_kwargs(_cfg(NEON_STYLE))
    assert kwargs["pool_size"] == 2
    assert kwargs["max_overflow"] == 3
    assert kwargs["pool_pre_ping"] is True
    assert kwargs["echo"] is False


def test_sqlite_has_no_pool_size_and_allows_threads():
    kwargs = engine_kwargs(_cfg("sqlite:///./x.db"))
    assert "pool_size" not in kwargs
    assert kwargs["connect_args"] == {"check_same_thread": False}


def test_sslmode_reaches_the_driver_unchanged():
    engine = create_engine(NEON_STYLE, **engine_kwargs(_cfg(NEON_STYLE)))
    _, params = engine.dialect.create_connect_args(engine.url)
    assert params["sslmode"] == "require"
    assert params["channel_binding"] == "require"
    assert engine.pool.size() == 2
    engine.dispose()


def test_defaults_and_validation():
    fresh = Settings()
    assert (fresh.DB_POOL_SIZE, fresh.DB_MAX_OVERFLOW) == (10, 20)
    with pytest.raises(ValidationError):
        Settings(DB_POOL_SIZE=0)
