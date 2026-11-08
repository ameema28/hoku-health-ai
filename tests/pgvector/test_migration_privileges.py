"""
Migration 003 with a role that may not enable pgvector (pgvector is not a
trusted extension). This stands in for hosted Postgres where the app role is
not a superuser. Needs DATABASE_URL to be a PostgreSQL superuser; a throwaway
role and database are created and dropped.
"""

import secrets

import pytest
from alembic import command
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from app.core.config import settings

pytestmark = pytest.mark.pgvector

IS_POSTGRES = settings.DATABASE_URL.startswith("postgresql")


@pytest.fixture
def unprivileged_db():
    if not IS_POSTGRES:
        pytest.skip("needs DATABASE_URL=postgresql://... (superuser)")
    admin = create_engine(settings.DATABASE_URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        if not conn.execute(
            text("SELECT rolsuper FROM pg_roles WHERE rolname = current_user")
        ).scalar():
            pytest.skip("DATABASE_URL role is not a superuser")
        role = "hoku_t_" + secrets.token_hex(4)
        database = "hoku_t_" + secrets.token_hex(4)
        password = secrets.token_urlsafe(16)
        conn.execute(text(f"CREATE ROLE {role} LOGIN PASSWORD '{password}'"))
        conn.execute(text(f"CREATE DATABASE {database} OWNER {role}"))
    base = make_url(settings.DATABASE_URL)
    app_url = base.set(username=role, password=password, database=database)
    owner_url = base.set(database=database)
    try:
        yield app_url, owner_url
    finally:
        with admin.connect() as conn:
            conn.execute(text(f"DROP DATABASE IF EXISTS {database} WITH (FORCE)"))
            conn.execute(text(f"DROP ROLE IF EXISTS {role}"))
        admin.dispose()


def _cfg(url):
    import init_db

    cfg = init_db.alembic_config()
    cfg.attributes["configure_logger"] = False
    cfg.set_main_option(
        "sqlalchemy.url", url.render_as_string(hide_password=False).replace("%", "%%")
    )
    return cfg


def test_upgrade_stops_with_instructions_then_works_after_admin_step(
    unprivileged_db,
):
    app_url, superuser_url = unprivileged_db

    with pytest.raises(RuntimeError, match="CREATE EXTENSION IF NOT EXISTS vector"):
        command.upgrade(_cfg(app_url), "head")

    # The one-time administrator step documented in DEPLOYMENT.md.
    su = create_engine(superuser_url, isolation_level="AUTOCOMMIT")
    with su.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    su.dispose()

    command.upgrade(_cfg(app_url), "head")
    engine = create_engine(app_url)
    with engine.connect() as conn:
        udt = conn.execute(
            text(
                "SELECT udt_name FROM information_schema.columns "
                "WHERE table_name = 'vector_store' AND column_name = 'embedding'"
            )
        ).scalar()
    engine.dispose()
    assert udt == "vector"
