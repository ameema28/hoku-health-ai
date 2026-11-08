"""
Hoku Health Care - Alembic Environment Configuration.

Runs migrations against ``settings.DATABASE_URL`` unless the caller
(``init_db.py`` or the migration tests) passes a URL via
``sqlalchemy.url`` or an open connection via ``config.attributes``.
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

import app.models  # noqa: F401  (registers every model on Base.metadata)
from app.core.config import settings
from app.core.database import Base

config = context.config

if not config.get_main_option("sqlalchemy.url"):
    # ConfigParser treats "%" as interpolation, so escape it (URL-encoded
    # passwords contain "%XX" sequences).
    config.set_main_option("sqlalchemy.url", settings.DATABASE_URL.replace("%", "%%"))

# Programmatic callers (tests) disable this so Alembic does not replace
# the application's logging handlers.
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without a DB connection)."""
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def _run_with_connection(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode, reusing a caller-supplied connection if given."""
    connection = config.attributes.get("connection")
    if connection is not None:
        _run_with_connection(connection)
        return

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as conn:
        _run_with_connection(conn)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
