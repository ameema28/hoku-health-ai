"""
Hoku Health Care - Database Module (Day 8: Connection Pooling Optimization).

Configures SQLAlchemy engine with optimized connection pooling parameters
for production resilience and NFR-02 compliance.

Day 8 updates:
- pool_size / max_overflow from DB_POOL_SIZE / DB_MAX_OVERFLOW (defaults 10 / 20)
- pool_recycle=3600 to prevent stale connections
- pool_pre_ping=True for connection health validation
- SQLite thread fallback (check_same_thread=False)
"""

from typing import Any, Dict, Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings

# ------------------------------------------------------------------
# Connection Pool Configuration (Day 8)
# ------------------------------------------------------------------
# Pooling keeps DB connections from becoming a bottleneck under
# concurrent load, which matters for the 4 s response budget.
# pool_pre_ping=True validates connections on checkout, preventing
# errors from stale/closed connections (~50-100ms saved per stale hit).
# ------------------------------------------------------------------


def sql_echo_enabled(cfg=settings) -> bool:
    """SQL echo logs bound parameters (patient messages): never in production."""
    return bool(cfg.DEBUG) and not cfg.is_production


def engine_kwargs(cfg=settings) -> Dict[str, Any]:
    """create_engine() arguments for the configured database."""
    kwargs: Dict[str, Any] = {
        "pool_pre_ping": True,  # also covers a hosted DB that suspended idle
        "pool_recycle": 3600,  # Recycle connections after 1 hour
        "echo": sql_echo_enabled(cfg),
    }
    if cfg.DATABASE_URL.startswith("sqlite"):
        # SQLite: allow cross-thread use; its pool ignores size/overflow.
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_size"] = cfg.DB_POOL_SIZE
        kwargs["max_overflow"] = cfg.DB_MAX_OVERFLOW
    return kwargs


engine = create_engine(settings.DATABASE_URL, **engine_kwargs())

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Iterator[Session]:
    """
    Yield a database session for dependency injection.

    Ensures sessions are properly closed after each request,
    preventing connection leaks in long-running processes.

    Yields:
        Session: SQLAlchemy database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
