"""
Hoku Health Care - Database Module (Day 8: Connection Pooling Optimization).

Configures SQLAlchemy engine with optimized connection pooling parameters
for production resilience and NFR-02 compliance.

Day 8 updates:
- pool_size=10, max_overflow=20 for concurrent request handling
- pool_recycle=3600 to prevent stale connections
- pool_pre_ping=True for connection health validation
- SQLite thread fallback (check_same_thread=False)
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from app.core.config import settings

# ------------------------------------------------------------------
# Connection Pool Configuration (Day 8)
# ------------------------------------------------------------------
# Production-grade pooling ensures DB connections don't become a
# bottleneck under concurrent load, protecting the <4s NFR-02 ceiling.
# pool_pre_ping=True validates connections on checkout, preventing
# errors from stale/closed connections (~50-100ms saved per stale hit).
# ------------------------------------------------------------------

_pool_kwargs = {
    "pool_pre_ping": True,      # Health-check connections before use
    "pool_size": 10,            # Baseline persistent connections
    "max_overflow": 20,         # Burst capacity under load
    "pool_recycle": 3600,       # Recycle connections after 1 hour
    "echo": settings.DEBUG,
}

# SQLite-specific: allow cross-thread usage for async contexts
if settings.DATABASE_URL.startswith("sqlite"):
    _pool_kwargs["connect_args"] = {"check_same_thread": False}
    # SQLite uses SingletonThreadPool by default; for file-based SQLite
    # we still benefit from pool_pre_ping and pool_recycle settings
    # though pool_size/max_overflow apply to QueuePool on PostgreSQL

engine = create_engine(
    settings.DATABASE_URL,
    **_pool_kwargs,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Session:
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