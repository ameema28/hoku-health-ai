#!/usr/bin/env python3
"""
Hoku Health Care - Database Initialization Script.

Run this once after deleting the SQLite database to recreate all tables.
Not needed in production (Alembic handles migrations there).
"""

from app.core.database import Base, engine

# Import all models so they register with Base.metadata
from app.models.chat import ChatHistory  # noqa: F401


def init_database() -> None:
    """Create all SQLAlchemy tables in the configured database."""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully!")


if __name__ == "__main__":
    init_database()