#!/usr/bin/env python3
"""
Hoku Health Care - Database initialization.

Brings the configured database (``DATABASE_URL``) to the latest schema with
Alembic, then seeds the sample doctors and the FAQ knowledge base. Every
step is idempotent, so it is safe to run on every deploy.

Usage:
    python init_db.py            # migrate + seed doctors + seed FAQs
    python init_db.py --no-faqs  # migrate + seed doctors only
"""

import argparse
from pathlib import Path
from typing import List, Optional

from alembic import command
from alembic.config import Config

ROOT = Path(__file__).resolve().parent


def alembic_config() -> Config:
    """Alembic config for this repo, independent of the current directory."""
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    return cfg


def run_migrations() -> None:
    """Upgrade the configured database to the latest revision."""
    command.upgrade(alembic_config(), "head")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-faqs", action="store_true", help="skip FAQ seeding")
    args = parser.parse_args(argv)

    print("Running migrations (alembic upgrade head)...")
    run_migrations()

    from app.core.database import SessionLocal
    from app.scripts.seed_doctors import seed_doctors

    db = SessionLocal()
    try:
        result = seed_doctors(db)
    finally:
        db.close()
    print(
        f"Doctors: {result['doctors_added']} added, {result['doctors_skipped']} already present."
    )

    if not args.no_faqs:
        from app.scripts import seed_faqs

        seed_faqs.main()

    print("Database initialized.")


if __name__ == "__main__":
    main()
