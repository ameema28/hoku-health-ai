"""
Hoku Health Care - Print a short-lived access token for an existing user (dev only).

Usage:
    python -m app.scripts.dev_token --email you@example.com [--minutes 15]

- Refuses to run when ENVIRONMENT=production.
- The user must exist and be active; the role comes from the database.
- Signs with SECRET_KEY via ``create_access_token`` (same claims as login).
- Prints only the token (that is the tool's output); nothing else on stdout.
"""

import argparse
import sys
from datetime import timedelta
from typing import Callable, List, Optional

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import create_access_token
from app.crud.crud_user import get_user_by_email

DEFAULT_MINUTES = 15


def main(
    argv: Optional[List[str]] = None,
    session_factory: Optional[Callable[[], Session]] = None,
) -> int:
    parser = argparse.ArgumentParser(description="Print a dev access token.")
    parser.add_argument("--email", required=True)
    parser.add_argument("--minutes", type=int, default=DEFAULT_MINUTES)
    args = parser.parse_args(argv)

    if settings.is_production:
        print(
            "Refusing to issue dev tokens when ENVIRONMENT=production.", file=sys.stderr
        )
        return 1
    if not 1 <= args.minutes <= 24 * 60:
        print("--minutes must be between 1 and 1440.", file=sys.stderr)
        return 2

    if session_factory is None:
        from app.core.database import SessionLocal

        session_factory = SessionLocal
    db = session_factory()
    try:
        user = get_user_by_email(db, args.email)
        if user is None or not user.is_active:
            print("No active user with this email.", file=sys.stderr)
            return 1
        token = create_access_token(user.id, user.role, timedelta(minutes=args.minutes))
    finally:
        db.close()

    print(token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
