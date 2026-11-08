"""
Hoku Health Care - Create a user account (there is no default admin).

Usage:
    python -m app.scripts.create_user --email ops@example.com --role admin
    python -m app.scripts.create_user --email ops@example.com --reset-password

The password is read from the HOKU_NEW_USER_PASSWORD environment variable
if set, otherwise prompted for twice (input hidden). It is never printed.
An existing user is left untouched unless --reset-password is given, which
changes only the password.
"""

import argparse
import getpass
import os
import sys
from typing import Callable, List, Optional

from sqlalchemy.orm import Session

from app.core.security import password_policy_error
from app.crud.crud_user import create_user, get_user_by_email, set_password
from app.models.user import ROLES

PASSWORD_ENV = "HOKU_NEW_USER_PASSWORD"


def _read_password() -> Optional[str]:
    from_env = os.environ.get(PASSWORD_ENV)
    if from_env:
        return from_env
    first = getpass.getpass("New password: ")
    second = getpass.getpass("Repeat password: ")
    if first != second:
        print("Passwords do not match.", file=sys.stderr)
        return None
    return first


def main(
    argv: Optional[List[str]] = None,
    session_factory: Optional[Callable[[], Session]] = None,
) -> int:
    parser = argparse.ArgumentParser(description="Create a Hoku user account.")
    parser.add_argument("--email", required=True)
    parser.add_argument("--role", choices=ROLES, default="patient")
    parser.add_argument(
        "--reset-password",
        action="store_true",
        help="set a new password for an existing user",
    )
    args = parser.parse_args(argv)

    if session_factory is None:
        from app.core.database import SessionLocal

        session_factory = SessionLocal

    db = session_factory()
    try:
        existing = get_user_by_email(db, args.email)
        if existing is not None and not args.reset_password:
            print(
                "A user with this email already exists; nothing changed. "
                "Use --reset-password to set a new password.",
                file=sys.stderr,
            )
            return 1
        if existing is None and args.reset_password:
            print("No user with this email; nothing changed.", file=sys.stderr)
            return 1

        password = _read_password()
        if password is None:
            return 2
        problem = password_policy_error(password)
        if problem:
            print(problem, file=sys.stderr)
            return 2

        if existing is not None:
            set_password(db, existing, password)
            print(f"Password updated for user id={existing.id}.")
        else:
            user = create_user(db, args.email, password, role=args.role)
            print(f"Created user id={user.id} (role={user.role}).")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
