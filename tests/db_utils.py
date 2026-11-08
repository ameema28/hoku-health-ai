"""Throwaway SQLite files for tests, kept outside the repo and removed at exit."""

import atexit
import shutil
import tempfile
from pathlib import Path

TEST_DB_DIR = Path(tempfile.mkdtemp(prefix="hoku-tests-"))
atexit.register(shutil.rmtree, TEST_DB_DIR, ignore_errors=True)


def sqlite_test_url(name: str) -> str:
    """URL of a SQLite file named ``name`` in the per-run temp directory."""
    return f"sqlite:///{(TEST_DB_DIR / name).as_posix()}"
