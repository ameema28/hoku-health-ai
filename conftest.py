"""Root conftest: test environment defaults, and the repo root on sys.path.

Runs before any app module is imported. ENVIRONMENT=test also stops the
settings classes from reading a local .env file. The JWT signing key is
random per run, so no key ever comes from a file.
"""

import os
import secrets
import sys
from pathlib import Path

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("SECRET_KEY", secrets.token_urlsafe(48))

sys.path.insert(0, str(Path(__file__).parent))
