"""Root conftest: ensure the repo root is importable for all test packages."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))