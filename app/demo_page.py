"""
The public demo chat page served at GET / (app/static/index.html).

The page is one self-contained HTML file with exactly one inline <style> and
one inline <script>. It is read once at import, and the Content-Security-Policy
allows exactly those two blocks by their sha256 hashes (no 'unsafe-inline'),
plus same-origin fetch calls. Editing the page changes the hashes
automatically on the next start.

The page calls POST /api/auth/demo (only works with DEMO_LOGIN_ENABLED=true)
and keeps the token in memory only.
"""

import base64
import hashlib
import re
from pathlib import Path
from typing import Dict, Tuple

DEMO_PAGE_PATH = Path(__file__).resolve().parent / "static" / "index.html"

_SCRIPT_RE = re.compile(r"<script>(.*?)</script>", re.DOTALL)
_STYLE_RE = re.compile(r"<style>(.*?)</style>", re.DOTALL)


def csp_hash(source: str) -> str:
    """CSP source expression for an inline block, e.g. 'sha256-...'."""
    digest = hashlib.sha256(source.encode("utf-8")).digest()
    return "'sha256-" + base64.b64encode(digest).decode("ascii") + "'"


def load_demo_page(path: Path = DEMO_PAGE_PATH) -> Tuple[str, Dict[str, str]]:
    """
    Return (html, response headers) for the demo page.

    Raises:
        RuntimeError: the page does not have exactly one inline <script> and
            one inline <style> (the CSP would not match it).
    """
    html = path.read_text(encoding="utf-8")
    scripts = _SCRIPT_RE.findall(html)
    styles = _STYLE_RE.findall(html)
    if len(scripts) != 1 or len(styles) != 1:
        raise RuntimeError(
            "demo page must have exactly one inline <script> and one <style>"
        )
    policy = "; ".join(
        [
            "default-src 'none'",
            f"script-src {csp_hash(scripts[0])}",
            f"style-src {csp_hash(styles[0])}",
            "connect-src 'self'",
            "img-src 'self'",
            "base-uri 'none'",
            "form-action 'none'",
            "frame-ancestors 'none'",
        ]
    )
    headers = {
        "Content-Security-Policy": policy,
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Cache-Control": "no-cache",
    }
    return html, headers


DEMO_PAGE_HTML, DEMO_PAGE_HEADERS = load_demo_page()
