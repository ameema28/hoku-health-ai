"""
Client IP for the per-IP rate limits (demo login and the chat IP cap).

With TRUST_PROXY=false (the default) the socket peer is used and any
X-Forwarded-For header is ignored.

With TRUST_PROXY=true the IP is the X-Forwarded-For entry TRUST_PROXY_HOPS
places from the right (default 1 = right-most, the entry the trusted proxy
appended). Entries further left are client-supplied and ignored, so a client
cannot pick its own rate-limit key by sending the header. If the header has
fewer entries than that, the socket peer is used.

uvicorn must run with --no-proxy-headers so request.client stays the socket
peer (see the Dockerfile).
"""

from typing import Optional

from fastapi import Request

from app.core.config import Settings, settings


def client_ip(request: Request, cfg: Optional[Settings] = None) -> str:
    """Return the client IP used as the per-IP rate-limit key."""
    cfg = cfg or settings
    if cfg.TRUST_PROXY:
        joined = ",".join(request.headers.getlist("x-forwarded-for"))
        entries = [entry.strip() for entry in joined.split(",") if entry.strip()]
        hops = cfg.TRUST_PROXY_HOPS
        if len(entries) >= hops:
            return entries[-hops]
    return request.client.host if request.client else "unknown"
