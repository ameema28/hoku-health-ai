"""Latency benchmark for a Hoku AI chatbot URL (demo-login flow).

Stdlib only, no API key. Gets one demo session, sends one warm-up chat (not
counted), then N sequential chats over the three non-emergency demo-page
examples (the emergency example skips the LLM, so it would not measure it).
Requests start --interval seconds apart; the default 6.5 s stays under the
per-user limit of 10 chats per minute.

Prints nearest-rank p50 / p95 / max for the client time (includes your
network) and for the server's own X-Response-Time-Sec header, and how many
replies had fallback_used=true. Exits non-zero unless every measured request
returned 200.

Usage:
    python scripts/benchmark.py --url https://your-service-url -n 20 --out C:/path/outside/repo/bench.json
"""

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from smoke_check import (  # noqa: E402
    EXAMPLES,
    REPO_ROOT,
    Session,
    normalize_base_url,
)

MESSAGES = [
    message for _, message, intents, _ in EXAMPLES if "emergency" not in intents
]
MIN_INTERVAL_S = 6.0


def resolve_out_path(out):
    """Output path outside the repository (default: the system temp dir)."""
    path = (
        Path(out).resolve()
        if out
        else Path(tempfile.gettempdir()) / "hoku_benchmark.json"
    )
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise ValueError(f"--out must be outside the repository ({REPO_ROOT})")
    return path


def validate_interval(interval: float, force: bool) -> float:
    if interval < MIN_INTERVAL_S and not force:
        raise ValueError(
            f"--interval below {MIN_INTERVAL_S}s would exceed 10 chats/min per user; use --force to override"
        )
    return interval


def percentile(values, p):
    """Nearest-rank percentile (p in 0..100), None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def stats(values):
    return {
        "n": len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values) if values else None,
    }


def one_request(base, session, message):
    status, headers, data, elapsed = session.chat(base, message)
    server = headers.get("x-response-time-sec")
    try:
        server_s = float(server) if server is not None else None
    except ValueError:
        server_s = None
    return {
        "message": message,
        "status": status,
        "client_s": round(elapsed, 3),
        "server_s": server_s,
        "fallback_used": data.get("fallback_used"),
        "intent": data.get("intent"),
    }


def summarize(records):
    ok = [r for r in records if r["status"] == 200]
    return {
        "requests": len(records),
        "ok": len(ok),
        "fallback_used": sum(r["fallback_used"] is True for r in ok),
        "client_s": stats([r["client_s"] for r in ok]),
        "server_s": stats([r["server_s"] for r in ok if r["server_s"] is not None]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", required=True, help="Base URL of the service")
    parser.add_argument(
        "-n", type=int, default=20, help="Measured requests (default 20)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=6.5,
        help="Seconds between request starts (default 6.5)",
    )
    parser.add_argument(
        "--force", action="store_true", help=f"Allow --interval below {MIN_INTERVAL_S}s"
    )
    parser.add_argument(
        "--out",
        help="JSON results file, outside the repository (default: system temp dir)",
    )
    args = parser.parse_args(argv)
    base = normalize_base_url(args.url)
    interval = validate_interval(args.interval, args.force)
    out = resolve_out_path(args.out)

    session = Session(base)
    if not session.token:
        print(f"Demo login failed (HTTP {session.status})")
        return 1

    warm = one_request(base, session, MESSAGES[0])
    print(f"warm-up: HTTP {warm['status']}, {warm['client_s']}s (not counted)")
    records = []
    next_start = time.monotonic() + interval
    for i in range(args.n):
        delay = next_start - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        next_start = time.monotonic() + interval
        record = one_request(base, session, MESSAGES[i % len(MESSAGES)])
        records.append(record)
        print(
            f"{i + 1:3d} HTTP {record['status']} client={record['client_s']}s "
            f"server={record['server_s']}s fallback={record['fallback_used']}"
        )

    summary = summarize(records)
    print(json.dumps(summary, indent=2))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"url": base, "summary": summary, "records": records}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out}")
    return 0 if summary["ok"] == summary["requests"] else 1


if __name__ == "__main__":
    sys.exit(main())
