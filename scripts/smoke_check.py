"""Post-deploy smoke check for a Hoku AI chatbot URL.

Stdlib only and no API key: it calls the service over HTTP with the public
demo-login flow (DEMO_LOGIN_ENABLED=true on the service), the same way the
demo page does. About 6 live LLM calls per run, paced under the per-user
chat limit (10/min).

Checks:
  /health (status ok, rag_ready), the demo page at / (CSP, banner), a demo
  login, chat without a token (401), the 4 demo-page examples (expected
  intent, fallback_used=false; the emergency example must set
  X-Hoku-Emergency; the symptom example must return a doctor suggestion),
  two WARN-only scope probes (off-topic questions that pass the retrieval
  gate, see data/eval/README.md; the reply should decline), and, unless
  --skip-rate-limit: the per-user chat limit and the per-IP chat cap with
  spoofed X-Forwarded-For entries. The rate-limit checks send empty messages
  (rejected as invalid, so no LLM call), which the limiters still count; each
  first waits 61 s for a fresh window.

  --check-demo-limit also checks the demo-login limit (20/hour per IP). It
  uses up that hour's demo logins for your IP, so the demo page will not
  start a session from your network for up to an hour.

Usage:
    python scripts/smoke_check.py --url https://your-service-url --out C:/path/outside/repo/smoke.json
"""

import argparse
import base64
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE_MARKER = '<meta name="hoku-demo" content="chat">'
LOGIN_PATH = "/api/auth/demo"
CHAT_PATH = "/api/ai/chat"
# The demo page's example buttons (app/static/index.html).
EMERGENCY_EXAMPLE = "My father has chest pain and can't breathe"
EXAMPLES = (
    # name, message, accepted intents, needs doctor suggestion
    ("example: emergency", EMERGENCY_EXAMPLE, {"emergency"}, False),
    (
        "example: symptom",
        "I've had an itchy skin rash on my arm for a week",
        {"symptom", "general"},
        True,
    ),
    ("example: FAQ", "Do you provide home nursing care in Lahore?", {"general"}, False),
    (
        "example: booking",
        "How soon can I get a home visit after booking?",
        {"booking", "general"},
        False,
    ),
)
# Held-out off-topic questions that pass the 0.35 retrieval gate. The reply
# should decline (scope rule in the prompt) instead of repeating the matched
# FAQ. WARN-only: a text heuristic; read the printed reply yourself.
SCOPE_PROBES = (
    (
        "scope: shopping mall",
        "What are the opening hours of Lahore's Emporium mall?",
        "2-4 hours",
    ),
    ("scope: other company", "Does Amazon deliver to Lahore?", "local pharmacies"),
)
# Phrases of a decline. ("health" alone is not one: almost every reply says it.)
DECLINE_MARKERS = (
    "only help",
    "can only",
    "not able to help",
    "can't help",
    "cannot help",
    "outside",
)
INVALID_STATUSES = (400, 422)
RATE_WINDOW_WAIT_S = 61
CHAT_PACE_S = 6.5
CHAT_LIMIT = 10
CHAT_IP_LIMIT = 20
DEMO_LOGIN_LIMIT = 20


def resolve_out_path(out: str) -> Path:
    """Absolute output path; refuses anything inside the repository."""
    path = Path(out).resolve()
    if path == REPO_ROOT or REPO_ROOT in path.parents:
        raise ValueError(f"--out must be outside the repository ({REPO_ROOT})")
    return path


def normalize_base_url(url: str) -> str:
    if not re.match(r"^https?://[^/\s]+", url):
        raise ValueError("--url must start with http:// or https://")
    return url.rstrip("/")


def http_request(method, url, body=None, headers=None, timeout=90):
    """(status, lower-cased headers, text, elapsed s); HTTP errors are returned, not raised."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, resp_headers, raw = resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as err:
        status, resp_headers, raw = err.code, err.headers, err.read()
    elapsed = time.perf_counter() - start
    lowered = {k.lower(): v for k, v in resp_headers.items()}
    return status, lowered, raw.decode("utf-8", errors="replace"), elapsed


def _json(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def token_subject(token: str) -> int:
    """The user id in the token's sub claim (what the demo page sends as userId)."""
    part = token.split(".")[1]
    part += "=" * (-len(part) % 4)
    return int(json.loads(base64.urlsafe_b64decode(part))["sub"])


def _result(name, status, detail, **extra):
    print(f"[{status}] {name}: {detail}")
    return {"check": name, "status": status, "detail": detail, **extra}


class Session:
    """One demo login: token plus the user id the chat API expects."""

    def __init__(self, base, headers=None):
        self.status, _, text, _ = http_request(
            "POST", f"{base}{LOGIN_PATH}", headers=headers
        )
        self.token = _json(text).get("access_token") if self.status == 200 else None
        self.user_id = token_subject(self.token) if self.token else None

    def chat(self, base, message, extra_headers=None):
        headers = dict(extra_headers or {})
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        body = {"message": message, "userId": self.user_id or 1}
        status, resp_headers, text, elapsed = http_request(
            "POST", f"{base}{CHAT_PATH}", body, headers
        )
        return status, resp_headers, _json(text), elapsed


def check_health(base):
    status, _, text, _ = http_request("GET", f"{base}/health")
    data = _json(text)
    if status != 200:
        return _result("health", "FAIL", f"HTTP {status}")
    ok = data.get("status") == "ok" and data.get("rag_ready") is True
    return _result("health", "PASS" if ok else "FAIL", json.dumps(data))


def check_demo_page(base):
    status, headers, text, _ = http_request("GET", f"{base}/")
    csp = headers.get("content-security-policy", "")
    ok = (
        status == 200
        and PAGE_MARKER in text
        and "Not medical advice." in text
        and "script-src 'sha256-" in csp
        and "unsafe-inline" not in csp
    )
    return _result(
        "demo page",
        "PASS" if ok else "FAIL",
        f"HTTP {status}, CSP {'hash-based' if 'sha256' in csp else 'missing'}",
    )


def check_demo_login(session):
    ok = session.status == 200 and session.token and session.user_id
    return _result("demo login", "PASS" if ok else "FAIL", f"HTTP {session.status}")


def check_requires_token(base):
    status, _, _, _ = http_request(
        "POST", f"{base}{CHAT_PATH}", {"message": "hello", "userId": 1}
    )
    return _result(
        "chat without token",
        "PASS" if status == 401 else "FAIL",
        f"HTTP {status} (expected 401)",
    )


def check_example(base, session, name, message, intents, needs_doctor):
    status, headers, data, elapsed = session.chat(base, message)
    if status != 200:
        return _result(name, "FAIL", f"HTTP {status}")
    problems = []
    if data.get("intent") not in intents:
        problems.append(
            f"intent={data.get('intent')} (expected {'/'.join(sorted(intents))})"
        )
    if data.get("fallback_used") is not False:
        problems.append(f"fallback_used={data.get('fallback_used')}")
    if "emergency" in intents and headers.get("x-hoku-emergency") != "true":
        problems.append("no X-Hoku-Emergency header")
    if needs_doctor and not data.get("doctor_suggestion"):
        problems.append("no doctor_suggestion")
    if "Please consult a doctor" not in str(data.get("reply", "")):
        problems.append("disclaimer missing")
    detail = "; ".join(problems) or f"intent={data.get('intent')}, {elapsed:.2f}s"
    return _result(
        name, "FAIL" if problems else "PASS", detail, reply=data.get("reply")
    )


def check_scope_probe(base, session, name, message, leaked_fact):
    status, _, data, _ = session.chat(base, message)
    reply = str(data.get("reply", ""))
    print(f"    reply: {reply}")
    if status != 200:
        return _result(name, "WARN", f"HTTP {status}", reply=reply)
    declined = any(marker in reply.lower() for marker in DECLINE_MARKERS)
    leaked = leaked_fact.lower() in reply.lower()
    ok = declined and not leaked and data.get("fallback_used") is False
    detail = f"declined={declined}, repeated FAQ fact={leaked}, fallback_used={data.get('fallback_used')}"
    return _result(name, "PASS" if ok else "WARN", detail, reply=reply)


def _wait():
    print(f"    waiting {RATE_WINDOW_WAIT_S} s for a fresh rate-limit window...")
    time.sleep(RATE_WINDOW_WAIT_S)


def limit_reached(statuses, ok_statuses):
    """True if the last status is 429 and all earlier ones were allowed."""
    return (
        bool(statuses)
        and statuses[-1] == 429
        and all(s in ok_statuses for s in statuses[:-1])
    )


def check_chat_rate_limit(base):
    _wait()
    session = Session(base)
    statuses = [session.chat(base, "")[0] for _ in range(CHAT_LIMIT + 1)]
    ok = limit_reached(statuses, INVALID_STATUSES)
    return _result(
        "per-user chat limit", "PASS" if ok else "FAIL", f"statuses={statuses}"
    )


def check_chat_ip_cap(base):
    """CHAT_IP_LIMIT+1 requests from 3 sessions; spoofed left-hand XFF entries must not help."""
    _wait()
    sessions = [Session(base) for _ in range(3)]
    statuses = []
    for i in range(CHAT_IP_LIMIT + 1):
        spoof = {"X-Forwarded-For": f"203.0.113.{i + 1}"}
        statuses.append(sessions[i % 3].chat(base, "", spoof)[0])
    ok = limit_reached(statuses, INVALID_STATUSES)
    return _result("per-IP chat cap", "PASS" if ok else "FAIL", f"statuses={statuses}")


def check_demo_login_limit(base):
    statuses = [
        Session(base, {"X-Forwarded-For": f"198.51.100.{i + 1}"}).status
        for i in range(DEMO_LOGIN_LIMIT + 1)
    ]
    ok = 429 in statuses and all(s in (200, 429) for s in statuses)
    return _result("demo login limit", "PASS" if ok else "FAIL", f"statuses={statuses}")


def run(base, skip_rate_limit=False, check_demo_limit=False):
    results = [check_health(base), check_demo_page(base)]
    session = Session(base)
    results.append(check_demo_login(session))
    results.append(check_requires_token(base))
    if session.token:
        for name, message, intents, needs_doctor in EXAMPLES:
            results.append(
                check_example(base, session, name, message, intents, needs_doctor)
            )
            time.sleep(CHAT_PACE_S)
        for name, message, leaked in SCOPE_PROBES:
            results.append(check_scope_probe(base, session, name, message, leaked))
            time.sleep(CHAT_PACE_S)
    if not skip_rate_limit:
        results.append(check_chat_rate_limit(base))
        results.append(check_chat_ip_cap(base))
    if check_demo_limit:
        results.append(check_demo_login_limit(base))
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", required=True, help="Base URL of the service")
    parser.add_argument(
        "--out", required=True, help="JSON results file, outside the repository"
    )
    parser.add_argument(
        "--skip-rate-limit",
        action="store_true",
        help="Skip the chat rate-limit checks (saves ~2 min)",
    )
    parser.add_argument(
        "--check-demo-limit",
        action="store_true",
        help="Also check the demo-login limit (uses up the hour's logins for your IP)",
    )
    args = parser.parse_args(argv)
    base = normalize_base_url(args.url)
    out = resolve_out_path(args.out)

    results = run(base, args.skip_rate_limit, args.check_demo_limit)
    counts = {
        s: sum(r["status"] == s for r in results) for s in ("PASS", "WARN", "FAIL")
    }
    print(f"\n{counts['PASS']} PASS, {counts['WARN']} WARN, {counts['FAIL']} FAIL")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"url": base, "counts": counts, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {out}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
