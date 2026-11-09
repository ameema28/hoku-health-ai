# Hoku Health Care — AI Chatbot API Reference

Base URL: your deployment (see `DEPLOYMENT.md`), or `http://localhost:8000`
with docker compose. Interactive docs: `/docs` (Swagger) and `/redoc`.

> Every chatbot reply ends with a disclaimer: **"Please consult a doctor for
> proper diagnosis."** for health, symptom, medication and emergency replies,
> **"General information only, not medical advice."** for service, booking and
> FAQ replies. The fixed self-harm reply ends with "You can also talk to a
> doctor or a mental-health professional." instead. Emergency messages get a fixed urgent reply (no LLM) and
> special headers.

---

## Authentication

Authenticated endpoints expect `Authorization: Bearer <JWT>` (HS256, with
`sub` = user id and `role` = `patient` or `admin`).

| How | Endpoint / command | Notes |
|:---|:---|:---|
| Email + password | `POST /api/auth/login` with `{"email", "password"}` | Every failure is the same 401. |
| Demo session | `POST /api/auth/demo` (no body) | Only when `DEMO_LOGIN_ENABLED=true` (otherwise 404). Creates a throwaway patient, 15-minute token. Rate limited per client IP. |
| Local dev | `python -m app.scripts.dev_token --email you@example.com` | Prints a token for an existing user. Refuses to run in production. |

Users are created with `python -m app.scripts.create_user --email ... --role patient|admin`.

Token response:

```json
{ "access_token": "eyJ...", "token_type": "bearer", "expires_in": 900 }
```

The chat request's `userId` must equal the token's `sub`, or the API
returns `404`.

---

## Rate limits

`POST /api/ai/chat` is limited per user (`RATE_LIMIT_REQUESTS_PER_MINUTE`,
default 10) and per client IP (`CHAT_IP_RATE_PER_MINUTE`, default 20).
Successful chat responses carry:

| Header | Meaning |
|--------|---------|
| `RateLimit-Limit` | Requests allowed per window (per user) |
| `RateLimit-Remaining` | Requests left in the current window |
| `RateLimit-Reset` | Seconds until the window clears |

On breach the API returns `429` with `Retry-After`:

```json
{ "detail": "Rate limit exceeded. Try again in 42 seconds.", "correlation_id": "..." }
```

Demo sessions are limited per client IP (`DEMO_RATE_LIMIT_PER_HOUR`, default 20).

---

## Correlation IDs

Every response has an `X-Correlation-ID` header. You may send your own
(1–64 letters, digits or hyphens); anything else is replaced with a new id.
Error bodies include the same value as `correlation_id`.

---

## Endpoints

| Method | Path | Auth |
|:---|:---|:---|
| POST | `/api/ai/chat` | patient or admin |
| GET | `/api/ai/chat/history` | patient or admin |
| GET | `/api/ai/doctors?specialty=...` | patient or admin |
| GET | `/api/ai/doctors/{doctor_id}/availability` | patient or admin |
| POST | `/api/ai/rag/seed` | admin |
| GET | `/api/ai/rag/search?q=...` | admin |
| GET | `/api/ai/monitoring/metrics` | admin |
| GET | `/health`, `/api/ai/health` | none |
| GET | `/metrics` | see below |
| GET | `/` | none (demo page) |

A valid token with the wrong role gets `403`.

### POST `/api/ai/chat`

```bash
curl -X POST "$BASE/api/ai/chat" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "I have a headache and fever for 3 days", "userId": 1}'
```

Response fields:

| Field | Type | Notes |
|:---|:---|:---|
| `reply` | string | Plain text, ends with the disclaimer. |
| `suggestedSpecialist` | string or null | |
| `severity` | string | `mild`, `moderate`, `severe` or `unknown`. |
| `shouldSeeDoctor` | bool | False for booking, service and FAQ replies; true for health replies (and always true for emergencies). |
| `intent` | string | `symptom`, `booking`, `medication`, `general` or `emergency`. |
| `confidence` | number | Intent confidence, 0–1 (0 when the classifier gave no usable label). |
| `doctor_suggestion` | object or null | `specialist`, `doctor_name`, `experience`, `availability` (list of `{id, doctor_id, day_of_week, start_time, end_time, is_booked}`, `day_of_week` 0 = Monday), `doctor_id`, `requested_specialist` (set when no doctor of the mapped specialty was available and a General Physician is suggested instead). |
| `fallback_used` | bool | `true` when the reply model was unavailable, timed out or failed: the reply is then built from the matched FAQ answer and doctor data, or is a fixed message. |

**Emergencies.** A red-flag phrase (English keyword list) or an
`emergency` intent from the classifier returns a fixed urgent reply with
regional emergency numbers (Pakistan, UAE, UK). Self-harm wording gets a
separate reply that points to the local emergency number or a crisis line,
with no phone numbers. Headers:

```
X-Hoku-Emergency: true
X-Hoku-Emergency-Severity: severe   (or moderate; same value as "severity")
```

**Medicine doses.** A reply that states a dose, amount or dosing interval
is replaced with a fixed message asking the patient to check with a doctor
or pharmacist.

### GET `/api/ai/chat/history`

Query params: `limit` (1–100, default 20), `skip` (≥ 0, default 0).
Returns the authenticated user's turns, oldest first:

```json
{
  "user_id": 1,
  "messages": [
    { "role": "human", "content": "What services does Hoku offer?", "timestamp": "..." },
    { "role": "ai", "content": "... Please consult a doctor for proper diagnosis.", "timestamp": "..." }
  ]
}
```

### GET `/health` and `/api/ai/health`

Always HTTP 200, no token, not rate limited:

```json
{ "status": "ok", "service": "Hoku AI Chatbot", "rag_ready": true }
```

`status` is `degraded` (and `rag_ready` false) when the start-up warm-up
found no FAQ content.

### GET `/api/ai/doctors` and `/api/ai/doctors/{doctor_id}/availability`

Available doctors for a specialty, most experienced first, and a doctor's
weekly slots. Empty lists when nothing matches.

### POST `/api/ai/rag/seed` (admin)

Seeds the FAQ knowledge base. Idempotent: FAQs already stored (by content
hash) are skipped.

```json
{ "status": "ok", "documents_added": 0, "documents_skipped": 20, "collection": "hoku_health_faqs" }
```

### GET `/api/ai/rag/search?q=...` (admin)

Raw similarity search, for checking `RAG_SIMILARITY_THRESHOLD`. Returns the
query and the top matches with `question`, `answer`, `category` and `score`.

### GET `/api/ai/monitoring/metrics` (admin)

Human-readable safety and performance counters (`{"status": "ok", "metrics": {...}}`),
in memory and reset on restart.

### GET `/metrics`

Prometheus text format.

- `METRICS_ENABLED=false`: 404.
- `METRICS_TOKEN` set: requires `Authorization: Bearer <METRICS_TOKEN>` (401 otherwise).
- `METRICS_TOKEN` unset: open only when `ENVIRONMENT` is `test` or `development`; 404 elsewhere.

---

## Error codes

| Status | Meaning | Example `detail` |
|--------|---------|------------------|
| `400` | Message empty after trimming whitespace | `Message cannot be empty or exceeds maximum length.` |
| `401` | Missing, invalid or expired token | `Not authenticated` / `Could not validate credentials` |
| `403` | Valid token, role not allowed | `Insufficient permissions` |
| `404` | `userId` ≠ authenticated user, unknown user, or demo login disabled | `User not found or access denied` |
| `422` | Body failed validation (e.g. message over 1000 characters) | *(Pydantic error list)* |
| `429` | Rate limit exceeded | `Rate limit exceeded. Try again in N seconds.` |
| `500` | Unexpected server error | `An unexpected error occurred while processing your request.` |

Error bodies include `correlation_id`. On `/api/auth/*`, validation errors
never echo submitted values.

---

## Notes for clients

- Read `X-Hoku-Emergency` on every chat response and switch to the
  emergency UI when it is `true`; don't parse the reply text.
- Render `reply` as plain text (it is not HTML-escaped).
- Back off using `Retry-After` on 429.
- `doctor_suggestion` is `null` for booking, medication and emergency
  replies; render it only when present.
