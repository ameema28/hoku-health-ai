# Hoku Health Care — AI Chatbot API Reference

Base URL (local): `http://localhost:8000`
Base URL (prod): `https://hoku-health-backend.onrender.com`

All examples use `curl`. Interactive docs live at `/docs` (Swagger) and
`/redoc`.

> Every chatbot reply ends with **"Please consult a doctor for proper
> diagnosis."** Emergency messages are detected before the LLM and return
> an urgent response with special headers.

---

## Authentication

Authenticated endpoints expect a **Bearer JWT** in the `Authorization`
header. For local testing, mint one:

```bash
python token_gen.py
# → Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

Then pass it:

```bash
export TOKEN="Bearer eyJhbGciOiJIUzI1NiI..."
curl -H "Authorization: $TOKEN" http://localhost:8000/api/ai/chat/history
```

The token's `sub`/`id` claim is the user id; it must match the `userId`
in the chat request body, or the API returns `404`.

---

## Rate Limits

`POST /api/ai/chat` is limited to **5 requests per minute per user**
(configurable via `RATE_LIMIT_REQUESTS_PER_MINUTE`). Every response
carries:

| Header | Meaning |
|--------|---------|
| `RateLimit-Limit` | Requests allowed per window |
| `RateLimit-Remaining` | Requests left in the current window |
| `RateLimit-Reset` | Seconds until the window clears |
| `Retry-After` | (on 429 only) seconds to wait before retrying |

On breach the API returns `429 Too Many Requests`:

```json
{ "detail": "Rate limit exceeded. Try again in 42 seconds.", "correlation_id": "9f2c..." }
```

---

## Correlation IDs

Every response includes an `X-Correlation-ID` header. Send your own via
the request header of the same name to trace a call end-to-end through
the logs; otherwise the server generates one.

---

## Endpoints

### POST `/api/ai/chat` — AI Health Chatbot

**Auth:** required · **Rate limit:** 5/min

```bash
curl -X POST http://localhost:8000/api/ai/chat \
  -H "Authorization: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "I have a headache and fever for 3 days", "userId": 1}'
```

**200 OK**

```json
{
  "reply": "A persistent fever with a headache is worth checking. I can't diagnose the cause, but a General Physician can evaluate you. Please consult a doctor for proper diagnosis.",
  "suggestedSpecialist": "General Physician",
  "severity": "moderate",
  "shouldSeeDoctor": true,
  "intent": "symptom",
  "confidence": 0.95,
  "doctor_suggestion": {
    "specialist": "General Physician",
    "doctors": [
      { "id": 3, "name": "Dr. Sara Ahmed", "specialty": "General Physician", "experience_years": 12 }
    ]
  }
}
```

**Emergency example**

```bash
curl -i -X POST http://localhost:8000/api/ai/chat \
  -H "Authorization: $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "I have severe chest pain and can'\''t breathe", "userId": 1}'
```

Response headers include:

```
X-Hoku-Emergency: true
X-Hoku-Emergency-Severity: severe
```

Body: `severity` is `severe`, `shouldSeeDoctor` is `true`, and the reply
lists regional emergency numbers (Pakistan 1122, UAE 998/999, UK 999).

---

### GET `/api/ai/chat/history` — Chat History

**Auth:** required

```bash
curl -H "Authorization: $TOKEN" \
  "http://localhost:8000/api/ai/chat/history?limit=20&skip=0"
```

```json
{
  "user_id": 1,
  "messages": [
    { "role": "human", "content": "What services does Hoku offer?", "timestamp": "2026-07-24T18:04:11" },
    { "role": "ai", "content": "Hoku offers home health, palliative, and hospice care. Please consult a doctor for proper diagnosis.", "timestamp": "2026-07-24T18:04:12" }
  ]
}
```

Query params: `limit` (1–100, default 20), `skip` (≥0, default 0).

---

### GET `/api/ai/health` — Health Check

**Auth:** none

```bash
curl http://localhost:8000/api/ai/health
# {"status":"ok","service":"Hoku AI Chatbot"}
```

---

### GET `/api/ai/doctors` — List Doctors by Specialty

**Auth:** required

```bash
curl -H "Authorization: $TOKEN" \
  "http://localhost:8000/api/ai/doctors?specialty=Cardiologist"
```

Returns doctors with `is_available=true`, ordered by `experience_years`
descending. Empty list if none match.

---

### GET `/api/ai/doctors/{doctor_id}/availability` — Doctor Availability

**Auth:** required

```bash
curl -H "Authorization: $TOKEN" \
  http://localhost:8000/api/ai/doctors/3/availability
```

---

### POST `/api/ai/rag/seed` — Seed FAQ Vector Store (admin)

**Auth:** none (restrict at the ingress in production)

```bash
curl -X POST http://localhost:8000/api/ai/rag/seed
# {"status":"ok","documents_added":20,"collection":"hoku_health_faqs"}
```

Not idempotent — each call adds another copy of the FAQ set.

---

### GET `/api/ai/rag/search` — Debug Similarity Search (admin)

```bash
curl "http://localhost:8000/api/ai/rag/search?q=palliative%20care"
```

Returns scored FAQ matches for tuning `RAG_SIMILARITY_THRESHOLD`.

---

### GET `/api/ai/monitoring/metrics` — Safety & Performance Summary

**Auth:** required

```bash
curl -H "Authorization: $TOKEN" \
  http://localhost:8000/api/ai/monitoring/metrics
```

```json
{
  "status": "ok",
  "metrics": {
    "requests_total": 1240,
    "emergency_detections_total": 7,
    "safety_violations_total": 2,
    "nfr02_breaches_total": 1,
    "average_latency_ms": 1830.4,
    "p99_latency_ms": 2450.0,
    "breach_rate_percent": 0.08,
    "cache_hit_ratio": 0.34,
    "prometheus_enabled": true
  }
}
```

---

### GET `/metrics` — Prometheus Exposition

**Auth:** none (scrape from a private network)

```bash
curl http://localhost:8000/metrics
```

```
# HELP hoku_chatbot_requests_total Total chat requests processed...
# TYPE hoku_chatbot_requests_total counter
hoku_chatbot_requests_total{intent="general",emergency_flag="false"} 812.0
hoku_chatbot_response_time_seconds_bucket{endpoint="/api/ai/chat",le="4.0"} 1239.0
hoku_chatbot_cache_hit_ratio 0.34
...
```

---

## Error Codes

| Status | Meaning | Example `detail` |
|--------|---------|------------------|
| `400` | Empty message or over 1000 chars | `Message cannot be empty or exceeds maximum length.` |
| `401` | Missing/invalid Bearer token | `Not authenticated` |
| `404` | `userId` ≠ authenticated user, or unknown user | `User not found or access denied` |
| `422` | Request body failed validation | *(Pydantic error list)* |
| `429` | Rate limit exceeded | `Rate limit exceeded. Try again in N seconds.` |
| `500` | Unexpected server error | `An unexpected error occurred... Please consult a doctor for proper diagnosis.` |

Error bodies include a `correlation_id` you can quote in a support
request; it matches the `X-Correlation-ID` response header and the server
logs.

---

## Notes for the Frontend Team

- Read `X-Hoku-Emergency` on every chat response and route to the
  emergency UI when it is `true` — don't rely on parsing the reply text.
- Surface `RateLimit-Remaining` so patients aren't surprised by a 429;
  back off using `Retry-After`.
- Send a stable `X-Correlation-ID` per user action to make support
  triage trivial.
- `doctor_suggestion` is `null` for booking, medication, and emergency
  intents by design — render it only when present.