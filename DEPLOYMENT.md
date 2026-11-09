# Hoku Health Care AI chatbot: Deployment Guide

This guide covers local runs (Docker, docker compose) and a public demo on **Google Cloud Run** with a **Neon** PostgreSQL database.
Numbers marked *measured* come from local runs on Windows 11 with Docker Desktop (2026-11-09). Anything marked *not measured* has not been checked on Cloud Run or Neon yet.

---

## 1. Docker

### 1.1 Build and run

```bash
docker build -t hoku-health-ai:phase3 .
```

The image:
- installs **CPU-only torch** (2.15.0+cpu) from the PyTorch CPU index, then the runtime requirements only (`requirements-dev.txt` is never installed);
- runs as a **non-root user (uid 1000)**; the app code is owned by root and read-only for that user;
- contains the embedding model (`sentence-transformers/all-MiniLM-L6-v2`) and the tiktoken encoding, downloaded in a layer **before** the app code is copied. At runtime Hugging Face is offline (`HF_HUB_OFFLINE=1`);
- listens on **`$PORT` (default 8080)** with one uvicorn worker and `--no-proxy-headers`;
- has **no secret inside**: `SECRET_KEY` (32+ characters) must be passed at runtime;
- runs `init_db.py` before uvicorn when `DB_AUTO_MIGRATE=true` (the default): Alembic migrations, then idempotent doctor and FAQ seeding. A failure stops the container.

```powershell
# PowerShell: generate a secret into an env var without printing it, then pass it by name
$env:SECRET_KEY = python -c "import secrets; print(secrets.token_urlsafe(48), end='')"
docker run --rm -p 8080:8080 -e SECRET_KEY -e DEMO_LOGIN_ENABLED=true hoku-health-ai:phase3
```

Then open http://localhost:8080/ (demo page) or http://localhost:8080/docs.

Image defaults: `ENVIRONMENT=production`, `DEBUG=false`, `LOG_FORMAT=json`, `DATABASE_URL=sqlite:////tmp/hoku.db`, `DB_AUTO_MIGRATE=true`, `DEMO_LOGIN_ENABLED=false`.

Without `SECRET_KEY` the container exits at start-up with `SECRET_KEY must be at least 32 characters outside the test environment` (the value is never printed). Without `GROQ_API_KEY` every reply is built from the matched FAQ answer and doctor data (`fallback_used: true`).

### 1.2 docker compose (PostgreSQL + pgvector)

```bash
cp .env.example .env     # set SECRET_KEY; GROQ_API_KEY is optional
docker compose up -d --build
docker compose logs -f app
docker compose down      # add -v to delete the database volume
```

Compose starts `pgvector/pgvector:pg16`, a one-shot `migrate` service (`python init_db.py`), then the app on http://localhost:8000 (container port 8080) with demo login on.

---

## 2. Google Cloud Run (public demo)

Visitors open the service URL and get the demo page at `/`. The page calls `POST /api/auth/demo`, which creates a throwaway patient account (deleted after 24 hours) and returns a 15-minute token. The page keeps the token in memory only and calls `POST /api/ai/chat` on the same origin.

### 2.1 Service settings

| Setting | Value | Why |
|:---|:---|:---|
| Maximum instances | **1** | Rate limiters, response cache and metrics are in memory, and migrations run at start-up (`DB_AUTO_MIGRATE=true`). More instances would mean separate limits and racing migrations. |
| Minimum instances | **0** | Scales to zero when idle. The first visitor after an idle period waits for a cold start. |
| Container port | 8080 (Cloud Run default) | The app reads `$PORT`. Do not set `PORT` yourself. |
| Memory | **2 GiB** | Measured locally: 541.7 MiB after 4 chats (`docker stats`, SQLite in `/tmp`, no Groq key, so no live LLM call); 436.8 MiB with Postgres in compose. Rule used: the smaller of 1 / 2 GiB that is at least twice the highest measurement (2 × 541.7 MiB > 1 GiB). Cloud Run's `/tmp` is in memory and counts against this limit. Not measured on Cloud Run. |
| CPU | 1, with startup CPU boost (`--cpu-boost`) | One uvicorn worker. The boost shortens model loading during a cold start. |
| Authentication | Allow unauthenticated invocations | Visitors must be able to open the page. The chat API itself needs a bearer token. |
| Request timeout | Default (300 s) | A chat request has its own 4 s time budget (3.5 s Groq timeout). |

### 2.2 Secrets (Secret Manager)

| Secret | Value | Notes |
|:---|:---|:---|
| `GROQ_API_KEY` | Your Groq key | Without it every reply is a fallback (`fallback_used: true`). |
| `SECRET_KEY` | Random, at least 32 characters | Signs the JWTs. Generate it; never paste it into chat, code or tickets. Changing it invalidates all tokens (the demo page simply starts a new session). |
| `DATABASE_URL` | Neon connection string with `sslmode=require` | See 2.5. Without it the database is SQLite in `/tmp`, lost on every restart. |
| `METRICS_TOKEN` | Random string | Bearer token for `GET /metrics`. Without it `/metrics` returns 404 in production. |

Each secret is exposed to the service as an environment variable of the same name. The runtime service account needs the **Secret Manager Secret Accessor** role on each secret (commands in 2.8).

### 2.3 Environment variables

| Name | Value | Notes |
|:---|:---|:---|
| `ENVIRONMENT` | `production` | Image default. |
| `DEMO_LOGIN_ENABLED` | `true` | **Required for the demo page.** The image default is `false` (the endpoint then answers 404 and the page says demo login is disabled). |
| `TRUST_PROXY` | `true` | Take the client IP from `X-Forwarded-For`. |
| `TRUST_PROXY_HOPS` | `1` | The client IP is the right-most `X-Forwarded-For` entry, which Google's front end appends. Entries to its left are client-supplied and ignored. Use `2` if an external HTTPS load balancer sits in front of the service (see 2.10). |
| `DB_AUTO_MIGRATE` | `true` | Image default. Safe only with maximum instances 1. |
| `DB_POOL_SIZE` / `DB_MAX_OVERFLOW` | `2` / `3` | Small pool for one instance on a free Neon project. |
| `LOG_LEVEL` | optional, default `INFO` | Logs contain no message text, symptoms, tokens or keys. |
| `RATE_LIMIT_REQUESTS_PER_MINUTE` | optional, default `10` | Chats per user per minute (each demo session is a new user). |
| `CHAT_IP_RATE_PER_MINUTE` | optional, default `20` | Chats per client IP per minute, across all sessions. |
| `DEMO_RATE_LIMIT_PER_HOUR` | optional, default `20` | Demo sessions per client IP per hour. |
| `RAG_SIMILARITY_THRESHOLD` | optional, default `0.35` | Minimum FAQ similarity for grounding (see 2.11). |
| `GROQ_FAST_MODEL` / `GROQ_MAIN_MODEL` | optional, defaults `llama-3.1-8b-instant` / `llama-3.3-70b-versatile` | Change them if Groq stops serving a model. |

### 2.4 Health, readiness and cold start

- **`GET /health`** (same payload as `GET /api/ai/health`) needs no token and is not rate limited. It returns `{"status": "ok" | "degraded", "service": "Hoku AI Chatbot", "rag_ready": true | false}` with HTTP 200.
  - `rag_ready: true` means a warm-up query at start-up found FAQ content: the embedding model is loaded, FAQ rows exist and, on PostgreSQL, the pgvector query works.
  - `status: "degraded"` means it did not; the app still answers, and FAQ questions take the no-match path.
- The warm-up runs in the FastAPI lifespan, before uvicorn opens the port, so Cloud Run's default TCP startup probe already waits for a warm app.
- **Cold start, measured locally:** 19.9 s from `docker run` to the first `/health` with `rag_ready: true` (Windows PC, Docker Desktop, image already on disk, `--network none`, fresh SQLite, so this includes migrations and embedding the 20 FAQs). With an already seeded database the seeding step skips existing rows.
- **First chats after start-up, measured locally (offline, no Groq key):** server time 25 ms for the emergency example, about 34 ms for the FAQ and booking examples (FAQ answer fallback), 450 ms for the symptom example (doctor lookup).
- On Cloud Run a cold start also includes starting the instance and pulling the image (565 MB, `docker image inspect` size): **not measured yet**.

### 2.5 Neon PostgreSQL with pgvector

1. Create a free project at neon.tech in a region close to your Cloud Run region.
2. In **Connection details** copy the connection string for the default database. Use the **direct** connection (not "pooled"). It looks like `postgresql://<user>:<password>@<host>/<db>?sslmode=require` (it may also contain `channel_binding=require`; keep it). Keep the string out of chat, tickets and the repository; it goes straight into Secret Manager (2.8 step 4).
3. **pgvector.** Migration 003 runs `CREATE EXTENSION IF NOT EXISTS vector`. Neon documents pgvector as supported, and the project owner role it creates should be allowed to enable it, but **this has not been tested on Neon**. If the first start fails with `Could not enable the pgvector extension`, open the Neon **SQL Editor** for that database and run once:

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

   then redeploy (or restart the revision). You can check the result with `SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';`.
   - Measured locally: with a non-superuser role, migration 003 stops with that message (the transaction is not left half-applied); after a superuser created the extension, the same role migrated successfully (`tests/pgvector/test_migration_privileges.py`).
4. The app passes the URL to psycopg2 unchanged, so `sslmode=require` is used as is.
5. Neon's free tier suspends an idle database; the first query after that waits for it to wake up (**not measured**). `pool_pre_ping` replaces connections the suspend closed.

### 2.6 What resets, and what persists

- **With `DATABASE_URL` (Neon):** users, chat history, doctors and FAQ embeddings survive restarts. Demo users, their chat history and their safety-log events are deleted after 24 hours (on the next demo login). Emergency events store only a placeholder with the message length, never the message text.
- **Without `DATABASE_URL`:** everything is SQLite in `/tmp/hoku.db`. Cloud Run's `/tmp` is in memory, belongs to one instance and counts against the memory limit. It is **lost on every restart, new revision or scale to zero**; the start-up seeding recreates doctors and FAQs.
- **Rate-limit counters and the response cache** are in memory and reset with the instance.
- The FAQs and doctors are sample data.

### 2.7 Rate limits

| Limit | Default | Key |
|:---|:---|:---|
| Chat, per user | 10 / minute | Verified token subject (each demo session is a new user) |
| Chat, per client IP | 20 / minute | Client IP (proxy-aware) |
| Demo sessions | 20 / hour | Client IP (proxy-aware) |

`/`, `/health`, `/api/ai/health`, `/docs` and `/openapi.json` are not rate limited. A 429 response carries `Retry-After`.

**Shared Wi-Fi / events.** Everyone behind one public IP (an office, a university network, conference Wi-Fi, a mobile carrier's NAT) shares the per-IP limits: together they get 20 chats a minute and 20 new demo sessions an hour. For a live demo to a group, raise the caps temporarily (this creates a new revision, which also resets the in-memory counters):

```bash
gcloud run services update hoku-health-ai --region <REGION> \
  --update-env-vars CHAT_IP_RATE_PER_MINUTE=100,DEMO_RATE_LIMIT_PER_HOUR=100
# afterwards, back to the defaults:
gcloud run services update hoku-health-ai --region <REGION> \
  --remove-env-vars CHAT_IP_RATE_PER_MINUTE,DEMO_RATE_LIMIT_PER_HOUR
```

Groq's free plan has its own request and token limits per minute and per day. When Groq rejects a call, the reply is a fallback (`fallback_used: true`), not an error.

### 2.8 Step by step (commands for the owner)

Replace `<PROJECT_ID>`, `<PROJECT_NUMBER>` and `<REGION>` (for example `europe-west1`). Nothing below is run by the build tooling.

**1. Neon.** Create the project and copy the direct connection string (2.5).

**2. Project and APIs.**

```bash
gcloud auth login
gcloud config set project <PROJECT_ID>
gcloud services enable compute.googleapis.com run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
```

**3. Artifact Registry repository.**

```bash
gcloud artifacts repositories create hoku --repository-format=docker --location=<REGION>
```

**4. Secrets.** Simplest: **Console > Security > Secret Manager > Create secret**, name it, paste the value, create; repeat for `GROQ_API_KEY`, `SECRET_KEY`, `DATABASE_URL`, `METRICS_TOKEN`.

Or with gcloud, from a file you delete straight away (PowerShell; `-NoNewline` matters, a trailing newline would become part of the secret):

```powershell
$f = "$env:TEMP\hoku-secret.txt"

# SECRET_KEY and METRICS_TOKEN: generated, never printed
python -c "import secrets; print(secrets.token_urlsafe(48), end='')" | Set-Content -NoNewline -Encoding ascii $f
gcloud secrets create SECRET_KEY --replication-policy=automatic --data-file=$f
python -c "import secrets; print(secrets.token_urlsafe(32), end='')" | Set-Content -NoNewline -Encoding ascii $f
gcloud secrets create METRICS_TOKEN --replication-policy=automatic --data-file=$f

# GROQ_API_KEY: paste the key when prompted (input is hidden)
$k = Read-Host -AsSecureString "Groq key"; [Net.NetworkCredential]::new("", $k).Password | Set-Content -NoNewline -Encoding ascii $f
gcloud secrets create GROQ_API_KEY --replication-policy=automatic --data-file=$f

# DATABASE_URL: paste the Neon connection string when prompted
$k = Read-Host -AsSecureString "Neon URL"; [Net.NetworkCredential]::new("", $k).Password | Set-Content -NoNewline -Encoding ascii $f
gcloud secrets create DATABASE_URL --replication-policy=automatic --data-file=$f

Remove-Item $f
```

**5. Let the runtime service account read the secrets.** Cloud Run uses the Compute Engine default service account unless you choose another one:

```bash
gcloud iam service-accounts list --filter="email~-compute@developer.gserviceaccount.com"
gcloud projects describe <PROJECT_ID> --format="value(projectNumber)"
```

Then, for each of `GROQ_API_KEY`, `SECRET_KEY`, `DATABASE_URL` and `METRICS_TOKEN`:

```bash
gcloud secrets add-iam-policy-binding <SECRET_NAME> \
  --member=serviceAccount:<PROJECT_NUMBER>-compute@developer.gserviceaccount.com \
  --role=roles/secretmanager.secretAccessor
```

**6. Push the image built locally** from the commit you want to deploy (an image built before a code change does not contain it):

```bash
docker build -t hoku-health-ai:phase3 .
gcloud auth configure-docker <REGION>-docker.pkg.dev
docker tag hoku-health-ai:phase3 <REGION>-docker.pkg.dev/<PROJECT_ID>/hoku/hoku-health-ai:v1
docker push <REGION>-docker.pkg.dev/<PROJECT_ID>/hoku/hoku-health-ai:v1
```

**7. Deploy.**

```bash
gcloud run deploy hoku-health-ai \
  --image <REGION>-docker.pkg.dev/<PROJECT_ID>/hoku/hoku-health-ai:v1 \
  --region <REGION> \
  --allow-unauthenticated \
  --max-instances 1 --min-instances 0 \
  --cpu 1 --memory 2Gi --cpu-boost \
  --set-secrets GROQ_API_KEY=GROQ_API_KEY:latest,SECRET_KEY=SECRET_KEY:latest,DATABASE_URL=DATABASE_URL:latest,METRICS_TOKEN=METRICS_TOKEN:latest \
  --set-env-vars ENVIRONMENT=production,DEMO_LOGIN_ENABLED=true,TRUST_PROXY=true,TRUST_PROXY_HOPS=1,DB_AUTO_MIGRATE=true,DB_POOL_SIZE=2,DB_MAX_OVERFLOW=3
```

PowerShell: put the command on one line, or end each line with a backtick instead of `\`.

**8. Smoke check, then benchmark** (2.9). **9. Check the client IP** (2.10).

### 2.9 Post-deploy checks

`scripts/smoke_check.py` (Python stdlib only, no API key) checks a deployed URL and writes a JSON summary to `--out`, which must be outside the repository:

```bash
python scripts/smoke_check.py --url https://<service-url> --out ../_facts/hoku_smoke.json
```

- It checks `/health` (`status=ok`, `rag_ready=true`), the demo page (marker, banner, hash-based CSP), a demo login, a chat without a token (401), and the 4 demo-page examples: the emergency example must return `intent=emergency` with `X-Hoku-Emergency: true`; the others must return the expected intent and `fallback_used=false`, and the symptom example a doctor suggestion.
- Two **WARN-only scope probes** ("What are the opening hours of Lahore's Emporium mall?", "Does Amazon deliver to Lahore?"). Both pass the retrieval threshold (see 2.11), so the reply should decline under the prompt's scope rule. The check is a text heuristic; read the printed reply yourself.
- Unless `--skip-rate-limit`: the per-user chat limit and the per-IP chat cap with spoofed `X-Forwarded-For` entries. They send empty messages, which cost no LLM call; each waits 61 s first.
- `--check-demo-limit` also checks the demo-session limit. It uses up the hour's demo sessions for your IP.
- About 6 live LLM calls, paced 6.5 s apart; about 2.5 minutes with the rate-limit checks.
- Measured locally against the container without a Groq key: 7 PASS, 2 WARN, 3 FAIL. The FAILs were only `fallback_used: true` on the three LLM examples (expected without a key); the WARNs were the scope probes, answered with the matched FAQ by the fallback; both rate-limit checks got their 429. Against docker compose (Postgres + pgvector, `--skip-rate-limit`): the same results.

`scripts/benchmark.py` (stdlib only) starts one demo session, sends one warm-up request (not counted), then N sequential chats over the 3 non-emergency examples, and prints nearest-rank p50 / p95 / max for client time and for the server's `X-Response-Time-Sec`:

```bash
python scripts/benchmark.py --url https://<service-url> -n 20 --out ../_facts/hoku_benchmark.json
```

- Requests start 6.5 s apart (under 10 per minute per user). Wait 60 s after the smoke check before starting.
- It exits non-zero unless every measured request returned 200, and reports how many replies were fallbacks.
- **Real Groq latency has not been measured yet**; run this after deploying.

`scripts/eval_retrieval.py` (no LLM, no service needed) measures FAQ retrieval on the dev and held-out sets in `data/eval/` (2.11):

```bash
python scripts/eval_retrieval.py --out ../_facts/hoku_retrieval_eval.json
```

### 2.10 Checking the client IP the rate limiter sees

**Not verified on Cloud Run yet:** whether the right-most `X-Forwarded-For` entry is the visitor's IP. After the smoke check has triggered a 429 on the per-IP cap, the service log has a line `Rate limit exceeded for client IP <ip>` (Cloud Run > the service > Logs). Compare `<ip>` with your own public IP:
- It matches: the setting is correct.
- It is a Google address: all visitors share one limit. Spoofed headers still cannot bypass it, but set `TRUST_PROXY_HOPS=2`.

The IP appears in the log only on a per-IP 429.

### 2.11 Retrieval threshold and known leaks

FAQ content grounds a reply only when the top FAQ's cosine similarity is at least `RAG_SIMILARITY_THRESHOLD` (**0.35**, unchanged). Measured on a dev set (32 in-scope + 8 off-topic) and a held-out set written before any measurement (20 in-scope + 12 off-topic, 6 near-topic); same numbers on SQLite and on pgvector:

| Set | hit@1 | hit@3 | Gated hit@1 | Off-topic rejected |
|:---|:---|:---|:---|:---|
| dev | 0.938 | 1.000 | 0.875 | 8/8 |
| held-out | 0.900 | 0.900 | 0.900 | 10/12 |

**Known leaks:** "What are the opening hours of Lahore's Emporium mall?" (0.468) and "Does Amazon deliver to Lahore?" (0.363) pass the threshold. The reply prompt's scope rule asks the model to decline such questions; whether the live model does is **not measured** until the smoke check's scope probes run against the deployed service. Without a Groq key the fallback reply would show the matched FAQ answer. Full numbers: `data/eval/README.md`.

### 2.12 After changing code

The image bakes in the code and the model. After any code change, rebuild, push a new tag (`:v2`) and deploy that tag. The model is downloaded in a layer before the app code is copied, so a code-only rebuild reuses it (one took 11 s locally). In one local build with the torch layer already cached, the dependency install took 216 s and the model download 89 s. A change to `requirements.txt` or to the Dockerfile above that layer reruns the dependency install and the model download.

---

## 3. Troubleshooting

| Symptom | Likely cause |
|:---|:---|
| Container exits with `SECRET_KEY must be at least 32 characters` | The secret is missing, not mapped to the service, or too short. |
| Container exits with `Could not enable the pgvector extension` | The database role may not create the extension: run `CREATE EXTENSION IF NOT EXISTS vector;` once in the Neon SQL Editor (2.5). |
| Demo page says "Demo login is disabled on this server." | `DEMO_LOGIN_ENABLED` is not `true`. |
| `fallback_used: true` in every reply | `GROQ_API_KEY` missing or rejected, Groq rate limit reached, or a model no longer served. Check the logs for the Groq error type. |
| `/health` shows `"status": "degraded"` | The warm-up found no FAQ content; check the start-up log for `RAG warm-up` lines. |
| Many 429s for a group of people | They share one public IP: see "Shared Wi-Fi / events" in 2.7. |

---

## 4. Known limits

- One instance only (in-memory limits, cache and metrics).
- Direct dependencies are pinned in `requirements.txt`, and every transitive version is locked in `constraints.txt` (Dockerfile and CI install with `-c constraints.txt`). Regenerate it from `pip freeze` of a fresh build whenever `requirements.txt` changes. Not checked with a vulnerability scanner.
- Real Groq latency, Cloud Run cold start, Neon wake-up time and the live model's scope behaviour are not measured yet.
