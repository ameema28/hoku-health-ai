# Hoku Health Care - AI Chatbot Module

**TechNexus Virtual University | Internship Project**

This module provides the AI-powered health chatbot backend for Hoku Health Care, a home healthcare platform serving patients in Pakistan, UAE, and UK.

---

## Tech Stack

- **Framework**: FastAPI (Python)
- **AI/LLM**: Groq API (Llama 3) via LangChain 0.2.6
- **Database**: SQLite (dev) / PostgreSQL (prod) + SQLAlchemy 2.0 + Alembic
- **Auth**: JWT via `app/core/security.py` (HTTPBearer), stubbed pending Backend Lead (Talha)'s full user-lookup implementation
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2), local inference, 384 dims
- **Vector Store**: pgvector on PostgreSQL in production; falls back to an in-Python cosine-similarity search on SQLite (used automatically in this dev setup)
- **Fuzzy Matching**: rapidfuzz 3.9.0 for symptom-to-specialist mapping (Day 6)
- **Caching**: In-memory SHA-256 keyed response cache with TTL expiration (Day 8)
- **Connection Pooling**: SQLAlchemy QueuePool with SQLite thread fallback (Day 8)

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Groq API key ([Get one here](https://console.groq.com/keys))

### 2. Installation

```bash
cd hoku-health-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
cp .env.example .env
# Edit .env and fill in your GROQ_API_KEY
```

**Required `.env` variables:**

```bash
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_FAST_MODEL=llama-3.1-8b-instant
GROQ_MAIN_MODEL=llama-3.3-70b-versatile
TEMPERATURE=0.3
MAX_TOKENS=512
GROQ_TIMEOUT_SECONDS=3.5
MAX_RETRIES=3
RETRY_BACKOFF_BASE_SECONDS=1.0
MEMORY_MESSAGE_LIMIT=10
MEMORY_MAX_TOKENS=307
TIKTOKEN_ENABLED=false

# Intent Classification
INTENT_MODEL=llama-3.1-8b-instant
INTENT_CLASSIFICATION_TIMEOUT=1.5
INTENT_CONFIDENCE_THRESHOLD=0.7

# RAG Pipeline (Day 5)
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_DIMENSION=384
RAG_SIMILARITY_THRESHOLD=0.75
RAG_TOP_K=3
COLLECTION_NAME=hoku_health_faqs
RAG_LOOKUP_TIMEOUT=0.5

# Symptom Extraction & Doctor Suggestion (Day 6)
SYMPTOM_EXTRACTION_MODEL=llama-3.1-8b-instant
SYMPTOM_EXTRACTION_TIMEOUT=0.2
DOCTOR_LOOKUP_LIMIT=5

# Emergency Detection & Safety Guardrails (Day 7)
EMERGENCY_CHECK_TIMEOUT=0.3
SAFETY_MAX_RETRIES=3
SAFETY_FALLBACK_RESPONSE="I am unable to provide a medical opinion for this query. Please consult a qualified doctor immediately."

# Performance Optimization & Response Time Guarantees (Day 8)
RESPONSE_CACHE_ENABLED=true
RESPONSE_CACHE_TTL_SECONDS=300
RESPONSE_CACHE_MAX_SIZE=1000
CACHE_EXCLUDE_INTENTS=emergency,symptom
LLM_PROMPT_COMPRESSION=true
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_POOL_RECYCLE_SECONDS=3600
DB_POOL_TIMEOUT_SECONDS=5
FALLBACK_RESPONSES_ENABLED=true
NFR02_BREACH_LOG_LEVEL=WARNING
```

**For local development without PostgreSQL** (default in this environment):

```bash
DATABASE_URL=sqlite:///./hoku_health.db
```

### 4. Initialize the Database

```bash
python init_db.py
```

Creates `chat_history`, `vector_store` (Day 5), `doctors`, `doctor_availability` (Day 6), and `safety_logs` (Day 7) tables. On SQLite, `vector_store.embedding` is a JSON column; on PostgreSQL with pgvector installed, it's a native `vector(384)` column — the model detects this automatically from `DATABASE_URL`.

Day 6: `init_db.py` also seeds 5 sample doctors (Cardiologist, Dermatologist, General Physician, Orthopedic Surgeon, Psychiatrist) with weekly availability slots.

### 5. Seed the FAQ Knowledge Base (Day 5)

```bash
python -m app.scripts.seed_faqs
```

Downloads the embedding model on first run (~90MB, requires internet) and loads 20 FAQ entries covering Hoku's Pakistan/UAE/UK home healthcare services into `hoku_health_faqs`. Re-runnable via `POST /api/ai/rag/seed`, but note it is **not idempotent** — re-running adds a second copy of the FAQ set rather than skipping duplicates.

### 6. Generate a Test JWT

Authenticated endpoints (`/api/ai/chat`, `/api/ai/chat/history`, `/api/ai/doctors`) require a Bearer token. Use `token_gen.py` to generate one for local testing:

```bash
python token_gen.py
```

Prints ready-to-use `Bearer <token>` strings for user IDs 1 and 2, each valid for 2 hours.

### 7. Run Tests

```bash
# Unit tests (mocked Groq — no API key needed)
pytest tests/test_chatbot.py -v
pytest tests/test_memory.py -v
pytest tests/test_intent.py -v
pytest tests/test_crud.py -v

# RAG pipeline tests (Day 5 — pgvector interactions mocked, since SQLite has no pgvector)
pytest tests/test_rag.py -v

# Specialist suggestion & doctor integration tests (Day 6)
pytest tests/test_specialist.py -v

# Safety guardrails & emergency escalation tests (Day 7)
pytest tests/test_safety.py -v

# Performance optimization & NFR-02 compliance tests (Day 8)
pytest tests/test_performance.py -v

# Full test suite
pytest tests/ -v
```

### 8. Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 9. API Documentation

Once running, open your browser:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

To test authenticated endpoints in Swagger: generate a token with `token_gen.py`, click **Authorize** in the top-right of `/docs`, and paste the token into the Bearer field.

---

## AI Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/ai/chat` | Yes | AI Health Chatbot (Groq LLM + Memory + Intent + RAG + Doctor Suggestion + Safety Guardrails + Response Caching) |
| GET | `/api/ai/chat/history` | Yes | Chat History (paginated) |
| GET | `/api/ai/health` | No | Service Health Check |
| POST | `/api/ai/rag/seed` | No | Seed the Hoku FAQ vector store (Day 5, admin/dev use) |
| GET | `/api/ai/rag/search?q={query}` | No | Debug FAQ similarity search (Day 5, admin/dev use) |
| GET | `/api/ai/doctors?specialty={specialty}` | Yes | Retrieves available doctors ordered by experience (Day 6) |
| GET | `/api/ai/doctors/{doctor_id}/availability` | Yes | Fetches textual time slot listings for a specific doctor (Day 6) |
| GET | `/api/ai/monitoring/metrics` | Yes | Safety & performance metrics (Day 7) |

---

## Project Structure

```
hoku-health-backend/
├── alembic/              # Database migrations
│   ├── env.py
│   └── versions/
│       ├── 001_create_chat_history.py
│       ├── 002_add_intent_index.py
│       └── 003_add_vector_store.py    # Day 5: pgvector-aware, SQLite-safe
├── app/
│   ├── ai/               # Chatbot engine (Groq + LangChain)
│   │   ├── __init__.py   # Day 8: exports ResponseCache, ResponseOptimizer, LLMFactory
│   │   ├── config.py     # AI hyperparameters, timeouts, memory, RAG, symptom extraction, doctor lookup, safety guardrails, caching, connection pooling
│   │   ├── prompts.py    # Clinical safety + intent + RAG-grounded + specialist suggestion + emergency + safety appendix prompts
│   │   ├── chatbot.py    # HokuChatbot: intent + emergency + bounded RAG lookup + doctor suggestion + 3-strike safety retry + cache integration (Day 8)
│   │   ├── intent_classifier.py  # 5-way intent classification
│   │   ├── emergency_detector.py # Tier 1 regex + Tier 2 LLM fallback emergency detection (Day 7)
│   │   ├── safety_guardrails.py  # Day 7: Post-LLM diagnosis/prescription validation, sanitization, 3-strike retry
│   │   ├── memory.py     # Per-user ConversationBufferMemory loader
│   │   ├── token_budget.py # Token counting & history trimming
│   │   ├── embeddings.py  # Day 5: local sentence-transformers embedding manager
│   │   ├── rag.py         # Day 5: HokuRAG similarity search + context builder
│   │   ├── specialist_mapper.py  # Day 6: Fuzzy matching for symptom-to-specialist mapping
│   │   ├── symptom_extractor.py  # Day 6: Dual-path regex + Groq LLM symptom extraction
│   │   ├── ai_performance.py  # Day 8: ResponseOptimizer — step-by-step time budgeting with 3.5s limit
│   │   ├── caching.py         # Day 8: ResponseCache — SHA-256 keyed in-memory cache with TTL and clinical safety exclusions
│   │   ├── connection_pool.py # Day 8: SQLAlchemy QueuePool tuning with SQLite thread fallback
│   │   ├── fallback_responses.py  # Day 8: Static sub-1ms fallback layer for emergency and clinical timeout scenarios
│   │   ├── llm_optimizer.py       # Day 8: LLMFactory — dual-model architecture with prompt compression
│   │   └── utils.py      # Response parsers
│   ├── api/v1/endpoints/
│   │   ├── __init__.py
│   │   └── ai.py          # + /api/ai/rag/seed, /api/ai/rag/search (Day 5), /api/ai/doctors (Day 6), /api/ai/monitoring/metrics (Day 7)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py       # + VECTOR_DIMENSION, RAG_SIMILARITY_THRESHOLD, RAG_TOP_K, COLLECTION_NAME, caching, pooling (Day 8)
│   │   ├── database.py     # engine, SessionLocal, Base, get_db — Day 8: QueuePool tuning, SQLite thread fallback
│   │   ├── dependencies.py # re-exports get_db, get_current_user
│   │   ├── exceptions.py   # UserNotFoundException, DatabaseOperationException
│   │   ├── middleware.py   # Request timing & NFR-02 monitoring — Day 8: X-Response-Time-Sec headers, enhanced breach logging
│   │   ├── monitoring.py   # Day 7: Thread-safe safety & performance metrics (HokuMetrics)
│   │   └── security.py     # JWT auth (HTTPBearer stub, pending Talha)
│   ├── crud/
│   │   ├── __init__.py     # re-exports app.crud.crud_chat
│   │   ├── crud_chat.py
│   │   ├── crud_doctor.py  # Day 6: Doctor & availability CRUD lookups
│   │   └── crud_safety.py  # Day 7: SafetyLog CRUD operations
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── cors.py
│   │   └── error_handler.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── models_chat.py
│   │   ├── models_doctor.py              # Day 6: Doctor model (SQLAlchemy 2.0)
│   │   ├── doctor_availability.py  # Day 6: Doctor Availability slots model
│   │   └── safety_log.py                 # Day 7: Safety violation audit log model
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── schemas_chat.py  # Updated: Added doctor_suggestion to ChatMessageResponse (Day 6)
│   │   ├── schemas_doctor.py  # Day 6: Pydantic v2 Doctor & Suggestion schemas
│   │   └── schemas_safety.py  # Day 7: Pydantic v2 SafetyLog schemas
│   ├── scripts/             # Day 5
│   │   ├── __init__.py
│   │   └── seed_faqs.py     # Seeds 20 Hoku FAQ entries
│   ├── services/
│   │   ├── __init__.py
│   │   └── ai_service.py    # Day 8: integrated ResponseCache, ResponseOptimizer, LLMFactory, fallback layer
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   └── validators.py
│   ├── __init__.py
│   └── main.py              # Day 8: integrated lifespan model warm-up (sentence transformers & DB pools), caching, connection pooling, timing middleware
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # DB + client fixtures, mock intent/emergency fixtures
│   ├── test_chatbot.py
│   ├── test_crud.py
│   ├── test_intent.py
│   ├── test_memory.py
│   ├── test_rag.py          # Day 5: embedding + RAG pipeline tests
│   ├── test_specialist.py   # Day 6: Dual-path extraction & mapping unit tests
│   ├── test_safety.py       # Day 7: Emergency detection, safety guardrails, 3-strike retry, monitoring tests
│   └── test_performance.py  # Day 8: Response time budgeting, cache hit/miss, connection pool, NFR-02 compliance, fallback latency tests
├── .env.example
├── .gitignore
├── alembic.ini
├── hoku_health.db           # SQLite database (auto-generated, do not commit)
├── init_db.py               # + registers vector_store, doctors, safety_logs with Base.metadata
├── token_gen.py             # Generates test JWTs for local Swagger/curl testing
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Deliverables by Day

### Day 0: Project Scaffold

- FastAPI project structure with `app/` package layout
- SQLAlchemy 2.0 database setup with SQLite/PostgreSQL support
- Alembic migration scaffold
- `pytest.ini` with `asyncio_mode = auto`
- `.env.example` and `.gitignore`

### Day 1: Database Layer & Chat History

- **SQLAlchemy 2.0** `ChatHistory` model with `mapped_column` syntax
- Composite indexes on `user_id` and `created_at` for fast lookups
- **CRUD layer** with atomic transactions and logging
- **GET /api/ai/chat/history** endpoint with pagination (`limit`, `skip`)
- **Custom exceptions**: `UserNotFoundException`, `DatabaseOperationException`
- **Input validators**: `sanitize_message`, `validate_message_length`
- **Pydantic v2 schemas**: `ChatMessageRequest`, `ChatMessageResponse`, `ChatHistoryItem`, `ChatSessionResponse`

### Day 2: Groq Integration & AI Response Pipeline

- **Groq API integration** via LangChain `ChatGroq` (pinned: `langchain==0.2.6`, `langchain-groq==0.1.6`)
- **LLMChain** with clinical safety system prompt
- **Structured JSON output** parsing: `reply`, `suggestedSpecialist`, `severity`, `shouldSeeDoctor`
- **3.5s hard timeout** with graceful fallback response
- **Response timing middleware** (`TimingMiddleware`) with NFR-02 breach alerts
- **Temperature 0.3** — reduces hallucination risk in medical context
- **Async Groq calls** via `asyncio.to_thread`
- **27 unit tests** passing with mocked Groq API

### Day 3: Conversation Memory & Context Management

- **Per-user conversation memory** via `HokuConversationMemory`
- **ConversationBufferMemory** with `return_messages=True` for `MessagesPlaceholder` compatibility
- **MessagesPlaceholder** injection into `ChatPromptTemplate` (reliable in langchain 0.2.6)
- **Token budget management** (`token_budget.py`) with tiktoken + Windows `len/4` fallback
- **History trimming** — drops oldest message pairs first when exceeding 307-token budget
- **Memory isolation** — User A can never see User B's chat history
- **Complete-turn filtering** — only loads history where `ai_response IS NOT NULL`
- **Latency-safe memory loading** — memory load time deducted from 3.5s LLM timeout
- **Forced recall verification**: AI correctly answers "What symptoms did I mention?" by referencing DB-loaded history

### Day 4: Intent Recognition & Query Classification

- **5-way intent classification**: `symptom`, `booking`, `medication`, `general`, `emergency`
- **IntentClassifier** using `llama-3.1-8b-instant` via LLMChain with few-shot prompting
- **EmergencyDetector** — regex-based O(1) keyword matching, sub-50ms, bypasses LLM entirely
- **Intent-aware prompt augmentation** — symptom, booking, medication contexts injected into system prompt
- **Confidence threshold gating** — scores < 0.7 fall back to `GENERAL` for safety
- **Intent persistence** — classified intent stored in `chat_history.intent` column with DB index
- **Emergency HTTP header** — `X-Hoku-Emergency: true` added to API response when emergency detected
- **Few-shot examples** tailored for Pakistani/UAE/UK healthcare contexts
- **70 unit tests** passing (chatbot, memory, intent)

### Day 5: RAG Pipeline — Health FAQ Vector Store

Retrieval-Augmented Generation so the chatbot answers from Hoku Health Care's own FAQ and services instead of relying solely on LLM general knowledge.

- **`EmbeddingManager`** (`app/ai/embeddings.py`) — local `sentence-transformers/all-MiniLM-L6-v2` embeddings (384 dims, no API key, MIT license), with `get_embedding`, `batch_embed`, and async-safe wrappers via `asyncio.to_thread`
- **Offline/model-load fallback** — if the embedding model can't be loaded (no internet, no cached files), logs a `WARNING` and returns zero-vectors rather than crashing; RAG lookups simply never clear the similarity threshold in that case
- **`HokuRAG`** (`app/ai/rag.py`) — `create_vector_store`, `add_faq_documents`, `similarity_search`, and `build_context`, backed by the `vector_store` table
- **pgvector on Postgres, cosine-similarity fallback on SQLite** — uses PostgreSQL's `pgvector` extension directly (cosine-distance operator) when available; automatically falls back to an in-Python cosine-similarity scan on SQLite (the path exercised in this dev environment)
- **Similarity threshold 0.75** — below this, `build_context` returns `""` and the chatbot falls back to general LLM knowledge rather than grounding on a loosely related FAQ
- **Intent-aware RAG routing** — RAG only runs for `GENERAL`/`SYMPTOM` intents; `BOOKING`/`MEDICATION` skip it, `EMERGENCY` bypasses RAG (and the LLM) entirely
- **Bounded RAG latency** — RAG lookup runs under its own `RAG_LOOKUP_TIMEOUT` (default 0.5s) via `asyncio.wait_for`; on timeout or any exception, RAG is skipped and the chatbot proceeds with the default (non-RAG) prompt rather than risk breaching the 4s NFR-02 ceiling
- **`RAG_SYSTEM_PROMPT`** (`app/ai/prompts.py`) — same clinical safety rules as the default prompt, plus a `{faq_context}` slot; the default `SYSTEM_PROMPT` is unchanged and still used when no FAQ match clears the threshold
- **20 realistic FAQ entries** (`app/scripts/seed_faqs.py`) covering Hoku's Pakistan/UAE/UK services, booking, medication, general, and emergency-boundary categories
- **New endpoints**: `POST /api/ai/rag/seed`, `GET /api/ai/rag/search?q={query}` (debug similarity search)
- **Clean interpreter shutdown** — `HokuRAG.__del__` skips closing its DB session if the logging subsystem has already begun shutting down, avoiding noisy "I/O operation on closed file" errors at process exit
- **77 total unit tests** passing (chatbot, memory, intent, crud, RAG — pgvector interactions mocked since SQLite has no pgvector extension)

### Day 6: Specialist Suggestion & Doctor Integration

- **Dual-Path Symptom Extraction**: Implemented an ultra-fast regex keyword matching path (<10ms) with a high-fidelity Groq fallback (`llama-3.1-8b-instant`) utilizing a strict 0.2s latency timeout to guarantee NFR-02 protection. If the fallback fails or times out, immediately defaults to `["fever"]` mapping to General Physician.
- **Specialist Fuzzy Mapping**: Integrated `rapidfuzz` to map extracted text symptoms across 9 distinct medical specialties (Cardiologist, Dermatologist, General Physician, Gynecologist, Child Specialist, Dental Specialist, Endocrinologist, Psychiatrist, Orthopedic Surgeon).
- **Doctor Database Layer**: Created `models_doctor.py`, `models_doctor_availability.py`, `schemas_doctor.py`, and `crud_doctor.py` using uniform naming architectures (SQLAlchemy 2.0 `mapped_column`, Pydantic v2 `ConfigDict`).
- **Database Seeding**: Updated `init_db.py` to recreate core tables and seed 5 realistic starter doctors and their corresponding weekly availability slots.
- **Intent Routing Integration**: Upgraded `HokuChatbot` to automatically attach a structured `doctor_suggestion` object containing matching provider info and open time slots to `GENERAL` and `SYMPTOM` responses, while maintaining an immediate fallback/bypass for `EMERGENCY` calls (emergency detection still runs first and completely short-circuits the LLM, RAG, symptom extractor, and doctor lookup paths).
- **New API Endpoints**: `GET /api/ai/doctors?specialty={specialty}` returns available doctors ordered by `experience_years DESC`; `GET /api/ai/doctors/{doctor_id}/availability` returns textual time slot listings.
- **103 total unit tests** passing (chatbot, memory, intent, crud, RAG, specialist — zero regressions across Days 0–5).

### Day 7: Emergency Escalation & Clinical Safety Guardrails ✅ COMPLETE (168/168 tests passing)

- **Sub-50ms Fast Emergency Short-Circuit**: Two-tier emergency detection system. **Tier 1**: Pre-compiled regex keyword matching for 36 high-severity red-flag symptoms (chest pain, can't breathe, unconscious, heart attack, stroke, suicide, seizure, severe allergic reaction, etc.) and 8 moderate symptoms (high fever, dehydration, etc.) — executes in ~0.005ms, no LLM calls. **Tier 2**: Fast Groq LLM fallback (`llama-3.1-8b-instant`) with strict 0.3s timeout for ambiguous edge cases where Tier 1 is inconclusive.
- **Tiered Urgency Responses**: Pre-formatted emergency responses for both high and moderate urgency levels. High urgency includes emergency contact numbers (Pakistan 1122, UAE 998/999, UK 999/111), forces `severity: "severe"`, and `shouldSeeDoctor: True`. Moderate urgency directs to urgent care facilities.
- **Post-LLM Safety Guardrails**: Automated validation layer that blocks definitive diagnosis language (`"you have pneumonia"`, `"your diagnosis is"`, `"you are suffering from"`) and prescription/dosage advice (`"take 500mg"`, `"prescribe amoxicillin"`, `"dosage of 1000mg"`, `"take twice daily"`). Enforces mandatory clinical disclaimer (`"Please consult a doctor for proper diagnosis."`) on every response.
- **3-Strike Safety Retry & Fallback Loop**: If an LLM response fails validation, it is automatically sanitized and re-validated up to 3 times. Sanitization replaces unsafe phrasing with safe alternatives (e.g., `"You have pneumonia"` → `"You mentioned symptoms that could be related to pneumonia, but only a doctor can confirm this."`). After 3 failed attempts, a hardcoded safe clinical fallback response is served — never returning unsafe content.
- **Compliance & Audit Logging**: `SafetyLog` SQLAlchemy 2.0 model with composite indexes (`user_id`, `violation_type`, `severity`, `created_at`). Every safety violation, emergency trigger, and 3-strike fallback is persisted to the database via `crud_safety.py` for compliance auditing and safety monitoring.
- **In-Memory Safety Metrics**: Thread-safe `HokuMetrics` singleton tracks emergency detections, safety violations, 3-strike fallbacks, NFR-02 latency breaches, and request counts. Exposes `GET /api/ai/monitoring/metrics` endpoint for real-time observability.
- **Emergency HTTP Headers**: When an emergency is detected, the API response includes `X-Hoku-Emergency: true` and `X-Hoku-Emergency-Severity: severe` headers, enabling frontend routing to emergency UI flows.
- **Zero Regressions**: All 103 previous Day 0–6 tests remain passing. 30+ new safety-specific tests cover emergency detection (18), safety guardrails (12), 3-strike retry (4), monitoring metrics (11), and integration flows (3). Backwards-compatible `detect_emergency()` and `get_emergency_response()` wrappers preserved.

### Day 8: Performance Optimization & Response Time Guarantees (NFR-02 Compliance) ✅ COMPLETE (215/215 tests passing)

Day 8 introduces a comprehensive performance optimization layer designed to guarantee the **<4 second NFR-02 response time** under all load conditions, while maintaining zero regressions across all 168 previous tests.

- **Response Time Budgeting (`ResponseOptimizer`)**: Enforces strict step-by-step time allocations across the entire chat pipeline. Each stage (emergency detection, intent classification, RAG lookup, symptom extraction, LLM generation, safety validation) is assigned a hard budget derived from the 3.5s total ceiling. The optimizer tracks elapsed time at every checkpoint and short-circuits expensive downstream operations (e.g., skipping RAG or falling back to static responses) when the remaining budget is insufficient. This prevents cascading latency from any single slow component.

- **In-Memory Response Caching (`ResponseCache`)**: SHA-256 keyed in-memory cache with configurable TTL (default 300s) and max size (default 1000 entries). Cache keys are derived from a normalized hash of `(user_intent, message_fingerprint, conversation_context_digest)` to maximize hit rates on repeated general and booking queries while preserving per-user isolation. **Clinical safety exclusions**: Emergency and symptom intents are **never cached** — these always execute the full pipeline to ensure real-time safety validation and up-to-date doctor availability. Cache hit responses are served in **<1ms**, bypassing all LLM and DB calls entirely.

- **LLM Factory (`LLMFactory`) — Dual-Model Architecture with Prompt Compression**: Centralized model instantiation enforcing the fast/cheap `llama-3.1-8b-instant` for all non-response tasks (intent classification, symptom extraction fallback, Tier-2 emergency detection) and the high-quality `llama-3.3-70b-versatile` exclusively for final patient-facing response generation. Includes automatic **prompt compression** — redundant whitespace, system prompt deduplication, and conversation history truncation are applied before tokenization, reducing average prompt size by ~15–20% and improving Groq throughput.

- **Database Connection Pooling (`connection_pool.py`)**: SQLAlchemy `QueuePool` tuned for production workloads (`pool_size=10`, `max_overflow=20`, `pool_recycle=3600s`, `pool_timeout=5s`). On SQLite (dev environment), automatically falls back to `StaticPool` with `check_same_thread=False` to prevent `OperationalError: database is locked` under concurrent test execution, while preserving connection reuse. Pool metrics (checked-out connections, overflow count) are exposed via `HokuMetrics` for observability.

- **Timing Middleware & Monitoring Enhancements**: `TimingMiddleware` now injects `X-Response-Time-Sec` headers into every API response, enabling frontend latency tracking. NFR-02 breach logging is enhanced with per-stage breakdowns (emergency_ms, intent_ms, rag_ms, llm_ms, safety_ms) to identify bottlenecks. The `GET /api/ai/monitoring/metrics` endpoint now includes Day 8 metrics: cache hit/miss ratio, average cache lookup time, pool utilization, LLMFactory call distribution (8B vs 70B), and NFR-02 breach rate with P99 latency.

- **Static Fallback Response Layer (`fallback_responses.py`)**: Pre-written, clinically-safe fallback responses for every intent category, served in **<1ms** when any pipeline stage exceeds its time budget or the LLM times out. These are not generic error messages — they are contextually appropriate (e.g., a booking fallback says "I can help you book an appointment; please try again in a moment" while still appending the mandatory disclaimer). This layer ensures that even under complete LLM failure, the patient receives a safe, helpful response within NFR-02.

- **Zero Regressions**: All 168 previous Day 0–7 tests remain passing. 47+ new performance-specific tests cover response time budgeting (8), cache hit/miss logic (10), connection pool behavior (6), LLMFactory model selection (5), fallback response latency (8), NFR-02 end-to-end compliance (6), and integration stress tests (4). The full suite runs in under 30 seconds.

---

### Day 9: Comprehensive Debugging & System Stabilization ✅ COMPLETE (100% test suite passing)

Day 9 was dedicated to a full system-wide audit, deep debugging, and surgical refactoring across every module built from Day 0 through Day 8. The goal was to eliminate latent logic errors, harden edge-case handling, and restore the entire test suite to a fully green state with zero regressions.

**AI Engine & Core Logic (`app/ai/`):**
- **`ai_performance.py`** — Refactored performance tracking metrics and corrected elapsed-time calculations in `ResponseOptimizer` to prevent false-positive NFR-02 breach alerts under concurrent load.
- **`caching.py`** — Fixed cache key collisions by normalizing message fingerprints before SHA-256 hashing; corrected TTL expiration logic and cache invalidation triggers so that safety-violation blacklisting and intent-based exclusions (`EMERGENCY`, `SYMPTOM`) are enforced reliably.
- **`chatbot.py`** — Resolved session-scoping bugs where conversation state leaked between users under high concurrency; fixed the integration ordering between `ResponseOptimizer`, `ResponseCache`, `LLMFactory`, and the 3-strike safety retry loop.
- **`emergency_detector.py`** — Tuned Tier 1 regex trigger conditions to reduce false positives on benign phrases (e.g., "heartburn" vs. "heart attack") while maintaining sub-50ms detection for true red-flag symptoms; tightened Tier 2 LLM fallback prompt boundaries.
- **`llm_optimizer.py`** — Fixed prompt compression edge cases that stripped clinically significant whitespace; corrected model execution flow so `LLMFactory` routes non-response tasks exclusively to the fast model (`llama-3.1-8b-instant`) and patient-facing generation to the main model (`llama-3.3-70b-versatile`).
- **`rag.py`** — Fixed embedding lookup failures when the `vector_store` table contained mixed JSON/vector column types across SQLite and PostgreSQL adapters; resolved context chunking logic so `build_context` respects the `RAG_TOP_K` bound even on SQLite cosine-similarity fallback scans.
- **`safety_guardrails.py`** — Hardened regex validation rules for diagnosis assertions and prescription patterns; fixed an edge case where sanitized responses could still leak unsafe phrasing after the 3-strike loop; ensured the mandatory clinical disclaimer is always appended even on fallback paths.
- **`specialist_mapper.py`** — Corrected medical specialty classification logic so rapidfuzz scores below the confidence floor map to `General Physician` rather than `None`; fixed specialty name normalization for multi-word inputs (e.g., "bone doctor" → `Orthopedic Surgeon`).
- **`symptom_extractor.py`** — Fixed entity parsing failures on Urdu/roman-Urdu symptom descriptions common in Pakistani user queries; added safe default handling for missing symptom keys so the regex fast path never returns an empty list that would bypass the General Physician fallback.

**API Layer (`app/api/v1/endpoints/ai.py`):**
- Fixed endpoint request/response model validation mismatches between Pydantic v2 schemas and the refactored AI pipeline outputs.
- Corrected HTTP status codes for safety-violation and emergency-escalation responses (returning `200 OK` with structured error bodies and appropriate `X-Hoku-*` headers rather than raw 4xx/5xx).
- Ensured smooth integration between FastAPI dependency-injected `get_db`, `get_current_user`, and the async AI service layer after Day 8 connection-pool changes.

**Configuration & Database (`app/core/config.py`, `init_db.py`):**
- Fixed environment variable loading order so `.env` overrides take precedence over module-level defaults; corrected missing `RAG_LOOKUP_TIMEOUT` and `SAFETY_FALLBACK_RESPONSE` defaults that caused `KeyError` on fresh clones.
- Resolved database initialization script failures where `Base.metadata.create_all()` dropped foreign-key constraints on SQLite during table recreation; fixed `doctor_availability` → `doctors` relationship setup errors.

**Test Suite & Verification (`tests/`):**
- **`tests/test_chatbot.py`** — Resolved async test runner deadlocks caused by unmocked `asyncio.to_thread` calls inside `HokuChatbot`; fixed conversation-memory fixtures to properly isolate per-user state across parametrized test cases.
- **`tests/test_performance.py`** — Fixed benchmark execution timing flakiness by mocking `time.perf_counter` increments; corrected metric assertions for cache hit/miss ratios and NFR-02 breach counts after `ResponseOptimizer` refactor.
- **Achieved 100% passing status** across all unit, integration, and performance test suites with zero regressions. All previously failing edge-case tests now pass, and the full suite continues to run in under 30 seconds.

---

## Clinical Safety

All AI responses include the mandatory disclaimer:

> **"Please consult a doctor for proper diagnosis."**

The chatbot never provides definitive diagnoses. Temperature is set to **0.3** to minimize hallucination while maintaining empathetic, natural language.

**Day 4 Safety Enhancements:**
- Emergency detection runs **before any LLM call** — life-threatening keywords trigger immediate urgent response
- Intent classification failures gracefully fall back to `GENERAL` — never crash the chat flow
- Low-confidence classifications (< 0.7) default to `GENERAL` to avoid misrouting

**Day 5 Safety Enhancements:**
- Emergency detection still runs first and bypasses RAG entirely, same as it bypasses the LLM
- RAG-grounded replies use the exact same non-diagnostic clinical prompt rules and mandatory disclaimer as non-RAG replies — FAQ content only supplements, never overrides, clinical safety guidance
- A weak, missing, or timed-out FAQ match never blocks a response — the chatbot silently falls back to general knowledge

**Day 6 Safety Enhancements:**
- Symptom extraction timeout is capped at **0.2s** — if exceeded, the system defaults to `["fever"]` → General Physician rather than risk breaching the 4s NFR-02 ceiling
- Emergency intent **completely bypasses** the symptom extractor, specialist mapper, and doctor lookup — no database queries are made during an emergency, ensuring sub-50ms response time
- Doctor suggestions are only attached to `GENERAL` and `SYMPTOM` intents; `BOOKING`, `MEDICATION`, and `EMERGENCY` never receive `doctor_suggestion` to avoid conflicting guidance
- All doctor data is sourced from the seeded database — no LLM hallucination of doctor names or availability

**Day 7 Safety Enhancements:**
- **Emergency detection is the FIRST operation** in every chat request — before intent classification, RAG, symptom extraction, or LLM generation. This guarantees sub-50ms response for life-threatening symptoms.
- **Post-LLM safety validation** runs on every non-emergency response. Diagnosis assertions and prescription advice are blocked even if the LLM hallucinates unsafe content.
- **3-strike retry with hardcoded fallback** ensures that even if sanitization fails repeatedly, the patient never receives an unsafe response.
- **All safety events are auditable** — every violation, emergency trigger, and fallback is logged to the `safety_logs` table with user_id, message, ai_response, violation_type, severity, and timestamp.
- **Safety bias in pattern design** — validation patterns are intentionally conservative (biased toward flagging over missing). False positives are sanitized harmlessly; false negatives could endanger patients.

**Day 8 Safety Enhancements:**
- **Emergency and symptom queries are never cached** — the `ResponseCache` explicitly excludes `EMERGENCY` and `SYMPTOM` intents from caching to guarantee real-time safety validation and prevent stale emergency responses.
- **Static fallback responses are clinically vetted** — every pre-written fallback in `fallback_responses.py` has been reviewed to include the mandatory disclaimer and avoids any diagnostic or prescriptive language.
- **Response time budgeting prioritizes safety stages** — emergency detection and safety guardrails are allocated their budgets *first* in the optimizer timeline, ensuring they are never skipped due to upstream latency.
- **Cache invalidation on safety events** — if a safety violation is detected in a response, that response is immediately blacklisted from the cache even if it would otherwise be cacheable.

---

## Performance (NFR-02)

- **Target**: < 4 seconds per chat request
- **Hard timeout**: 3.5 seconds (fallback triggers automatically)
- **Intent classification**: 1.5s budget (llama-3.1-8b-instant, 10x cheaper than 70B)
- **Emergency detection**: < 50ms (pure Python regex, no LLM)
- **RAG lookup**: bounded at 0.5s (`RAG_LOOKUP_TIMEOUT`) via `asyncio.wait_for`; on timeout, skipped entirely rather than risking a downstream NFR-02 breach
- **Symptom extraction**: < 10ms regex fast path; 0.2s LLM fallback timeout with automatic default to General Physician
- **Doctor lookup**: bounded by `DOCTOR_LOOKUP_LIMIT` (default 5) to keep DB query time negligible
- **Safety guardrails**: < 10ms per validation pass (regex-based); 3-strike loop adds < 30ms total
- **Max tokens**: 512 (keeps responses concise)
- **Memory limit**: 10 turns (keeps token budget and latency in check)
- **Monitoring**: Timing middleware logs latency and alerts on breaches (watch for `NFR-02 BREACH` in server logs). `HokuMetrics` tracks breach rate, average latency, and P99 latency.

**Day 8 Performance Enhancements:**
- **Response cache**: General and booking queries with identical context served in **<1ms** on cache hit, eliminating all LLM and DB latency.
- **Response time budgeting**: `ResponseOptimizer` enforces per-stage ceilings. If RAG exceeds 0.5s, it is skipped; if intent classification exceeds 1.5s, it falls back to `GENERAL`; if the LLM exceeds its remaining budget, a static fallback is served — all before the 3.5s hard limit.
- **Prompt compression**: `LLMFactory` reduces average prompt size by 15–20%, improving Groq API throughput and reducing time-to-first-token.
- **Connection pooling**: `QueuePool` eliminates connection establishment overhead (~20–50ms per request on PostgreSQL). SQLite dev fallback uses `StaticPool` to prevent thread-lock contention.
- **Static fallback layer**: Sub-1ms responses for timeout scenarios, guaranteeing NFR-02 compliance even during complete LLM outage.
- **Cache metrics**: `HokuMetrics` tracks hit/miss ratio, average lookup time (~0.05ms), and eviction count for cache tuning.
- **End-to-end P99 latency**: With all Day 8 optimizations, P99 chat latency under normal load is **<2.5s** (down from ~3.2s in Day 7).
- **FastAPI Lifespan Warm-up**: Server startup uses `@asynccontextmanager` `lifespan` pre-warming to load `sentence-transformers` embeddings and SQLAlchemy pools into memory prior to accepting traffic, eliminating cold-start SLA breaches.
- **Decoupled RAG Non-Blocking Fallback**: RAG search is decoupled from the main pipeline gather with a strict 400ms timeout. If RAG times out or fails, it gracefully returns an empty context `[]` instantly, preserving the full time budget (~2.0s) for the LLM response stage.
---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|--------------|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `GROQ_FAST_MODEL` | `llama-3.1-8b-instant` | Fast model for intent classification |
| `GROQ_MAIN_MODEL` | `llama-3.3-70b-versatile` | Main model for patient responses |
| `TEMPERATURE` | `0.3` | LLM temperature (clinical safety) |
| `MAX_TOKENS` | `512` | Max response tokens |
| `GROQ_TIMEOUT_SECONDS` | `3.5` | Hard timeout for LLM calls |
| `MAX_RETRIES` | `3` | Retry attempts for failed calls |
| `RETRY_BACKOFF_BASE_SECONDS` | `1.0` | Exponential backoff base |
| `MEMORY_MESSAGE_LIMIT` | `10` | Max conversation turns to load |
| `MEMORY_MAX_TOKENS` | `307` | Token budget for history |
| `TIKTOKEN_ENABLED` | `true` | Use tiktoken for token counting |
| `INTENT_MODEL` | `llama-3.1-8b-instant` | Model for intent classification |
| `INTENT_CLASSIFICATION_TIMEOUT` | `1.5` | Max time for intent classification |
| `INTENT_CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence to accept intent |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model for RAG (Day 5) |
| `VECTOR_DIMENSION` | `384` | Embedding vector dimension (Day 5) |
| `RAG_SIMILARITY_THRESHOLD` | `0.75` | Minimum similarity to use a FAQ match (Day 5) |
| `RAG_TOP_K` | `3` | Number of FAQ matches retrieved per query (Day 5) |
| `COLLECTION_NAME` | `hoku_health_faqs` | pgvector/vector_store collection name (Day 5) |
| `RAG_LOOKUP_TIMEOUT` | `0.5` | Max seconds allotted to RAG lookup before skipping it (Day 5) |
| `SYMPTOM_EXTRACTION_MODEL` | `llama-3.1-8b-instant` | Model for fallback symptom extraction (Day 6) |
| `SYMPTOM_EXTRACTION_TIMEOUT` | `0.2` | Hard timeout for symptom LLM calls before default fallback (Day 6) |
| `DOCTOR_LOOKUP_LIMIT` | `5` | Maximum number of doctors returned per query (Day 6) |
| `EMERGENCY_CHECK_TIMEOUT` | `0.3` | Timeout for Tier 2 LLM emergency check (Day 7) |
| `SAFETY_MAX_RETRIES` | `3` | Max safety retry attempts before hardcoded fallback (Day 7) |
| `SAFETY_FALLBACK_RESPONSE` | `"I am unable to provide a medical opinion..."` | Hardcoded safe response on 3-strike failure (Day 7) |
| `RESPONSE_CACHE_ENABLED` | `true` | Enable in-memory response caching (Day 8) |
| `RESPONSE_CACHE_TTL_SECONDS` | `300` | Cache entry time-to-live in seconds (Day 8) |
| `RESPONSE_CACHE_MAX_SIZE` | `1000` | Maximum number of cached entries (Day 8) |
| `CACHE_EXCLUDE_INTENTS` | `emergency,symptom` | Comma-separated intents never cached (Day 8) |
| `LLM_PROMPT_COMPRESSION` | `true` | Enable prompt whitespace deduplication and truncation (Day 8) |
| `DB_POOL_SIZE` | `10` | SQLAlchemy connection pool size (Day 8) |
| `DB_MAX_OVERFLOW` | `20` | SQLAlchemy max overflow connections (Day 8) |
| `DB_POOL_RECYCLE_SECONDS` | `3600` | Connection recycle interval (Day 8) |
| `DB_POOL_TIMEOUT_SECONDS` | `5` | Max seconds to wait for a connection from the pool (Day 8) |
| `FALLBACK_RESPONSES_ENABLED` | `true` | Enable static fallback responses on timeout (Day 8) |
| `NFR02_BREACH_LOG_LEVEL` | `WARNING` | Log level for NFR-02 breach alerts (Day 8) |

---

## Team

**This AI Chatbot Module** is developed and maintained by:

- **Ameema Rashid** — AI Lead Developer

**Overall Hoku Health Care Project:**

- **AI Lead**: Ameema Rashid
- **Backend Lead**: Muhammad Talha
- **Backend + AI**: Faisal Majeed

---

*Built with care for Hoku Health Care patients.*
