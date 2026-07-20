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
```

**For local development without PostgreSQL** (default in this environment):

```bash
DATABASE_URL=sqlite:///./hoku_health.db
```

### 4. Initialize the Database

```bash
python init_db.py
```

Creates `chat_history`, `vector_store` (Day 5), `doctors`, and `doctor_availability` (Day 6) tables. On SQLite, `vector_store.embedding` is a JSON column; on PostgreSQL with pgvector installed, it's a native `vector(384)` column — the model detects this automatically from `DATABASE_URL`.

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
| POST | `/api/ai/chat` | Yes | AI Health Chatbot (Groq LLM + Memory + Intent + RAG + Doctor Suggestion) |
| GET | `/api/ai/chat/history` | Yes | Chat History (paginated) |
| GET | `/api/ai/health` | No | Service Health Check |
| POST | `/api/ai/rag/seed` | No | Seed the Hoku FAQ vector store (Day 5, admin/dev use) |
| GET | `/api/ai/rag/search?q={query}` | No | Debug FAQ similarity search (Day 5, admin/dev use) |
| GET | `/api/ai/doctors?specialty={specialty}` | Yes | Retrieves available doctors ordered by experience (Day 6) |
| GET | `/api/ai/doctors/{doctor_id}/availability` | Yes | Fetches textual time slot listings for a specific doctor (Day 6) |

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
│   │   ├── __init__.py
│   │   ├── config.py     # AI hyperparameters, timeouts, memory, RAG_LOOKUP_TIMEOUT, symptom extraction, doctor lookup
│   │   ├── prompts.py    # Clinical safety + intent + RAG-grounded + specialist suggestion prompts
│   │   ├── chatbot.py    # HokuChatbot: intent + emergency + bounded RAG lookup + doctor suggestion (Day 6)
│   │   ├── intent_classifier.py  # 5-way intent classification
│   │   ├── emergency_detector.py # Regex-based emergency detection
│   │   ├── memory.py     # Per-user ConversationBufferMemory loader
│   │   ├── token_budget.py # Token counting & history trimming
│   │   ├── embeddings.py  # Day 5: local sentence-transformers embedding manager
│   │   ├── rag.py         # Day 5: HokuRAG similarity search + context builder
│   │   ├── specialist_mapper.py  # Day 6: Fuzzy matching for symptom-to-specialist mapping
│   │   ├── symptom_extractor.py  # Day 6: Dual-path regex + Groq LLM symptom extraction
│   │   └── utils.py      # Response parsers
│   ├── api/v1/endpoints/
│   │   ├── __init__.py
│   │   └── ai.py          # + /api/ai/rag/seed, /api/ai/rag/search (Day 5), /api/ai/doctors (Day 6)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py       # + VECTOR_DIMENSION, RAG_SIMILARITY_THRESHOLD, RAG_TOP_K, COLLECTION_NAME
│   │   ├── database.py     # engine, SessionLocal, Base, get_db
│   │   ├── dependencies.py # re-exports get_db, get_current_user
│   │   ├── exceptions.py   # UserNotFoundException, DatabaseOperationException
│   │   ├── middleware.py   # Request timing & NFR-02 monitoring
│   │   └── security.py     # JWT auth (HTTPBearer stub, pending Talha)
│   ├── crud/
│   │   ├── __init__.py     # re-exports app.crud.crud_chat
│   │   ├── crud_chat.py
│   │   └── crud_doctor.py  # Day 6: Doctor & availability CRUD lookups
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── cors.py
│   │   └── error_handler.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── models_chat.py
│   │   ├── models_doctor.py              # Day 6: Doctor model (SQLAlchemy 2.0)
│   │   └── models_doctor_availability.py  # Day 6: Doctor Availability slots model
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── schemas_chat.py  # Updated: Added doctor_suggestion to ChatMessageResponse (Day 6)
│   │   └── schemas_doctor.py  # Day 6: Pydantic v2 Doctor & Suggestion schemas
│   ├── scripts/             # Day 5
│   │   ├── __init__.py
│   │   └── seed_faqs.py     # Seeds 20 Hoku FAQ entries
│   ├── services/
│   │   ├── __init__.py
│   │   └── ai_service.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   └── validators.py
│   ├── __init__.py
│   └── main.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # DB + client fixtures, mock intent/emergency fixtures
│   ├── test_chatbot.py
│   ├── test_crud.py
│   ├── test_intent.py
│   ├── test_memory.py
│   ├── test_rag.py          # Day 5: embedding + RAG pipeline tests
│   └── test_specialist.py   # Day 6: Dual-path extraction & mapping unit tests
├── .env.example
├── .gitignore
├── alembic.ini
├── hoku_health.db           # SQLite database (auto-generated, do not commit)
├── init_db.py               # + registers vector_store with Base.metadata (Day 5), seeds doctors (Day 6)
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

---

## Performance (NFR-02)

- **Target**: < 4 seconds per chat request
- **Hard timeout**: 3.5 seconds (fallback triggers automatically)
- **Intent classification**: 1.5s budget (llama-3.1-8b-instant, 10x cheaper than 70B)
- **Emergency detection**: < 50ms (pure Python regex, no LLM)
- **RAG lookup**: bounded at 0.5s (`RAG_LOOKUP_TIMEOUT`) via `asyncio.wait_for`; on timeout, skipped entirely rather than risking a downstream NFR-02 breach
- **Symptom extraction**: < 10ms regex fast path; 0.2s LLM fallback timeout with automatic default to General Physician
- **Doctor lookup**: bounded by `DOCTOR_LOOKUP_LIMIT` (default 5) to keep DB query time negligible
- **Max tokens**: 512 (keeps responses concise)
- **Memory limit**: 10 turns (keeps token budget and latency in check)
- **Monitoring**: Timing middleware logs latency and alerts on breaches (watch for `NFR-02 BREACH` in server logs)

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
