# Hoku Health Care - AI Chatbot Module

**TechNexus Virtual University | Internship Project**

This module provides the AI-powered health chatbot backend for Hoku Health Care, a home healthcare platform serving patients in Pakistan, UAE, and UK.

---

## Tech Stack

- **Framework**: FastAPI (Python)
- **AI/LLM**: Groq API (Llama 3) via LangChain 0.2.6
- **Database**: PostgreSQL + SQLAlchemy 2.0 + Alembic
- **Auth**: JWT (stubbed for AI module setup)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2) — stubbed for RAG

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Groq API key ([Get one here](https://console.groq.com/keys))

### 2. Installation

```bash
# Clone the repository
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

**Required `.env` variables for Day 3:**
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
```

**For local development without PostgreSQL**, change `DATABASE_URL` in `.env`:
```bash
DATABASE_URL=sqlite:///./hoku_health.db
```

### 4. Run Tests

```bash
# Unit tests (mocked Groq — no API key needed)
pytest tests/test_chatbot.py -v

# Memory tests
pytest tests/test_memory.py -v

# Full test suite
pytest tests/ -v
```

### 5. Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. API Documentation

Once running, open your browser:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## AI Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/ai/chat` | AI Health Chatbot (Groq LLM + Memory) |
| GET | `/api/ai/chat/history` | Chat History (paginated) |
| GET | `/api/ai/health` | Service Health Check |

---

## Project Structure

```
hoku-health-backend/
├── alembic/              # Database migrations
│   └── versions/
│       └── 001_create_chat_history.py
├── app/
│   ├── ai/               # Chatbot engine (Groq + LangChain)
│   │   ├── __init__.py
│   │   ├── config.py     # AI hyperparameters, timeouts, memory settings
│   │   ├── prompts.py    # Clinical safety prompts with MessagesPlaceholder
│   │   ├── chatbot.py    # HokuChatbot class with memory integration
│   │   ├── memory.py     # Per-user ConversationBufferMemory loader
│   │   ├── token_budget.py # Token counting & history trimming
│   │   └── utils.py      # Response parsers
│   ├── api/              # API routers
│   │   └── v1/
│   │       └── endpoints/
│   │           ├── __init__.py
│   │           └── ai.py
│   ├── core/             # Config, DB, security, middleware
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── dependencies.py
│   │   ├── exceptions.py
│   │   ├── middleware.py # Request timing & NFR monitoring
│   │   └── security.py
│   ├── crud/             # Database access layer
│   │   ├── __init__.py
│   │   └── chat.py
│   ├── middleware/       # CORS & error handlers
│   │   ├── __init__.py
│   │   ├── cors.py
│   │   └── error_handler.py
│   ├── models/           # SQLAlchemy models
│   │   ├── __init__.py
│   │   └── chat.py
│   ├── schemas/          # Pydantic schemas
│   │   ├── __init__.py
│   │   └── chat.py
│   ├── services/         # Business logic layer
│   │   ├── __init__.py
│   │   └── ai_service.py
│   ├── utils/            # Constants & validators
│   │   ├── __init__.py
│   │   ├── constants.py
│   │   └── validators.py
│   ├── __init__.py
│   └── main.py           # FastAPI application entry point
├── tests/
│   ├── __init__.py
│   ├── conftest.py       # Pytest fixtures
│   ├── test_chatbot.py   # Unit tests (mocked Groq)
│   ├── test_memory.py    # Memory & token budget tests
│   └── test_crud.py      # CRUD verification
├── .env.example
├── .gitignore
├── alembic.ini
├── hoku_health.db        # SQLite database (auto-generated, do not commit)
├── init_db.py
├── pytest.ini            # Pytest configuration
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
- **CRUD layer** (`app/crud/chat.py`) with atomic transactions and logging
- **GET /api/ai/chat/history** endpoint with pagination (`limit`, `skip`)
- **Custom exceptions**: `UserNotFoundException`, `DatabaseOperationException`
- **Input validators**: `sanitize_message`, `validate_message_length`
- **Pydantic v2 schemas**: `ChatMessageRequest`, `ChatMessageResponse`, `ChatHistoryItem`, `ChatSessionResponse`
- SQLite test script for offline verification

### Day 2: Groq Integration & AI Response Pipeline

- **Groq API integration** via LangChain `ChatGroq` (pinned: `langchain==0.2.6`, `langchain-groq==0.1.6`)
- **LLMChain** with clinical safety system prompt
- **Structured JSON output** parsing: `reply`, `suggestedSpecialist`, `severity`, `shouldSeeDoctor`
- **3.5s hard timeout** with graceful fallback response
- **Response timing middleware** (`TimingMiddleware`) with NFR-02 breach alerts
- **Temperature 0.3** — reduces hallucination risk in medical context
- **Async Groq calls** via `asyncio.to_thread`
- **27 unit tests** passing with mocked Groq API
- **Current models**: `llama-3.3-70b-versatile` (main), `llama-3.1-8b-instant` (fast)

### Day 3: Conversation Memory & Context Management

- **Per-user conversation memory** via `HokuConversationMemory`
- **ConversationBufferMemory** with `return_messages=True` for `MessagesPlaceholder` compatibility
- **MessagesPlaceholder** injection into `ChatPromptTemplate` (reliable in langchain 0.2.6)
- **Token budget management** (`token_budget.py`) with tiktoken + Windows `len/4` fallback
- **History trimming** — drops oldest message pairs first when exceeding 307-token budget
- **Memory isolation** — User A can never see User B's chat history (verified via DB query filtering)
- **Complete-turn filtering** — only loads history where `ai_response IS NOT NULL`
- **Memory settings** in `config.py`: `MEMORY_MESSAGE_LIMIT=10`, `MEMORY_MAX_TOKENS=307`
- **Clean service separation**: `process_chat`, `classify_intent` (Day 4 placeholder), `generate_response`
- **Latency-safe memory loading** — memory load time deducted from 3.5s LLM timeout
- **11 new unit tests** in `test_memory.py` (memory load, token budget, per-user isolation, message objects)
- **Forced recall verification**: AI correctly answers "What symptoms did I mention?" by referencing DB-loaded history

---

## Clinical Safety

All AI responses include the mandatory disclaimer:
> **"Please consult a doctor for proper diagnosis."**

The chatbot never provides definitive diagnoses. Temperature is set to **0.3** to minimize hallucination while maintaining empathetic, natural language.

---

## Performance (NFR-02)

- **Target**: < 4 seconds per chat request
- **Hard timeout**: 3.5 seconds (fallback triggers automatically)
- **Max tokens**: 512 (keeps responses concise)
- **Memory limit**: 10 turns (keeps token budget and latency in check)
- **Monitoring**: Timing middleware logs latency and alerts on breaches

---

## Testing

### Unit Tests
```bash
pytest tests/test_chatbot.py -v
pytest tests/test_memory.py -v
```

### Full Suite
```bash
pytest tests/ -v
```

### Live API Test — Multi-Turn Conversation (Swagger UI)

1. Open [http://localhost:8000/docs](http://localhost:8000/docs)
2. Authenticate with a valid JWT token
3. Execute the following sequence:

**Turn 1 — Initial symptom:**
```json
{
  "message": "I have a headache and fever for 3 days",
  "userId": 1
}
```

**Turn 2 — Follow-up (proves memory):**
```json
{
  "message": "What symptoms did I mention in my previous message?",
  "userId": 1
}
```
*Expected: AI references "headache and fever for 3 days" from loaded history.*

**Turn 3 — Continuity:**
```json
{
  "message": "Is it normal to feel dizzy too?",
  "userId": 1
}
```
*Expected: AI acknowledges prior symptoms in context.*

**Turn 4 — Per-user isolation:**
```json
{
  "message": "What were my previous symptoms?",
  "userId": 2
}
```
*Expected: Generic response — no access to User 1's history.*

---

## Team

## Team

**This AI Chatbot Module** is developed and maintained by:
- **Ameema Rashid** — AI Lead Developer

**Overall Hoku Health Care Project:**
- **AI Lead**: Ameema Rashid
- **Backend Lead**: Muhammad Talha
- **Backend + AI**: Faisal Majeed

---

*Built with care for Hoku Health Care patients.*
