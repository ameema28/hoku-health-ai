# Hoku Health Care - AI Chatbot Module

**TechNexus Virtual University | Internship Project**

This module provides the AI-powered health chatbot backend for Hoku Health Care,
a home healthcare platform serving patients in Pakistan, UAE, and UK.

## Tech Stack

- **Framework**: FastAPI (Python)
- **AI/LLM**: Groq API (Llama 3) via LangChain
- **Database**: PostgreSQL + SQLAlchemy 2.0 + Alembic
- **Auth**: JWT (stubbed for AI module setup)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2) — stubbed for RAG

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

**Required `.env` variables for Day 2:**
```bash
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_FAST_MODEL=llama-3.1-8b-instant
GROQ_MAIN_MODEL=llama-3.3-70b-versatile
TEMPERATURE=0.3
MAX_TOKENS=512
GROQ_TIMEOUT_SECONDS=3.5
MAX_RETRIES=3
RETRY_BACKOFF_BASE_SECONDS=1.0
```

**For local development without PostgreSQL**, change `DATABASE_URL` in `.env`:
```bash
DATABASE_URL=sqlite:///./hoku_health.db
```

### 4. Run Tests

```bash
# Unit tests (mocked Groq — no API key needed)
pytest tests/test_chatbot.py -v

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

## AI Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/ai/chat` | AI Health Chatbot (Groq LLM) |
| GET | `/api/ai/chat/history` | Chat History (paginated) |
| GET | `/api/ai/health` | Service Health Check |

## Project Structure

```
hoku-health-backend/
├── alembic/              # Database migrations
├── app/
│   ├── ai/               # Chatbot engine (Groq + LangChain)
│   │   ├── __init__.py
│   │   ├── config.py     # AI hyperparameters & timeouts
│   │   ├── prompts.py    # Clinical safety prompts
│   │   ├── chatbot.py    # HokuChatbot class
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

## Day 1 Deliverables

- **SQLAlchemy 2.0** ChatHistory model with indexes
- **CRUD layer** with atomic transactions and logging
- **GET /api/ai/chat/history** endpoint with pagination
- **Custom exceptions** (UserNotFoundException, DatabaseOperationException)
- **Input validators** (sanitize_message, validate_message_length)
- **SQLite test script** for offline verification

## Day 2 Deliverables

- **Groq API integration** via LangChain `ChatGroq`
- **LLMChain** with clinical safety system prompt
- **Structured JSON output** parsing (reply, specialist, severity, shouldSeeDoctor)
- **3.5s hard timeout** with graceful fallback
- **Response timing middleware** with NFR-02 breach alerts
- **Temperature 0.3** — reduces hallucination in medical context
- **Comprehensive unit tests** with mocked Groq API (27 tests passing)
- **Async Groq calls** via `asyncio.to_thread`
- **Current Groq models**: `llama-3.3-70b-versatile` (main), `llama-3.1-8b-instant` (fast)

## Clinical Safety

All AI responses include the mandatory disclaimer:
> **"Please consult a doctor for proper diagnosis."**

The chatbot never provides definitive diagnoses. Temperature is set to **0.3** to minimize hallucination while maintaining empathetic, natural language.

## Performance (NFR-02)

- **Target**: < 4 seconds per chat request
- **Hard timeout**: 3.5 seconds (fallback triggers automatically)
- **Max tokens**: 512 (keeps responses concise)
- **Monitoring**: Timing middleware logs latency and alerts on breaches

## Testing

### Unit Tests
```bash
pytest tests/test_chatbot.py -v
```

### Live API Test (with server running)
```bash
# Health check
curl http://localhost:8000/api/ai/health

# Chat (replace YOUR_JWT_TOKEN with valid token)
curl -X POST "http://localhost:8000/api/ai/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "message": "I have chest pain radiating to my left arm",
    "userId": 1
  }'
```

## Team

- **AI Lead**: Ameema Rashid
- **Backend Lead**: Muhammad Talha
- **Backend + AI**: Faisal Majeed

---

*Built with care for Hoku Health Care patients.*
