# Hoku Health Care - AI Chatbot Module

**TechNexus Virtual University | Internship Project**

This module provides the AI-powered health chatbot backend for Hoku Health Care,
a home healthcare platform serving patients in Pakistan, UAE, and UK.

## Tech Stack

- **Framework**: FastAPI (Python)
- **AI/LLM**: Groq API (Llama 3 / Mixtral) via LangChain
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

**For local development without PostgreSQL**, change `DATABASE_URL` in `.env`:
```bash
DATABASE_URL=sqlite:///./hoku_health.db
```

### 4. Run the CRUD Test (No Server Needed)

```bash
python test_crud.py
# Expected output: "All CRUD tests passed"
```

This test uses SQLite and verifies all database operations work correctly.

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
| POST | `/api/ai/chat` | AI Health Chatbot |
| GET | `/api/ai/chat/history` | Chat History (paginated) |
| GET | `/api/ai/health` | Service Health Check |

## Project Structure

```
hoku-health-backend/
├── alembic/           # Database migrations
│   └── versions/
│       └── 001_create_chat_history.py
├── app/
│   ├── ai/            # Chatbot engine (Groq LLM)
│   ├── api/           # API routers
│   │   └── v1/
│   │       └── endpoints/
│   │           └── ai.py
│   ├── core/          # Config, DB, security, dependencies
│   ├── crud/          # Database access layer
│   │   └── chat.py
│   ├── middleware/    # CORS & error handlers
│   ├── models/        # SQLAlchemy models
│   ├── schemas/       # Pydantic schemas
│   ├── services/      # Business logic layer
│   └── utils/         # Constants & validators
├── .env.example
├── requirements.txt
└── test_crud.py       # CRUD verification script
```

## Day 1 Deliverables

- **SQLAlchemy 2.0** ChatHistory model with indexes
- **CRUD layer** with atomic transactions and logging
- **GET /api/ai/chat/history** endpoint with pagination
- **Custom exceptions** (UserNotFoundException, DatabaseOperationException)
- **Input validators** (sanitize_message, validate_message_length)
- **SQLite test script** for offline verification

## Clinical Safety

All AI responses include the mandatory disclaimer:
> **"Please consult a doctor for proper diagnosis."**

The chatbot never provides definitive diagnoses.

## Team

- **AI Lead**: Ameema Rashid
- **Backend Lead**: Muhammad Talha
- **Backend + AI**: Faisal Majeed

---

*Built with care for Hoku Health Care patients.*
