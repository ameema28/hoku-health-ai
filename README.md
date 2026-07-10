# Hoku Health Care - AI Chatbot Module

**TechNexus Virtual University | Internship Project**

This module provides the AI-powered health chatbot backend for Hoku Health Care,
a home healthcare platform serving patients in Pakistan, UAE, and UK.

## Tech Stack

- **Framework**: FastAPI (Python)
- **AI/LLM**: Groq API (Llama 3 / Mixtral) via LangChain
- **Database**: PostgreSQL + SQLAlchemy + Alembic
- **Auth**: JWT (stubbed for AI module setup)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2) — stubbed for RAG

## Quick Start

### 1. Prerequisites

- Python 3.10+
- PostgreSQL 14+ running locally
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
# Edit .env and fill in your DATABASE_URL and GROQ_API_KEY
```

### 4. Database Setup

```bash
# Run migrations
alembic upgrade head
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
| POST | `/api/ai/chat` | AI Health Chatbot |
| GET | `/api/ai/health` | Service Health Check |

## Project Structure

```
hoku-health-backend/
├── app/
│   ├── core/          # Config, DB, security, dependencies
│   ├── models/        # SQLAlchemy models
│   ├── schemas/       # Pydantic schemas
│   ├── api/           # API routers
│   ├── ai/            # Chatbot engine
│   ├── services/      # Business logic layer
│   ├── utils/         # Constants & validators
│   └── middleware/    # CORS & error handlers
├── alembic/           # Database migrations
├── requirements.txt
└── .env.example
```

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
