# Hoku Health Care AI chatbot: container image for Google Cloud Run (and local Docker).
#
# - CPU-only torch, installed in its own layer from the PyTorch CPU index.
# - Runtime dependencies only: requirements-dev.txt is never installed.
# - Runs as a non-root user (uid 1000) and listens on $PORT (default 8080).
# - The embedding model and the tiktoken encoding are downloaded into the image
#   BEFORE the app code is copied (code changes reuse those layers), and
#   Hugging Face is offline at runtime.
# - No secret is in the image. Outside ENVIRONMENT=test the app refuses to
#   start without a SECRET_KEY of 32+ characters (Secret Manager on Cloud Run).
# - With DB_AUTO_MIGRATE=true (default) every start runs init_db.py first:
#   Alembic migrations plus idempotent doctor and FAQ seeding.

FROM python:3.11-slim

LABEL org.opencontainers.image.title="hoku-health-ai" \
      org.opencontainers.image.description="Hoku Health Care AI chatbot (FastAPI + Groq + LangChain)" \
      org.opencontainers.image.source="https://github.com/ameema28/hoku-health-ai"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /app

# Layer 1: CPU-only torch from the PyTorch CPU index only. The torch==2.15.0
# pin in requirements.txt is then already satisfied, so pip keeps this build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0

# Layer 2: the other runtime dependencies (all have binary wheels, so no
# compiler or libpq is needed). constraints.txt locks every transitive
# version, so a rebuild resolves the same tree the tests ran on.
COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt

# Non-root runtime user.
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin hoku

USER hoku
ENV HOME=/home/hoku \
    HF_HOME=/home/hoku/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/hoku/.cache/huggingface \
    TIKTOKEN_CACHE_DIR=/home/hoku/.cache/tiktoken

# Layer 3: the embedding model (same name as EMBEDDING_MODEL) and the
# tiktoken encoding, downloaded before any app code is copied.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
 && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

USER root

# Application code is owned by root, so the runtime user can only read it.
COPY --chown=root:root alembic.ini init_db.py ./
COPY --chown=root:root alembic ./alembic
COPY --chown=root:root app ./app

USER hoku

# Runtime defaults. Without DATABASE_URL the database is SQLite in /tmp,
# which is reset whenever the container restarts. Demo login stays off
# unless enabled.
ENV PORT=8080 \
    ENVIRONMENT=production \
    DEBUG=false \
    LOG_FORMAT=json \
    DATABASE_URL=sqlite:////tmp/hoku.db \
    DB_AUTO_MIGRATE=true \
    DEMO_LOGIN_ENABLED=false

EXPOSE 8080

# One worker: the rate limiters, response cache and metrics are in memory,
# per process. --no-proxy-headers keeps request.client as the socket peer;
# TRUST_PROXY decides whether X-Forwarded-For is used (app/core/client_ip.py).
CMD ["sh", "-c", "if [ \"$DB_AUTO_MIGRATE\" = \"true\" ]; then python init_db.py || exit 1; fi; exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --no-proxy-headers"]
