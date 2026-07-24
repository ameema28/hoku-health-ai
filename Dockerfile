# =============================================================================
# Hoku Health Care - AI Chatbot Backend (Day 10: Production Deployment)
#
# Multi-stage build:
#   Stage 1 (builder)  - compiles wheels into an isolated virtualenv
#   Stage 2 (runtime)  - slim image, non-root uid 1000, no build toolchain
#
# Design notes:
#   * python:3.11-slim keeps the base ~120MB; the virtualenv is copied
#     wholesale so no compiler ever lands in the runtime layer.
#   * PRELOAD_EMBEDDINGS bakes sentence-transformers/all-MiniLM-L6-v2
#     (~90MB) into the image so the first patient request never pays a
#     cold-start model download. Set to "false" for Render's 512MB free
#     tier if the image exceeds the disk quota - RAG then degrades
#     gracefully to zero-vectors (see app/ai/embeddings.py).
#   * HEALTHCHECK targets /api/ai/health, which is unauthenticated by
#     design (app/api/v1/endpoints/ai.py).
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ARG PRELOAD_EMBEDDINGS=true

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# psycopg2-binary ships wheels, but bcrypt/tiktoken may need a compiler on
# some architectures. libpq-dev is required for any psycopg2 source build.
RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# Pre-download the RAG embedding model into a cache directory that the
# runtime stage copies to the non-root user's HOME.
RUN mkdir -p /opt/model-cache
ENV HF_HOME=/opt/model-cache \
    SENTENCE_TRANSFORMERS_HOME=/opt/model-cache
RUN if [ "$PRELOAD_EMBEDDINGS" = "true" ]; then \
        python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" ; \
    else \
        echo "Skipping embedding model preload (PRELOAD_EMBEDDINGS=$PRELOAD_EMBEDDINGS)" ; \
    fi

# -----------------------------------------------------------------------------
# Stage 2: runtime
# -----------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="hoku-health-backend" \
      org.opencontainers.image.description="Hoku Health Care AI Chatbot (FastAPI + Groq + LangChain)" \
      org.opencontainers.image.vendor="TechNexus Virtual University" \
      org.opencontainers.image.source="https://github.com/technexus/hoku-health-backend"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/home/hoku/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/hoku/.cache/huggingface \
    ENVIRONMENT=production \
    PORT=8000

# libpq5 is the only runtime shared library psycopg2 needs.
# curl is used by HEALTHCHECK.
RUN apt-get update && apt-get install --no-install-recommends -y \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user, fixed uid/gid 1000 so bind-mounted volumes keep sane
# ownership in docker-compose and on Render.
RUN groupadd --gid 1000 hoku \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin hoku

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=1000:1000 /opt/model-cache /home/hoku/.cache/huggingface

WORKDIR /app

COPY --chown=1000:1000 alembic.ini pytest.ini ./
COPY --chown=1000:1000 alembic/ ./alembic/
COPY --chown=1000:1000 app/ ./app/

USER 1000

EXPOSE 8000

# The container is healthy only when the AI service reports "ok".
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail --silent http://localhost:${PORT}/api/ai/health || exit 1

# Single worker by default: HokuMetrics, ResponseCache and the Prometheus
# registry are all in-process singletons. Scale horizontally (more
# containers), not with --workers, or the metrics will fragment.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --proxy-headers --forwarded-allow-ips '*'"]