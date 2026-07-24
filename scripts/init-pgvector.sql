-- Hoku Health Care - Postgres bootstrap (Day 10)
-- Runs once, on first container start, via docker-entrypoint-initdb.d.
-- app/ai/rag.py checks for this extension to decide between the native
-- pgvector cosine-distance path and the in-Python fallback scan.
CREATE EXTENSION IF NOT EXISTS vector;