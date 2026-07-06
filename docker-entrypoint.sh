#!/usr/bin/env bash
# Wait for Postgres, apply migrations, seed (idempotent), then start the API.
# This is what makes `docker compose up` a true one-command run with data ready.
set -e

echo "Waiting for Postgres at ${POSTGRES_HOST:-postgres}:${POSTGRES_PORT:-5432}..."
until python -c "
import socket, os, sys
s = socket.socket()
s.settimeout(2)
try:
    s.connect((os.environ.get('POSTGRES_HOST', 'postgres'), int(os.environ.get('POSTGRES_PORT', '5432'))))
    sys.exit(0)
except Exception:
    sys.exit(1)
"; do
    echo "  ...postgres not ready, retrying"
    sleep 2
done
echo "Postgres is up."

echo "Applying migrations..."
alembic upgrade head

echo "Seeding (idempotent)..."
python -m db.seed || echo "Seed step reported an issue (continuing -- it's safe to re-run)."

# Pre-warm the RAG embedding model so the first /run doesn't pay the load mid-request.
# Uses the retriever's local-first logic: the vendored copy in models/embedding ships in
# the image, so this needs no network. Failure is non-fatal.
echo "Pre-warming embedding model..."
python -c "from rag.retriever import _lazy_load_model; _lazy_load_model(); print('embedding model ready')" \
    || echo "WARNING: embedding model failed to load -- reports will retry at first use."

echo "Starting API on :8000"
exec uvicorn server.main:app --host 0.0.0.0 --port 8000
