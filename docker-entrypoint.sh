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

echo "Starting API on :8000"
exec uvicorn server.main:app --host 0.0.0.0 --port 8000
