#!/usr/bin/env bash
# One-time setup for macOS (Apple Silicon or Intel) -- runs the stack natively so the
# report LLM uses the Mac's Metal GPU via Ollama. Docker can't pass the Metal GPU into a
# container, so on Mac we skip Docker and use Homebrew services instead.
#
#   ./setup-mac.sh      # install deps, create DB, pull model, seed   (run once)
#   ./run-mac.sh        # start everything                            (run each time)
set -euo pipefail
cd "$(dirname "$0")"

echo "==> Checking Homebrew"
if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is required. Install it from https://brew.sh then re-run this script."
    exit 1
fi

echo "==> Installing Postgres, pgvector, Ollama (skips any already installed)"
brew install postgresql@16 pgvector ollama || true

echo "==> Starting Postgres and Ollama as background services"
brew services start postgresql@16
brew services start ollama
# Give Postgres a moment to accept connections.
until pg_isready -q 2>/dev/null; do sleep 1; done

DB_NAME=industrial_safety
echo "==> Creating database '$DB_NAME' (if absent) and enabling pgvector"
createdb "$DB_NAME" 2>/dev/null || echo "    database already exists"
psql -d "$DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS vector;" >/dev/null

echo "==> Writing .env (native Mac config)"
# Homebrew Postgres trusts the local user over the Unix socket, so no password is needed.
cat > .env <<EOF
DATABASE_URL=postgresql+psycopg2://$(whoami)@/$DB_NAME
OLLAMA_URL=http://localhost:11434
OLLAMA_NUM_GPU=999
API_KEY_REQUIRED=false
CORS_ORIGINS=*
EOF

echo "==> Pulling the report LLM (llama3.1:8b, ~5GB, first run only -- runs on Metal GPU)"
ollama pull llama3.1:8b

echo "==> Creating Python venv and installing backend deps (this is the slow step)"
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip >/dev/null
./.venv/bin/pip install -r requirements.txt

echo "==> Applying migrations and seeding the database"
./.venv/bin/alembic upgrade head
./.venv/bin/python -m db.seed || echo "    (seed reported an issue -- safe to re-run)"

echo "==> Installing frontend deps"
( cd frontend && npm install )

echo ""
echo "Setup complete. Start everything with:  ./run-mac.sh"
