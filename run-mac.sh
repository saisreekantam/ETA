#!/usr/bin/env bash
# Start the stack natively on macOS (after ./setup-mac.sh). The LLM runs on the Metal GPU
# via the native Ollama service. Ctrl-C stops the backend + frontend; the Postgres/Ollama
# brew services keep running (stop them with: brew services stop postgresql@16 ollama).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo "No .venv found -- run ./setup-mac.sh first."
    exit 1
fi

echo "==> Ensuring Postgres and Ollama services are up"
brew services start postgresql@16 >/dev/null 2>&1 || true
brew services start ollama >/dev/null 2>&1 || true
until pg_isready -q 2>/dev/null; do sleep 1; done

echo "==> Starting backend on http://localhost:8000"
./.venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "==> Starting frontend on http://localhost:5173"
( cd frontend && npm run dev ) &
FRONTEND_PID=$!

# Clean up both child processes on Ctrl-C.
trap 'echo; echo "Stopping..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true; exit 0' INT TERM

echo ""
echo "Backend:  http://localhost:8000      Frontend: http://localhost:5173"
echo "Press Ctrl-C to stop."
wait
