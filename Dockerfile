# Backend: FastAPI + the agent pipeline (GNN, RAG, LangGraph).
# Single stage -- the heavy deps (torch, torch-geometric, sentence-transformers) are the
# bulk of the image regardless, so a multi-stage split buys little here.
FROM python:3.12-slim

# psycopg2 + opencv need a few system libs at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so the (slow) wheel layer is cached across code changes.
# Generous timeout + retries: the torch/CUDA wheels are large and a slow/flaky
# connection otherwise hangs the build with no error.
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 600 --retries 10 -r requirements.txt

# App code (model checkpoints + RAG corpus are committed, so they ship in the image).
COPY . .

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["docker-entrypoint.sh"]
