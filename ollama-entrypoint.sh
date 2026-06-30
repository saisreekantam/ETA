#!/usr/bin/env bash
# Start the Ollama server, then ensure the model the pipeline uses is pulled.
# The model is stored in the mounted volume, so the ~5GB pull only happens once.
set -e

MODEL="${OLLAMA_MODEL:-llama3.1:8b}"

ollama serve &
SERVER_PID=$!

echo "Waiting for Ollama to come up..."
until curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; do
    sleep 1
done

if ! ollama list | grep -q "${MODEL%%:*}"; then
    echo "Pulling ${MODEL} (first run only, ~5GB)..."
    ollama pull "${MODEL}"
else
    echo "${MODEL} already present."
fi

echo "Ollama ready."
wait $SERVER_PID
