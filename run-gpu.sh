#!/usr/bin/env bash
# Convenience: run the stack with GPU passthrough (needs NVIDIA Container Toolkit).
# Falls back is manual -- if this errors about nvidia devices, use plain `docker compose up`.
exec docker compose -f docker-compose.yml -f docker-compose.gpu.yml up "$@"
