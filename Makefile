SHELL := /bin/bash

.DEFAULT_GOAL := help

.PHONY: help up up-gpu down restart logs logs-backend wait ps build rebuild seed migrate hooks \
        mac-setup mac-run venv fmt-check clean-docker

help: ## Show this list
	@echo "Industrial Safety Intelligence -- common commands"
	@echo ""
	@awk 'BEGIN {FS = ":.*## "} /^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Mac users: Docker can't reach the Metal GPU or the built-in camera, so"
	@echo "'make mac-setup' + 'make mac-run' run the stack natively instead."

## --- Docker (Linux / Windows / WSL2, CPU or NVIDIA GPU) -----------------------------

up: ## Start the full stack (postgres, ollama, backend, frontend) -- CPU
	docker compose up -d
	@echo "Frontend: http://localhost:5173   Backend: http://localhost:8000"

up-gpu: ## Same as 'up' but with NVIDIA GPU passthrough (needs the NVIDIA Container Toolkit)
	./run-gpu.sh -d
	@echo "Frontend: http://localhost:5173   Backend: http://localhost:8000"

down: ## Stop the stack (keeps the postgres/ollama volumes)
	docker compose down

restart: ## Recreate every container from the current images (no rebuild)
	docker compose up -d --force-recreate

logs: ## Tail all container logs
	docker compose logs -f --tail 100

logs-backend: ## Tail just the backend log -- watch it wait for DB, migrate, seed, then serve
	docker compose logs -f --tail 100 backend

wait: ## Block until the backend API is actually ready to serve (polls /zones)
	@echo "Waiting for backend to become ready (migrations + seed + model load)..."
	@until curl -sf -o /dev/null http://localhost:8000/zones 2>/dev/null; do \
		printf '.'; sleep 2; \
	done; \
	echo ""; echo "Backend ready -> http://localhost:8000/docs"

ps: ## Show container status
	docker compose ps

build: ## Build (or rebuild) the backend + frontend images
	docker compose build

rebuild: ## Rebuild images (reusing the apt/pip layers) and restart -- picks up code changes
	docker compose build
	docker compose up -d --force-recreate

rebuild-clean: ## Full --no-cache rebuild (re-downloads apt/pip too) -- only if deps changed; network to apt/PyPI can be flaky here
	docker compose build --no-cache
	docker compose up -d --force-recreate

seed: ## Re-run the idempotent DB seed inside the running backend container
	docker compose exec backend python -m db.seed

migrate: ## Apply Alembic migrations inside the running backend container
	docker compose exec backend alembic upgrade head

clean-docker: ## Stop the stack AND delete its volumes (postgres data, ollama models) -- destructive
	docker compose down -v

## --- Native macOS (Metal GPU + real webcam access) -----------------------------------

mac-setup: ## One-time native macOS setup (Homebrew postgres/pgvector/ollama, venv, seed)
	./setup-mac.sh

mac-run: ## Run the stack natively on macOS (after mac-setup) -- Ctrl-C to stop
	./run-mac.sh

## --- Local dev (no Docker, backend already set up) -----------------------------------

venv: ## Create/update the local Python virtualenv from requirements.txt
	python3 -m venv .venv
	./.venv/bin/pip install -q --upgrade pip
	./.venv/bin/pip install -q -r requirements.txt

## --- Repo hygiene --------------------------------------------------------------------

hooks: ## Enable shared git hooks (strips AI co-author trailers from commit messages)
	git config core.hooksPath .githooks
	@echo "Hooks enabled. commit-msg will strip AI attribution trailers."
