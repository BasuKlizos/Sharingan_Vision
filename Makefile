# =========================
# Project Config
# =========================
SHELL := /bin/bash

APP_MODULE := app.main:app
HOST := 0.0.0.0
PORT := 8000

COMPOSE := docker compose

.PHONY: help install run run-local dev stop restart clean logs shell \
        docker-up docker-down docker-build docker-rebuild docker-logs \
        redis-logs backend-logs ps lint format freeze

# =========================
# Help
# =========================
help:
	@echo "Available commands:"
	@echo ""
	@echo "Local:"
	@echo "  make install        Install Python dependencies locally"
	@echo "  make run            Run FastAPI locally"
	@echo "  make run-local      Run FastAPI locally"
	@echo "  make dev            Run FastAPI locally with reload"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   Build containers"
	@echo "  make docker-up      Start containers"
	@echo "  make docker-down    Stop containers"
	@echo "  make docker-rebuild Rebuild and start containers"
	@echo "  make docker-logs    Show all container logs"
	@echo "  make backend-logs   Show backend logs"
	@echo "  make redis-logs     Show redis logs"
	@echo "  make ps             Show running containers"
	@echo "  make shell          Open shell in backend container"
	@echo ""
	@echo "Utilities:"
	@echo "  make stop           Stop local uvicorn on port 8000"
	@echo "  make restart        Restart docker containers"
	@echo "  make clean          Remove stopped containers"
	@echo "  make freeze         Freeze dependencies"

# =========================
# Local Run
# =========================
install:
	pip install -r backend/requirements.txt

run: run-local

# Local run (override Redis host)
local:
	cd backend && REDIS_HOST=localhost PYTHONPATH=. uvicorn $(APP_MODULE) --host $(HOST) --port $${PORT:-$(PORT)}  --reload
stop:
	@PID=$$(lsof -ti :$(PORT)); \
	if [ -n "$$PID" ]; then \
		kill -9 $$PID; \
		echo "Stopped process on port $(PORT)"; \
	else \
		echo "No process running on port $(PORT)"; \
	fi

# =========================
# Docker
# =========================
build:
	$(COMPOSE) build

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

rebuild:
	$(COMPOSE) up -d --build

restart: docker-down docker-up

logs:
	$(COMPOSE) logs -f

backend-logs:
	$(COMPOSE) logs -f backend

redis-logs:
	$(COMPOSE) logs -f redis

ps:
	$(COMPOSE) ps

shell:
	$(COMPOSE) exec backend /bin/bash

clean:
	$(COMPOSE) down --remove-orphans

# =========================
# Dev Helpers
# =========================
lint:
	python -m py_compile backend/app/main.py

format:
	@echo "Add formatter like black or ruff"

freeze:
	pip freeze > backend/requirements.txt