SHELL := /bin/bash

APP_MODULE := app.main:app
HOST := 0.0.0.0
PORT := 8000
COMPOSE := docker compose
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help venv install \
	run run-local local stop \
	build up down rebuild restart logs backend-logs redis-logs ps shell clean \
	docker-build docker-up docker-down docker-rebuild docker-logs freeze

help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make venv              Create local virtual environment (.venv)"
	@echo "  make install           Install backend dependencies into .venv"
	@echo ""
	@echo "Run app:"
	@echo "  make run               Run FastAPI locally with reload"
	@echo "  make stop              Stop local process on port $(PORT)"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build      Build containers"
	@echo "  make docker-up         Start containers"
	@echo "  make docker-down       Stop containers"
	@echo "  make docker-rebuild    Rebuild and start containers"
	@echo "  make docker-logs       Show all container logs"
	@echo "  make backend-logs      Show backend logs"
	@echo "  make redis-logs        Show redis logs"
	@echo "  make ps                Show running containers"
	@echo "  make shell             Open shell in backend container"
	@echo "  make clean             Remove stopped containers/orphans"

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r backend/requirements.txt

run: run-local
run-local:
	cd backend && REDIS_HOST=localhost PYTHONPATH=. ../$(VENV)/bin/uvicorn $(APP_MODULE) --host $(HOST) --port $${PORT:-$(PORT)} --reload
local: run-local

stop:
	@PID=$$(lsof -ti :$(PORT)); \
	if [ -n "$$PID" ]; then \
		kill -9 $$PID; \
		echo "Stopped process on port $(PORT)"; \
	else \
		echo "No process running on port $(PORT)"; \
	fi

build:
	$(COMPOSE) build
docker-build: build

up:
	$(COMPOSE) up -d
docker-up: up

down:
	$(COMPOSE) down
docker-down: down

rebuild:
	$(COMPOSE) up -d --build
docker-rebuild: rebuild

restart: down up

logs:
	$(COMPOSE) logs -f
docker-logs: logs

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

freeze:
	$(PIP) freeze > backend/requirements.txt
