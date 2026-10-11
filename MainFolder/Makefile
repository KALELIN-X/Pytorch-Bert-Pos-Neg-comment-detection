# Use from the repository root: make -f MainFolder/Makefile <target>
# Or from MainFolder: make <target>. Windows users can use README Python commands.
REPO_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
PYTHON ?= python
COMPOSE = docker compose -f "$(REPO_ROOT)/MainFolder/docker-compose.yml"

.PHONY: setup training-deps models db up down schema etl train eval api dash test

setup:
	cd "$(REPO_ROOT)" && $(PYTHON) -m pip install -r requirements.txt

training-deps:
	cd "$(REPO_ROOT)" && $(PYTHON) -m pip install -r requirements-training.txt

models:
	cd "$(REPO_ROOT)" && $(PYTHON) -m src.download_models
	cd "$(REPO_ROOT)" && $(PYTHON) -m src.download_sentiment

db:
	$(COMPOSE) --profile database up -d db

up:
	$(COMPOSE) up -d --build app

down:
	$(COMPOSE) down

schema:
	psql "$$DATABASE_URL" -f "$(REPO_ROOT)/MainFolder/sql/schema.sql"

etl:
	cd "$(REPO_ROOT)" && $(PYTHON) -m src.etl

train:
	cd "$(REPO_ROOT)" && $(PYTHON) -m src.train_multitask

eval:
	cd "$(REPO_ROOT)" && $(PYTHON) -m src.eval

api:
	cd "$(REPO_ROOT)" && $(PYTHON) -m uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload

dash:
	cd "$(REPO_ROOT)" && $(PYTHON) -m streamlit run MainFolder/app/Dashboard.py

test:
	cd "$(REPO_ROOT)" && $(PYTHON) -m pytest
