.PHONY: setup db up down schema etl train eval api dash fmt

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

# Start Postgres via docker compose (db service only)
db:
	docker compose up -d db

up:
	docker compose up -d --build

down:
	docker compose down -v

schema:
	psql "$$DATABASE_URL" -f sql/schema.sql

etl:
	python src/etl.py

train:
	python src/train_multitask.py --epochs 3 --batch_size 16 --lr 2e-5 --max_len 192

eval:
	python src/eval.py

api:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

dash:
	streamlit run app/Dashboard.py

fmt:
	python -c 'print("(add black/ruff if desired)")'