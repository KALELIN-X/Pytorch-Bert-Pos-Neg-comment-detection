#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp -n .env.example .env || true

# Start DB if using docker-compose
if command -v docker >/dev/null 2>&1; then
  docker compose up -d db
fi

psql "$DATABASE_URL" -f sql/schema.sql
python src/etl.py
python src/train_multitask.py --epochs 3 --batch_size 16 --lr 2e-5 --max_len 192
python src/eval.py

echo "\n✅ Setup complete. Start services in two shells:"
echo "  uvicorn src.api:app --port 8000 --reload"
echo "  streamlit run app/Dashboard.py"