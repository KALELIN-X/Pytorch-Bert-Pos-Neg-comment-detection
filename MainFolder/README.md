# CivicBalance — Promote Good, Reduce Harm

End-to-end NLP project: load Kaggle datasets into Postgres, analyze with SQL, fine-tune a **multi-task BERT** (toxicity + positive emotions) with PyTorch, then serve predictions via FastAPI and a Streamlit dashboard. Includes a single **Net Goodness Score (NGS)** to rank text.

## Datasets you chose
- GoEmotions (either DeBarshi or Shivamb mirror)
- Jigsaw Multilingual Toxic Comment Classification
- (Optional) Twitter Emotion Dataset (Parul Pandey)

## Quickstart (Local)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit paths and DB creds

# Start Postgres (either your local PG or docker-compose below)
# Example: docker compose up -d db

# Initialize DB schema and load data
psql "$DATABASE_URL" -f sql/schema.sql
python src/etl.py

# (Optional) SQL analytics and plots
python src/query_examples.py

# Train multi-task BERT
python src/train_multitask.py --epochs 3 --batch_size 16 --lr 2e-5 --max_len 192

# Evaluate & sample predictions
python src/eval.py

# Serve API & launch dashboard (use two terminals)
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
streamlit run app/Dashboard.py