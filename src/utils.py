import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from .config import DATABASE_URL, POSITIVE_LABELS, TOXIC_LABELS

engine = create_engine(DATABASE_URL, future=True)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def upsert_documents(df: pd.DataFrame, source: str):
    """Insert texts into documents and return list of new doc_ids. Expects 'text' column."""
    with engine.begin() as conn:
        doc_ids = []
        for txt in df["text"].astype(str).tolist():
            res = conn.execute(text("INSERT INTO documents (source, text) VALUES (:s, :t) RETURNING doc_id"),
                               {"s": source, "t": txt})
            doc_ids.append(res.scalar_one())
    return doc_ids

def insert_toxic(doc_ids, labels_df: pd.DataFrame):
    with engine.begin() as conn:
        for i, doc_id in enumerate(doc_ids):
            row = labels_df.iloc[i]
            conn.execute(text(
                """
                INSERT INTO toxic_labels (doc_id, toxic, severe_toxic, obscene, threat, insult, identity_hate)
                VALUES (:doc_id,:toxic,:severe,:obscene,:threat,:insult,:idh)
                ON CONFLICT (doc_id) DO NOTHING
                """),
                {
                    "doc_id": doc_id,
                    "toxic": bool(row.get("toxic",0)),
                    "severe": bool(row.get("severe_toxic",0)),
                    "obscene": bool(row.get("obscene",0)),
                    "threat": bool(row.get("threat",0)),
                    "insult": bool(row.get("insult",0)),
                    "idh": bool(row.get("identity_hate",0)),
                }
            )

def insert_emotions(doc_ids, labels_df: pd.DataFrame):
    with engine.begin() as conn:
        for i, doc_id in enumerate(doc_ids):
            row = labels_df.iloc[i]
            vals = {k: bool(row.get(k,0)) for k in POSITIVE_LABELS}
            cols = ','.join(POSITIVE_LABELS)
            placeholders = ','.join([':'+k for k in POSITIVE_LABELS])
            sql = text(f"INSERT INTO emotion_labels (doc_id,{cols}) VALUES (:doc_id,{placeholders}) ON CONFLICT (doc_id) DO NOTHING")
            params = {"doc_id": doc_id, **vals}
            conn.execute(sql, params)

def fetch_dataframe(sql: str) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn)

def compute_pos_weight(y: np.ndarray) -> np.ndarray:
    """Return per-label pos_weight = (N - pos) / pos to fight imbalance."""
    pos = y.sum(axis=0)
    neg = (len(y) - pos)
    return (neg / np.clip(pos, 1e-6, None)).astype("float32")