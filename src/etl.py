"""ETL for Kaggle datasets -> Postgres.
Supports:
- GoEmotions (DeBarshi or Shivamb mirrors). We keep only POSITIVE_LABELS.
- Jigsaw Multilingual Toxic Comment Classification (train).
- Optional Twitter Emotion dataset.
"""
import pandas as pd
from .config import GOEMOTIONS_CSV, JIGSAW_TRAIN_CSV, TWITTER_EMOTION_CSV, POSITIVE_LABELS
from .utils import upsert_documents, insert_toxic, insert_emotions

# --- GoEmotions ---
def load_goemotions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize likely column names across mirrors
    if 'text' not in df.columns and 'comment_text' in df.columns:
        df = df.rename(columns={'comment_text':'text'})
    present = [c for c in POSITIVE_LABELS if c in df.columns]
    if not present:
        raise ValueError("GoEmotions CSV missing expected emotion columns.")
    keep = ['text'] + present
    out = df[keep].copy()
    out[present] = out[present].fillna(0).astype(int)
    return out

# --- Jigsaw Multilingual Toxic ---
def load_jigsaw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={'comment_text':'text'})
    needed = ['text','toxic','severe_toxic','obscene','threat','insult','identity_hate']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Jigsaw CSV missing columns: {missing}")
    out = df[needed].copy()
    out[[c for c in needed if c!='text']] = out[[c for c in needed if c!='text']].fillna(0).astype(int)
    return out

# --- Twitter Emotion (optional; map to positivity if possible) ---
def load_twitter_emotion(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={c:c.lower() for c in df.columns})
    if not {'text','emotion'} <= set(df.columns):
        raise ValueError('Twitter emotion CSV must have text and emotion columns')
    mapping = {
        'joy':'joy', 'love':'love', 'surprise':'optimism',
        'admiration':'admiration', 'gratitude':'gratitude', 'pride':'pride'
    }
    rows = []
    for _, r in df.iterrows():
        row = {k:0 for k in POSITIVE_LABELS}
        m = mapping.get(str(r['emotion']).lower())
        if m and isinstance(r.get('text',''), str):
            row[m] = 1
            row['text'] = r['text']
            rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    from .config import DATABASE_URL
    print("Loading data into:", DATABASE_URL)

    ge = load_goemotions(GOEMOTIONS_CSV)
    ge_ids = upsert_documents(ge[['text']].copy(), source='goemotions')
    insert_emotions(ge_ids, ge[[c for c in ge.columns if c in POSITIVE_LABELS]])
    print(f"Inserted GoEmotions: {len(ge_ids)}")

    jg = load_jigsaw(JIGSAW_TRAIN_CSV)
    jg_ids = upsert_documents(jg[['text']].copy(), source='jigsaw')
    insert_toxic(jg_ids, jg.drop(columns=['text']))
    print(f"Inserted Jigsaw: {len(jg_ids)}")

    if TWITTER_EMOTION_CSV:
        tw = load_twitter_emotion(TWITTER_EMOTION_CSV)
        if not tw.empty:
            tw_ids = upsert_documents(tw[['text']].copy(), source='twitter_emotion')
            insert_emotions(tw_ids, tw[[c for c in tw.columns if c in POSITIVE_LABELS]])
            print(f"Inserted Twitter emotions: {len(tw_ids)}")