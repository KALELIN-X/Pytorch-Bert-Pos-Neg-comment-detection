"""Optional validated CSV -> Postgres ETL from the original project.

Use Jigsaw 2018's six binary toxicity columns and GoEmotions expanded into
binary emotion columns. One-label multilingual Jigsaw and label-ID lists
are incompatible formats and are rejected instead of inventing labels.
"""

try:
    import pandas as pd
except ImportError:
    raise SystemExit("ETL requires optional dependencies: python -m pip install -r requirements-training.txt") from None

from .config import GOEMOTIONS_CSV, JIGSAW_TRAIN_CSV, TWITTER_EMOTION_CSV, POSITIVE_LABELS, TOXIC_LABELS
from .utils import upsert_documents, insert_toxic, insert_emotions


def _load_labeled_csv(path, labels, name):
    try:
        frame = pd.read_csv(path)
    except FileNotFoundError:
        raise ValueError(f"{name} CSV is missing. Set its configured CSV path to a local file.") from None
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        raise ValueError(f"{name} CSV is empty or malformed.") from None
    if "text" not in frame.columns and "comment_text" in frame.columns:
        frame = frame.rename(columns={"comment_text": "text"})
    required = ["text", *labels]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{name} CSV missing columns: {', '.join(missing)}. "
            "Provide text and every expected binary label column; missing labels are not filled with zero."
        )
    out = frame[required].copy()
    if out.empty:
        raise ValueError(f"{name} CSV must contain at least one row.")
    valid_text = out["text"].map(lambda value: isinstance(value, str) and bool(value.strip()))
    if not valid_text.all():
        raise ValueError(f"{name} CSV contains missing, non-text, or blank comments.")
    numeric = out[labels].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not numeric.isin([0, 1]).all().all():
        raise ValueError(f"{name} labels must contain only binary 0/1 values without missing entries.")
    out[labels] = numeric.astype("int64")
    normalized = out["text"].map(lambda value: " ".join(value.split()).casefold())
    if (out.groupby(normalized)[labels].nunique() > 1).any().any():
        raise ValueError(f"{name} CSV assigns conflicting labels to duplicate normalized comments.")
    return out


def load_goemotions(path: str) -> pd.DataFrame:
    return _load_labeled_csv(path, POSITIVE_LABELS, "GoEmotions (expanded emotion columns)")


def load_jigsaw(path: str) -> pd.DataFrame:
    return _load_labeled_csv(path, TOXIC_LABELS, "Jigsaw 2018 (six-label toxicity)")


def load_twitter_emotion(path: str) -> pd.DataFrame:
    """Historical optional mapping; surprise is no longer assumed to be optimism."""
    frame = pd.read_csv(path)
    frame = frame.rename(columns={column: column.lower() for column in frame.columns})
    if not {"text", "emotion"} <= set(frame.columns):
        raise ValueError("Twitter emotion CSV must have text and emotion columns.")
    rows = []
    for _, record in frame.iterrows():
        emotion = str(record["emotion"]).lower()
        comment = record["text"]
        if emotion in POSITIVE_LABELS and isinstance(comment, str) and comment.strip():
            row = {label: int(label == emotion) for label in POSITIVE_LABELS}
            rows.append({"text": comment, **row})
    return pd.DataFrame(rows, columns=["text", *POSITIVE_LABELS])


def main():
    # Validate mandatory datasets before inserting anything.
    ge = load_goemotions(GOEMOTIONS_CSV)
    jg = load_jigsaw(JIGSAW_TRAIN_CSV)
    twitter = load_twitter_emotion(TWITTER_EMOTION_CSV) if TWITTER_EMOTION_CSV else None
    print("Loading validated local datasets into configured Postgres database.")
    try:
        ge_ids = upsert_documents(ge[["text"]], source="goemotions")
        insert_emotions(ge_ids, ge[POSITIVE_LABELS])
        print(f"Inserted GoEmotions: {len(ge_ids)}")
        jg_ids = upsert_documents(jg[["text"]], source="jigsaw")
        insert_toxic(jg_ids, jg[TOXIC_LABELS])
        print(f"Inserted Jigsaw: {len(jg_ids)}")
        if twitter is not None and not twitter.empty:
            ids = upsert_documents(twitter[["text"]], source="twitter_emotion")
            insert_emotions(ids, twitter[POSITIVE_LABELS])
            print(f"Inserted Twitter emotions: {len(ids)}")
    except Exception:
        # Never print driver exceptions containing passwords or private SQL values.
        raise RuntimeError(
            "Postgres import failed. Check dependencies, connectivity and schema. "
            "The legacy importer is append-only; partial imports may need manual cleanup before retrying."
        ) from None


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from None
