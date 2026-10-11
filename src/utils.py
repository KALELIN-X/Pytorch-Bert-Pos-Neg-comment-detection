"""Helpers for optional training and Postgres; no database connection at import."""

import os
from typing import TYPE_CHECKING

from .config import DATABASE_URL, POSITIVE_LABELS, TOXIC_LABELS

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

_engine = None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _get_engine():
    global _engine
    if _engine is None:
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:
            raise RuntimeError("Postgres tools require: python -m pip install -r requirements-training.txt") from exc
        try:
            _engine = create_engine(DATABASE_URL, future=True)
        except Exception:
            # Driver errors can contain a URL/password.
            raise RuntimeError("Could not configure Postgres. Check DATABASE_URL and database dependencies.") from None
    return _engine


def upsert_documents(df: "pd.DataFrame", source: str):
    """Insert texts and return IDs. The legacy name does not imply deduplication."""
    engine = _get_engine()
    from sqlalchemy import text

    with engine.begin() as conn:
        doc_ids = []
        for txt in df["text"].tolist():
            result = conn.execute(
                text("INSERT INTO documents (source, text) VALUES (:s, :t) RETURNING doc_id"),
                {"s": source, "t": txt},
            )
            doc_ids.append(result.scalar_one())
    return doc_ids


def insert_toxic(doc_ids, labels_df: "pd.DataFrame"):
    if len(doc_ids) != len(labels_df):
        raise ValueError("Document IDs and toxicity label rows must have equal length.")
    engine = _get_engine()
    from sqlalchemy import text

    with engine.begin() as conn:
        for i, doc_id in enumerate(doc_ids):
            row = labels_df.iloc[i]
            conn.execute(
                text("""
                    INSERT INTO toxic_labels (doc_id, toxic, severe_toxic, obscene, threat, insult, identity_hate)
                    VALUES (:doc_id,:toxic,:severe_toxic,:obscene,:threat,:insult,:identity_hate)
                    ON CONFLICT (doc_id) DO NOTHING
                """),
                {"doc_id": doc_id, **{name: bool(row[name]) for name in TOXIC_LABELS}},
            )


def insert_emotions(doc_ids, labels_df: "pd.DataFrame"):
    if len(doc_ids) != len(labels_df):
        raise ValueError("Document IDs and emotion label rows must have equal length.")
    engine = _get_engine()
    from sqlalchemy import text

    columns = ",".join(POSITIVE_LABELS)
    placeholders = ",".join(":" + name for name in POSITIVE_LABELS)
    statement = text(
        f"INSERT INTO emotion_labels (doc_id,{columns}) VALUES (:doc_id,{placeholders}) "
        "ON CONFLICT (doc_id) DO NOTHING"
    )
    with engine.begin() as conn:
        for i, doc_id in enumerate(doc_ids):
            row = labels_df.iloc[i]
            conn.execute(statement, {"doc_id": doc_id, **{name: bool(row[name]) for name in POSITIVE_LABELS}})


def fetch_dataframe(sql: str) -> "pd.DataFrame":
    import pandas as pd

    engine = _get_engine()
    from sqlalchemy import text

    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn)


def compute_pos_weight(y: "np.ndarray") -> "np.ndarray":
    """Negative/positive counts, requiring both classes for every training label."""
    import numpy as np

    y = np.asarray(y, dtype="float32")
    if y.ndim != 2 or not len(y) or not y.shape[1]:
        raise ValueError("Training labels must be a nonempty two-dimensional matrix.")
    if not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Training labels must contain only binary 0/1 values.")
    positives = y.sum(axis=0)
    negatives = len(y) - positives
    if np.any(positives == 0) or np.any(negatives == 0):
        raise ValueError(
            "Every label needs positive and negative examples in the training split. "
            "Supply a larger representative dataset; no label is silently synthesized."
        )
    return (negatives / positives).astype("float32")


def split_indices_by_source(frame: "pd.DataFrame", validation_fraction=0.1, seed=42):
    """Seeded split of normalized-text groups, shared across both data sources.

    Identical comments (case/whitespace ignored) cannot cross the boundary.
    The requested fraction is a fraction of text groups, not necessarily rows.
    """
    import numpy as np

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)
    normalized = frame["text"].map(lambda value: " ".join(value.split()).casefold())
    groups = normalized.drop_duplicates().to_numpy(copy=True)
    if len(groups) < 2:
        raise ValueError("At least two distinct normalized comments are needed for a train/validation split.")
    rng.shuffle(groups)
    n_validation = max(1, min(len(groups) - 1, int(len(groups) * validation_fraction)))
    validation_mask = normalized.isin(groups[:n_validation]).to_numpy()
    training = np.flatnonzero(~validation_mask).tolist()
    validation = np.flatnonzero(validation_mask).tolist()
    for source in frame["__src__"].drop_duplicates():
        source_mask = frame["__src__"].eq(source).to_numpy()
        if not (source_mask & validation_mask).any() or not (source_mask & ~validation_mask).any():
            raise ValueError(
                f"{source} needs rows on both sides of the grouped split. "
                "Supply more distinct comments or adjust --seed/--validation-fraction."
            )
    rng.shuffle(training)
    rng.shuffle(validation)
    return training, validation


def training_class_weights(frame: "pd.DataFrame", training_indices, source, labels):
    """Fit only on the selected task's training rows, never holdout rows."""
    training = frame.iloc[training_indices]
    rows = training.loc[training["__src__"] == source, labels]
    return compute_pos_weight(rows.to_numpy(dtype="float32"))
