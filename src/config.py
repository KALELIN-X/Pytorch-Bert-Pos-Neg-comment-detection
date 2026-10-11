"""Repository-relative configuration for the API and preserved training scripts."""

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_FOLDER = REPO_ROOT / "MainFolder"

# Explicit paths keep configuration independent of the shell's current folder.
# Existing environment variables always take precedence over optional .env files.
load_dotenv(REPO_ROOT / ".env", override=False)
load_dotenv(MAIN_FOLDER / ".env", override=False)


def _configured_path(name: str, default: str) -> str:
    value = os.getenv(name, default)
    if not value:
        return ""
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else REPO_ROOT / path)


# Postgres is optional: neither prediction nor the dashboard imports DB utilities.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/civicbalance")
GOEMOTIONS_CSV = _configured_path("GOEMOTIONS_CSV", "MainFolder/data/goemotions.csv")
JIGSAW_TRAIN_CSV = _configured_path("JIGSAW_TRAIN_CSV", "MainFolder/data/jigsaw_train.csv")
TWITTER_EMOTION_CSV = _configured_path("TWITTER_EMOTION_CSV", "")
ARTIFACT_DIR = _configured_path("ARTIFACT_DIR", "MainFolder/artifacts")
MODEL_DIR = _configured_path("MODEL_DIR", ".models")
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "public")
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

API_MAX_TOKENS = 192
MAX_COMMENT_CHARS = 2000
MAX_BATCH_COMMENTS = 20
MAX_REQUEST_BYTES = 64 * 1024

POSITIVE_LABELS = ["admiration", "gratitude", "joy", "pride", "love", "optimism"]
TOXIC_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
