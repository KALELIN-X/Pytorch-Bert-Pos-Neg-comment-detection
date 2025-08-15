from dotenv import load_dotenv # pyright: ignore[reportMissingImports]
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/civicbalance")
GOEMOTIONS_CSV = os.getenv("GOEMOTIONS_CSV", "data/goemotions_shivamb.csv")
JIGSAW_TRAIN_CSV = os.getenv("JIGSAW_TRAIN_CSV", "data/jigsaw_multilingual_train.csv")
TWITTER_EMOTION_CSV = os.getenv("TWITTER_EMOTION_CSV", "")
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

POSITIVE_LABELS = ["admiration","gratitude","joy","pride","love","optimism"]
TOXIC_LABELS = ["toxic","severe_toxic","obscene","threat","insult","identity_hate"]