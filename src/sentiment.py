"""New sentence sentiment classification, distinct from CivicBalance's 12 labels."""

import json
from pathlib import Path
from threading import RLock

import torch

from .config import MAX_BATCH_COMMENTS, MAX_COMMENT_CHARS, MODEL_DIR
from .inference import InputTooLongError, ModelUnavailableError, PredictionError


LABELS = ["positive", "neutral", "negative"]
MANIFEST_PATH = Path(__file__).with_name("sentiment_manifest.json")


def read_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _label_mapping(config) -> dict[int, str]:
    mapping = {int(index): label for index, label in config.id2label.items()}
    if config.num_labels != 3 or set(mapping) != {0, 1, 2} or set(mapping.values()) != set(LABELS):
        raise ModelUnavailableError("The local sentiment model has incompatible labels.")
    return mapping


class SentimentEvaluator:
    """Load the pinned local model once; serialize bounded CPU inference."""

    def __init__(self, model_dir=None, max_tokens: int = 192, device: str = "cpu"):
        if type(max_tokens) is not int or not 3 <= max_tokens <= 192:
            raise ValueError("max_tokens must be an integer between 3 and 192.")
        if device != "cpu":
            raise ValueError("This sentiment demonstration supports the CPU device.")
        self.model_dir = Path(model_dir if model_dir is not None else Path(MODEL_DIR) / "sentiment")
        self.max_tokens = max_tokens
        self.device = device
        self.backend = "sentiment"
        self._model = None
        self._tokenizer = None
        self._labels = {}
        self._lock = RLock()

    @property
    def ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None and len(self._labels) == 3

    @property
    def setup_command(self) -> str:
        return "python -m src.download_sentiment"

    def load(self) -> None:
        """Read local safetensors only; no implicit network or remote code."""
        with self._lock:
            if self.ready:
                return
            self._model, self._tokenizer, self._labels = None, None, {}
            try:
                manifest = read_manifest()
                if not all((self.model_dir / name).is_file() for name in manifest["required_files"]):
                    raise ModelUnavailableError(
                        "Local sentiment artifacts are missing. Run python -m src.download_sentiment, then restart the API."
                    )
                path = self.model_dir / "download_manifest.json"
                expected = {"repo_id": manifest["repo_id"], "revision": manifest["revision"]}
                if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != expected:
                    raise ModelUnavailableError(
                        "Local sentiment model provenance is incompatible. Run python -m src.download_sentiment again."
                    )
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                torch.set_num_threads(min(torch.get_num_threads(), 4))
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_dir, local_files_only=True, trust_remote_code=False,
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True,
                )
                labels = _label_mapping(model.config)
                model.to(self.device)
                model.eval()
                self._tokenizer, self._model, self._labels = tokenizer, model, labels
            except ModelUnavailableError:
                raise
            except Exception:
                raise ModelUnavailableError(
                    "The local sentiment model could not be loaded. Check the installation and "
                    "run python -m src.download_sentiment again."
                ) from None

    @staticmethod
    def _validate(texts) -> None:
        if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_BATCH_COMMENTS:
            raise ValueError(f"Provide a list of 1 to {MAX_BATCH_COMMENTS} comments.")
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Every comment must contain non-whitespace text.")
            if len(text) > MAX_COMMENT_CHARS:
                raise ValueError(f"Each comment must contain at most {MAX_COMMENT_CHARS} characters.")

    def predict(self, texts: list[str]) -> list[dict]:
        """Return a separate argmax sentiment label and score map for each text."""
        self._validate(texts)
        with self._lock:
            if not self.ready:
                raise ModelUnavailableError(
                    "The sentiment model is not loaded. Run python -m src.download_sentiment, then restart the API."
                )
            try:
                batch = self._tokenizer(
                    texts, padding=True, truncation=False, add_special_tokens=True,
                    return_tensors="pt", verbose=False,
                )
                ids, mask = batch["input_ids"], batch["attention_mask"]
                if ids.ndim != 2 or ids.shape[0] != len(texts) or mask.shape != ids.shape:
                    raise PredictionError("The sentiment tokenizer returned an invalid input shape.")
                for index, length in enumerate(mask.sum(dim=1).tolist()):
                    if length > self.max_tokens:
                        raise InputTooLongError(index, self.max_tokens)
                # DistilBERT uses input IDs and attention masks, without BERT's
                # segment token-type embeddings.
                encoding = {key: batch[key].to(self.device) for key in ("input_ids", "attention_mask")}
                with torch.inference_mode():
                    logits = self._model(**encoding).logits
                    if (
                        not isinstance(logits, torch.Tensor)
                        or logits.shape != (len(texts), 3)
                        or not logits.is_floating_point()
                        or not bool(torch.isfinite(logits).all())
                    ):
                        raise PredictionError("The sentiment model returned invalid prediction values.")
                    scores = torch.softmax(logits, dim=1).cpu()
                    if (
                        not bool(torch.isfinite(scores).all())
                        or not bool(((scores >= 0) & (scores <= 1)).all())
                        or not bool(torch.allclose(scores.sum(dim=1), torch.ones(len(texts), dtype=scores.dtype), atol=1e-5))
                    ):
                        raise PredictionError("The sentiment model returned invalid scores.")
                by_name = {label: index for index, label in self._labels.items()}
                results = []
                for row in scores:
                    label = self._labels[int(row.argmax())]
                    results.append({
                        "label": label,
                        "scores": {name: float(row[by_name[name]]) for name in LABELS},
                        "explanation": f"The highest model score is for {label} sentiment. "
                        "These scores are uncalibrated; context and sarcasm can affect the prediction.",
                    })
                return results
            except (InputTooLongError, PredictionError):
                raise
            except Exception:
                raise PredictionError(
                    "Sentiment prediction failed. Retry or restart the API if the problem persists."
                ) from None
