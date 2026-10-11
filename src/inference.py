"""Local, bounded multi-label inference for the reconstructed comment evaluator.

The original custom BERT checkpoint is unavailable. Two pinned public models
replace it for demonstration; their scores are not calibrated confidence.
Network access occurs only in ``python -m src.download_models``.
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from threading import RLock
from typing import Any

import torch

from .config import (
    ARTIFACT_DIR, MAX_BATCH_COMMENTS, MAX_COMMENT_CHARS, MODEL_DIR,
    POSITIVE_LABELS, TOXIC_LABELS,
)


MANIFEST_PATH = Path(__file__).with_name("model_manifest.json")
MAX_COMMENTS = MAX_BATCH_COMMENTS
MAX_CHARACTERS = MAX_COMMENT_CHARS


class ModelUnavailableError(RuntimeError):
    """The local model artifacts cannot be loaded."""


class PredictionError(RuntimeError):
    """A model/tokenizer failed or returned an invalid prediction."""


class InputTooLongError(ValueError):
    """A comment exceeds the token budget of at least one tokenizer."""

    def __init__(self, index: int, max_tokens: int):
        self.index = index
        self.max_tokens = max_tokens
        super().__init__(
            f"Comment {index + 1} exceeds {max_tokens} tokens. Shorten the comment; "
            "the application does not silently truncate it."
        )


@dataclass
class _LoadedModel:
    tokenizer: Any
    model: Any
    indices: dict[str, int]
    num_labels: int


def read_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def _label_indices(config: Any, required_labels: list[str], num_labels: int) -> dict[str, int]:
    """Resolve labels by name and reject incompatible artifacts."""
    mapping = {int(index): label for index, label in config.id2label.items()}
    if (
        config.num_labels != num_labels
        or set(mapping) != set(range(num_labels))
        or len(set(mapping.values())) != num_labels
        or not set(required_labels).issubset(mapping.values())
        or config.problem_type != "multi_label_classification"
    ):
        raise ModelUnavailableError("Local model labels are incompatible with this application.")
    by_name = {label: index for index, label in mapping.items()}
    return {label: by_name[label] for label in required_labels}


def _summary(positivity_bin: dict, toxicity_bin: dict) -> tuple[str, str]:
    positive = [label for label, detected in positivity_bin.items() if detected]
    toxic = [label for label, detected in toxicity_bin.items() if detected]
    if positive and toxic:
        return (
            "mixed_signals",
            f"Positive emotion signals ({', '.join(positive)}) and toxicity signals "
            f"({', '.join(toxic)}) meet the selected threshold.",
        )
    if toxic:
        return "toxicity_signals", f"Toxicity signals ({', '.join(toxic)}) meet the selected threshold."
    if positive:
        return "positive_signals", f"Positive emotion signals ({', '.join(positive)}) meet the selected threshold."
    return (
        "no_detected_signals",
        "None of the six positive emotion or six toxicity signals meet the selected "
        "threshold. This does not establish neutral or negative sentiment.",
    )


class CommentEvaluator:
    def __init__(self, model_dir=None, max_tokens: int = 192, device: str = "cpu", backend: str = "public", artifact_dir=None):
        if type(max_tokens) is not int or not 3 <= max_tokens <= 192:
            raise ValueError("max_tokens must be an integer between 3 and 192.")
        if device != "cpu":
            raise ValueError("This reconstruction supports the CPU device.")
        self.model_dir = Path(model_dir if model_dir is not None else MODEL_DIR)
        self.max_tokens = max_tokens
        self.device = device
        self.backend = backend
        self.artifact_dir = Path(artifact_dir if artifact_dir is not None else Path(ARTIFACT_DIR) / "civicbalance")
        self._models: dict[str, _LoadedModel] = {}
        self._shared: _LoadedModel | None = None
        self._lock = RLock()

    @property
    def ready(self) -> bool:
        if self.backend == "shared_bert":
            return self._shared is not None
        return self.backend == "public" and set(self._models) == {"toxicity", "positive_emotions"}

    @property
    def setup_command(self) -> str:
        if self.backend == "public":
            return "python -m src.download_models"
        if self.backend == "shared_bert":
            return "python -m src.train_multitask --help"
        return "Set INFERENCE_BACKEND to public or shared_bert, then restart the API."

    def _load_shared(self) -> None:
        from .train_multitask import load_local_checkpoint

        tokenizer, model, metadata = load_local_checkpoint(self.artifact_dir)
        if (
            metadata.get("artifact_format") != "civicbalance-shared-bert-v1"
            or metadata.get("positive_labels") != POSITIVE_LABELS
            or metadata.get("toxic_labels") != TOXIC_LABELS
            or type(metadata.get("max_tokens")) is not int
            or not 3 <= metadata["max_tokens"] <= 192
        ):
            raise ModelUnavailableError("The local shared-BERT checkpoint has incompatible metadata.")
        self.max_tokens = min(self.max_tokens, metadata["max_tokens"])
        model.to(self.device)
        model.eval()
        self._shared = _LoadedModel(tokenizer, model, {}, len(TOXIC_LABELS) + len(POSITIVE_LABELS))

    def load(self) -> None:
        """Load only local safetensors; never download or execute remote code."""
        with self._lock:
            if self.ready:
                return
            self._models = {}
            self._shared = None
            try:
                torch.set_num_threads(min(torch.get_num_threads(), 4))
                if self.backend == "shared_bert":
                    self._load_shared()
                    return
                if self.backend != "public":
                    raise ModelUnavailableError("INFERENCE_BACKEND must be public or shared_bert.")
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                manifest = read_manifest()
                provenance_path = self.model_dir / "download_manifest.json"
                if not provenance_path.is_file():
                    raise ModelUnavailableError(
                        "Local models are missing. Run python -m src.download_models, then restart the API."
                    )
                with provenance_path.open(encoding="utf-8") as file:
                    provenance = json.load(file)
                expected = {
                    spec["key"]: {"repo_id": spec["repo_id"], "revision": spec["revision"]}
                    for spec in manifest["models"]
                }
                if provenance != expected:
                    raise ModelUnavailableError(
                        "Local model provenance is incompatible. Run python -m src.download_models again."
                    )
                loaded = {}
                for spec in manifest["models"]:
                    folder = self.model_dir / spec["key"]
                    if not all((folder / filename).is_file() for filename in spec["required_files"]):
                        raise ModelUnavailableError(
                            "Local models are incomplete. Run python -m src.download_models again."
                        )
                    tokenizer = AutoTokenizer.from_pretrained(
                        folder, local_files_only=True, trust_remote_code=False
                    )
                    model = AutoModelForSequenceClassification.from_pretrained(
                        folder,
                        local_files_only=True,
                        trust_remote_code=False,
                        use_safetensors=True,
                    )
                    indices = _label_indices(model.config, spec["required_labels"], spec["num_labels"])
                    model.to(self.device)
                    model.eval()
                    loaded[spec["key"]] = _LoadedModel(tokenizer, model, indices, spec["num_labels"])
                # Publish both models together; a partially loaded service is never ready.
                self._models = loaded
            except ModelUnavailableError:
                raise
            except Exception:
                # Never expose raw model errors, paths, environment values, or user text.
                if self.backend == "shared_bert":
                    raise ModelUnavailableError(
                        "The local shared-BERT checkpoint could not be loaded. Prepare labeled "
                        "data and train it; see python -m src.train_multitask --help."
                    ) from None
                raise ModelUnavailableError(
                    "Local models could not be loaded. Check the installation and run "
                    "python -m src.download_models again."
                ) from None

    @staticmethod
    def _validate(texts: list[str], threshold: float, w_pos: float, w_tox: float) -> None:
        if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_COMMENTS:
            raise ValueError(f"Provide a list of 1 to {MAX_COMMENTS} comments.")
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Every comment must contain non-whitespace text.")
            if len(text) > MAX_CHARACTERS:
                raise ValueError(f"Each comment must contain at most {MAX_CHARACTERS} characters.")
        for name, value, upper in (("threshold", threshold, 1), ("w_pos", w_pos, 3), ("w_tox", w_tox, 3)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value <= upper
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number from 0 to {upper}.")

    def predict(self, texts: list[str], threshold: float = 0.5, w_pos: float = 1.0, w_tox: float = 1.2) -> list[dict]:
        """Return one independent result per comment, preserving input order."""
        self._validate(texts, threshold, w_pos, w_tox)
        with self._lock:
            if not self.ready:
                raise ModelUnavailableError(
                    f"Models are not loaded. See {self.setup_command}, then restart the API."
                )
            try:
                encoded = {}
                # Validate both tokenizers before running either model. Including
                # special tokens keeps the actual model input within the budget.
                tokenizers = self._models if self.backend == "public" else {"shared_bert": self._shared}
                for key, loaded in tokenizers.items():
                    batch = loaded.tokenizer(
                        texts, padding=True, truncation=False, add_special_tokens=True,
                        return_tensors="pt", verbose=False,
                    )
                    ids = batch["input_ids"]
                    mask = batch["attention_mask"]
                    if ids.ndim != 2 or ids.shape[0] != len(texts) or mask.shape != ids.shape:
                        raise PredictionError("The tokenizer returned an invalid input shape.")
                    for index, length in enumerate(mask.sum(dim=1).tolist()):
                        if length > self.max_tokens:
                            raise InputTooLongError(index, self.max_tokens)
                    encoded[key] = {name: value.to(self.device) for name, value in batch.items()}

                scores = {}
                with torch.inference_mode():
                    if self.backend == "shared_bert":
                        outputs = self._shared.model(**encoded["shared_bert"])
                        if not isinstance(outputs, tuple) or len(outputs) != 2:
                            raise PredictionError("The model returned invalid prediction values.")
                        toxic_logits, positive_logits = outputs
                        for key, logits, labels in (
                            ("toxicity", toxic_logits, TOXIC_LABELS),
                            ("positive_emotions", positive_logits, POSITIVE_LABELS),
                        ):
                            scores[key] = self._scores(logits, {label: i for i, label in enumerate(labels)}, len(labels), len(texts))
                    else:
                        for key, loaded in self._models.items():
                            logits = loaded.model(**encoded[key]).logits
                            scores[key] = self._scores(logits, loaded.indices, loaded.num_labels, len(texts))

                results = []
                for index in range(len(texts)):
                    tox = scores["toxicity"][index]
                    pos = scores["positive_emotions"][index]
                    tox_bin = {label: int(score >= threshold) for label, score in tox.items()}
                    pos_bin = {label: int(score >= threshold) for label, score in pos.items()}
                    label, explanation = _summary(pos_bin, tox_bin)
                    results.append({
                        "toxicity": tox,
                        "toxicity_bin": tox_bin,
                        "positivity": pos,
                        "positivity_bin": pos_bin,
                        "net_goodness": w_pos * sum(pos.values()) / len(POSITIVE_LABELS)
                        - w_tox * sum(tox.values()) / len(TOXIC_LABELS),
                        "label": label,
                        "explanation": explanation,
                    })
                return results
            except (InputTooLongError, PredictionError):
                raise
            except Exception:
                raise PredictionError(
                    "Prediction failed. Try again; if the problem persists, restart the API."
                ) from None

    @staticmethod
    def _scores(logits, indices: dict[str, int], num_labels: int, count: int) -> list[dict]:
        if (
            not isinstance(logits, torch.Tensor)
            or logits.shape != (count, num_labels)
            or not logits.is_floating_point()
            or not bool(torch.isfinite(logits).all())
        ):
            raise PredictionError("The model returned invalid prediction values.")
        probabilities = torch.sigmoid(logits).cpu()
        if not bool(((probabilities >= 0) & (probabilities <= 1)).all()):
            raise PredictionError("The model returned invalid scores.")
        return [
            {label: float(row[index]) for label, index in indices.items()}
            for row in probabilities
        ]
