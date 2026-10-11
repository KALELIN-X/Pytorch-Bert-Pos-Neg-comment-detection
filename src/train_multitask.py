"""Repaired optional shared-BERT, multi-label training workflow.

No training dataset or historical trained checkpoint is included. Local CSVs
must be supplied explicitly. Metrics describe the seeded validation split,
which also selects the checkpoint; they are not independent test accuracy.
"""

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoConfig, AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from safetensors.torch import load_file, save_file

from .config import POSITIVE_LABELS, TOXIC_LABELS, GOEMOTIONS_CSV, JIGSAW_TRAIN_CSV, ARTIFACT_DIR
from .utils import split_indices_by_source, training_class_weights

BASE_MODEL = "google-bert/bert-base-uncased"
BASE_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
ARTIFACT_FORMAT = "civicbalance-shared-bert-v1"


@dataclass
class Args:
    model_name: str = BASE_MODEL
    revision: str = BASE_REVISION
    jigsaw_csv: str = JIGSAW_TRAIN_CSV
    goemotions_csv: str = GOEMOTIONS_CSV
    output_dir: str = str(Path(ARTIFACT_DIR) / "civicbalance")
    max_len: int = 192
    batch_size: int = 16
    epochs: int = 3
    lr: float = 2e-5
    warmup_ratio: float = 0.1
    alpha: float = 1.0
    beta: float = 1.0
    validation_fraction: float = 0.1
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def validate(self):
        if not 3 <= self.max_len <= 192:
            raise ValueError("max_len must be between 3 and 192 tokens.")
        if self.batch_size < 1 or self.epochs < 1:
            raise ValueError("batch_size and epochs must be positive integers.")
        if not 0 < self.lr < 1 or not 0 <= self.warmup_ratio < 1:
            raise ValueError("lr must be between 0 and 1 and warmup_ratio between 0 and 1.")
        if not 0 < self.validation_fraction < 1 or not self.alpha > 0 or not self.beta > 0:
            raise ValueError("validation_fraction must be between 0 and 1; both loss weights must be positive.")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda.")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; use --device cpu.")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", self.revision):
            raise ValueError("Provide a full 40-character immutable Hugging Face model revision for reproducible training.")


class MixedDataset(Dataset):
    def __init__(self, jigsaw_csv, goemotions_csv, tokenizer, max_len=192):
        # Pandas/SQL extras are not imported by serving a trained checkpoint.
        import pandas as pd
        from .etl import load_goemotions, load_jigsaw

        self.tok, self.max_len = tokenizer, max_len
        jigsaw = load_jigsaw(jigsaw_csv)
        emotions = load_goemotions(goemotions_csv)
        jigsaw["__src__"] = "jigsaw"
        emotions["__src__"] = "goemotions"
        self.df = pd.concat([jigsaw, emotions], ignore_index=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        encoding = self.tok(
            row["text"], truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["yT"] = torch.full((len(TOXIC_LABELS),), float("nan"))
        item["yP"] = torch.full((len(POSITIVE_LABELS),), float("nan"))
        labels = TOXIC_LABELS if row["__src__"] == "jigsaw" else POSITIVE_LABELS
        key = "yT" if row["__src__"] == "jigsaw" else "yP"
        item[key] = torch.tensor(row[labels].to_numpy(dtype="float32"))
        return item


class CivicBalance(nn.Module):
    """One shared encoder and two independent six-label sigmoid heads."""

    def __init__(self, model_name=BASE_MODEL, revision=BASE_REVISION, *, backbone=None):
        super().__init__()
        self.backbone = backbone if backbone is not None else AutoModel.from_pretrained(
            model_name, revision=revision, use_safetensors=True,
        )
        hidden_size = self.backbone.config.hidden_size
        self.toxic_head = nn.Linear(hidden_size, len(TOXIC_LABELS))
        self.pos_head = nn.Linear(hidden_size, len(POSITIVE_LABELS))

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        encoding = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoding["token_type_ids"] = token_type_ids
        representation = self.backbone(**encoding).last_hidden_state[:, 0]
        return self.toxic_head(representation), self.pos_head(representation)


def masked_multitask_loss(logits_t, logits_p, targets_t, targets_p, criterion_t, criterion_p, alpha=1.0, beta=1.0):
    """Unknown task labels are masked, not treated as negative examples."""
    losses = []
    for logits, targets, criterion in (
        (logits_t, targets_t, criterion_t), (logits_p, targets_p, criterion_p),
    ):
        mask = ~torch.isnan(targets)
        known = torch.nan_to_num(targets)
        per_label = criterion(logits, known)
        losses.append((per_label * mask).sum() / mask.sum().clamp(min=1))
    return alpha * losses[0] + beta * losses[1]


def _encoding(batch, device):
    return {key: batch[key].to(device) for key in ("input_ids", "attention_mask", "token_type_ids") if key in batch}


def train_epoch(model, loader, optimizer, scheduler, criteria, device="cpu", alpha=1.0, beta=1.0):
    model.train()
    loss_sum, n_batches = 0.0, 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits_t, logits_p = model(**_encoding(batch, device))
        loss = masked_multitask_loss(
            logits_t, logits_p, batch["yT"].to(device), batch["yP"].to(device),
            criteria[0], criteria[1], alpha, beta,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Training produced a non-finite loss; inspect labels and hyperparameters.")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        loss_sum += loss.item()
        n_batches += 1
    if not n_batches:
        raise ValueError("Training loader is empty.")
    return loss_sum / n_batches


def classification_metrics(targets, predictions, labels):
    """Measured binary multi-label metrics; zero denominators use zero."""
    truth, predicted = targets.bool(), predictions.bool()
    true_positives = (truth & predicted).sum(0).float()
    false_positives = (~truth & predicted).sum(0).float()
    false_negatives = (truth & ~predicted).sum(0).float()
    precision = true_positives / (true_positives + false_positives).clamp(min=1)
    recall = true_positives / (true_positives + false_negatives).clamp(min=1)
    f1 = 2 * true_positives / (2 * true_positives + false_positives + false_negatives).clamp(min=1)
    micro = 2 * true_positives.sum() / (2 * true_positives.sum() + false_positives.sum() + false_negatives.sum()).clamp(min=1)
    return {
        "n_examples": len(targets), "micro_f1": micro.item(), "macro_f1": f1.mean().item(),
        "per_label": {
            label: {"precision": precision[i].item(), "recall": recall[i].item(),
                    "f1": f1[i].item(), "support": int(truth[:, i].sum())}
            for i, label in enumerate(labels)
        },
    }


def evaluate(model, loader, device="cpu", threshold=0.5):
    model.eval()
    predictions, targets = {"toxicity": [], "positivity": []}, {"toxicity": [], "positivity": []}
    with torch.inference_mode():
        for batch in loader:
            logits = model(**_encoding(batch, device))
            for name, key, scores in (("toxicity", "yT", logits[0]), ("positivity", "yP", logits[1])):
                known_rows = ~torch.isnan(batch[key]).any(dim=1)
                if known_rows.any():
                    predictions[name].append((scores.sigmoid().cpu()[known_rows] >= threshold))
                    targets[name].append(batch[key][known_rows])
    results = {}
    for name, labels in (("toxicity", TOXIC_LABELS), ("positivity", POSITIVE_LABELS)):
        if not targets[name]:
            raise ValueError(f"Validation split has no labeled {name} examples.")
        results[name] = classification_metrics(torch.cat(targets[name]), torch.cat(predictions[name]), labels)
    return results


def save_checkpoint(model, tokenizer, output_dir, metadata):
    """Write portable local files; weights use safetensors rather than pickle."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.backbone.config.save_pretrained(output)
    tokenizer.save_pretrained(output)
    weights = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    temporary_weights = output / "model.safetensors.tmp"
    save_file(weights, str(temporary_weights))
    temporary_weights.replace(output / "model.safetensors")
    temporary_metadata = output / "training_metadata.json.tmp"
    temporary_metadata.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary_metadata.replace(output / "training_metadata.json")


def load_local_checkpoint(artifact_dir):
    """Load a reconstructed shared-BERT checkpoint without any network access."""
    directory = Path(artifact_dir)
    metadata = json.loads((directory / "training_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("artifact_format") != ARTIFACT_FORMAT:
        raise ValueError("Unsupported shared-BERT checkpoint format.")
    if metadata.get("positive_labels") != POSITIVE_LABELS or metadata.get("toxic_labels") != TOXIC_LABELS:
        raise ValueError("Checkpoint labels do not match CivicBalance's original label order.")
    max_tokens = metadata.get("max_tokens")
    if type(max_tokens) is not int or not 3 <= max_tokens <= 192:
        raise ValueError("Checkpoint max_tokens must be an integer between 3 and 192.")
    config = AutoConfig.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    model = CivicBalance(backbone=AutoModel.from_config(config, trust_remote_code=False))
    model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    model.eval()
    return tokenizer, model, metadata


def _file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train(**kwargs):
    args = Args(**kwargs)
    args.validate()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # Validate local data before downloading/initializing a backbone.
    from .etl import load_jigsaw, load_goemotions
    load_jigsaw(args.jigsaw_csv)
    load_goemotions(args.goemotions_csv)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision, trust_remote_code=False)
    dataset = MixedDataset(args.jigsaw_csv, args.goemotions_csv, tokenizer, args.max_len)
    train_indices, val_indices = split_indices_by_source(dataset.df, args.validation_fraction, args.seed)
    class_weights = [
        training_class_weights(dataset.df, train_indices, "jigsaw", TOXIC_LABELS),
        training_class_weights(dataset.df, train_indices, "goemotions", POSITIVE_LABELS),
    ]
    criteria = tuple(nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights, device=args.device), reduction="none") for weights in class_weights)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size)
    model = CivicBalance(args.model_name, args.revision).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    split_counts = {
        side: dataset.df.iloc[indices]["__src__"].value_counts().to_dict()
        for side, indices in (("training", train_indices), ("validation", val_indices))
    }
    metadata = {
        "artifact_format": ARTIFACT_FORMAT, "base_model": args.model_name, "base_revision": args.revision,
        "max_tokens": args.max_len, "positive_labels": POSITIVE_LABELS, "toxic_labels": TOXIC_LABELS,
        "seed": args.seed, "split_method": "seeded normalized-text-group random split shared across data sources",
        "validation_fraction": args.validation_fraction, "split_counts": split_counts,
        "dataset_sha256": {"jigsaw": _file_hash(args.jigsaw_csv), "goemotions": _file_hash(args.goemotions_csv)},
        "hyperparameters": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                            "warmup_ratio": args.warmup_ratio, "alpha": args.alpha, "beta": args.beta},
        "pos_weights": {"toxicity": class_weights[0].tolist(), "positivity": class_weights[1].tolist()},
        "threshold": 0.5, "selection_metric": "mean of toxicity and positivity validation macro-F1",
        "evaluation_note": "Validation selects the best checkpoint; these are not independent test results. "
                           "Group splitting ignores case/whitespace but does not guarantee author/thread separation. "
                           "Validation fraction is measured in text groups; label metrics use zero_division=0 and report support. "
                           "GPU determinism is requested but some operations can remain nondeterministic.",
    }
    best_score, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, criteria, args.device, args.alpha, args.beta)
        measured = evaluate(model, val_loader, args.device)
        score = (measured["toxicity"]["macro_f1"] + measured["positivity"]["macro_f1"]) / 2
        history.append({"epoch": epoch, "training_loss": loss, "validation_metrics": measured})
        print(f"Epoch {epoch}/{args.epochs}: loss={loss:.4f}; validation mean macro-F1={score:.4f}")
        if score > best_score:
            best_score = score
            save_checkpoint(model, tokenizer, args.output_dir, {**metadata, "best_epoch": epoch, "validation_metrics": measured})
    output = Path(args.output_dir)
    (output / "training_history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved best validation checkpoint to {output}; no independent test accuracy was measured.")
    return json.loads((output / "training_metadata.json").read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", "--model_name", default=BASE_MODEL)
    parser.add_argument("--revision", default=BASE_REVISION, help="Immutable model revision; change together with --model-name")
    parser.add_argument("--jigsaw-csv", "--jigsaw_csv", default=JIGSAW_TRAIN_CSV)
    parser.add_argument("--goemotions-csv", "--goemotions_csv", default=GOEMOTIONS_CSV)
    parser.add_argument("--output-dir", "--output_dir", default=str(Path(ARTIFACT_DIR) / "civicbalance"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", "--max_len", type=int, default=192)
    parser.add_argument("--warmup-ratio", "--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--validation-fraction", "--validation_fraction", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=Args.device)
    try:
        train(**vars(parser.parse_args()))
    except ImportError:
        parser.exit(1, "Training dependencies are unavailable. Run: python -m pip install -r requirements-training.txt\n")
    except ValueError as error:
        parser.exit(1, f"Training input error: {error}\n")
    except (OSError, RuntimeError):
        parser.exit(1, "Training failed. Check local data/artifact permissions, model revision/download access, and device resources.\n")


if __name__ == "__main__":
    main()
