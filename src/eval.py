"""Run illustrative comments locally; this is not labeled-dataset evaluation."""

import argparse
import json
from pathlib import Path

from .config import ARTIFACT_DIR, INFERENCE_BACKEND, MAX_COMMENT_CHARS, MAX_BATCH_COMMENTS
from .inference import CommentEvaluator, ModelUnavailableError, PredictionError, InputTooLongError

SAMPLES = [
    "I appreciate your help, thanks a ton!",
    "You are a complete idiot and should be banned.",
    "The meeting starts at three o'clock.",
    "Great, another crash right before the deadline.",
]


def soft_scores(texts, **evaluator_options):
    """Preserve the original helper name while using the rebuilt prediction path."""
    evaluator = CommentEvaluator(**evaluator_options)
    evaluator.load()
    return evaluator.predict(texts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", action="append", help="Comment to evaluate; repeat for a batch. Defaults to demo samples.")
    parser.add_argument("--backend", choices=("public", "shared_bert"), default=INFERENCE_BACKEND)
    parser.add_argument("--model-dir", help="Local downloaded public model directory")
    parser.add_argument("--artifact-dir", default=str(Path(ARTIFACT_DIR) / "civicbalance"))
    arguments = parser.parse_args()
    texts = arguments.text if arguments.text is not None else SAMPLES
    if len(texts) > MAX_BATCH_COMMENTS:
        parser.error(f"At most {MAX_BATCH_COMMENTS} comments are allowed.")
    if any(not text.strip() or len(text) > MAX_COMMENT_CHARS for text in texts):
        parser.error(f"Comments must be nonblank and at most {MAX_COMMENT_CHARS} characters each.")
    try:
        results = soft_scores(
            texts, model_dir=arguments.model_dir, backend=arguments.backend, artifact_dir=arguments.artifact_dir,
        )
    except (ModelUnavailableError, PredictionError, InputTooLongError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    print("Illustrative predictions only; ambiguous/sarcastic examples have no required label.")
    print("Scores are independent model outputs, not calibrated confidence. No accuracy is measured here.")
    print(json.dumps([{"text": text, **result} for text, result in zip(texts, results)], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
