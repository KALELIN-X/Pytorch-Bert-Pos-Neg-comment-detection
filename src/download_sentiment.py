"""Explicit anonymous download of the pinned sentiment model for local inference."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import MODEL_DIR
from .sentiment import read_manifest


def download_sentiment(model_dir=None) -> None:
    folder = Path(model_dir if model_dir is not None else Path(MODEL_DIR) / "sentiment")
    folder.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()
    print(f"Downloading public sentiment model {manifest['repo_id']} at {manifest['revision']}...")
    snapshot_download(
        repo_id=manifest["repo_id"], revision=manifest["revision"], local_dir=folder,
        allow_patterns=manifest["allow_patterns"], token=False, max_workers=4,
    )
    if not all((folder / name).is_file() for name in manifest["required_files"]):
        raise RuntimeError("The sentiment download is incomplete. Run python -m src.download_sentiment again.")
    provenance = {"repo_id": manifest["repo_id"], "revision": manifest["revision"]}
    path = folder / "download_manifest.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print("The local sentiment model is ready. Inference uses local files only.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(MODEL_DIR) / "sentiment")
    arguments = parser.parse_args()
    try:
        download_sentiment(arguments.model_dir)
    except Exception:
        print("Sentiment model download failed. Check network access and free disk space, then retry.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
