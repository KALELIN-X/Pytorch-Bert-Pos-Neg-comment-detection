"""Explicitly download the pinned public artifacts for offline CPU inference."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import MODEL_DIR
from .inference import read_manifest


def download_models(model_dir=None) -> None:
    root = Path(model_dir if model_dir is not None else MODEL_DIR)
    root.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()
    provenance = {}
    for spec in manifest["models"]:
        print(f"Downloading public model {spec['repo_id']} at revision {spec['revision']}...")
        folder = root / spec["key"]
        snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            local_dir=folder,
            allow_patterns=manifest["allow_patterns"],
            token=False,
            max_workers=4,
        )
        if not all((folder / filename).is_file() for filename in spec["required_files"]):
            raise RuntimeError("The download is incomplete. Re-run python -m src.download_models.")
        provenance[spec["key"]] = {"repo_id": spec["repo_id"], "revision": spec["revision"]}
    # Write provenance only after both complete; inference rejects partial setup.
    path = root / "download_manifest.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print("Both local models are ready. Inference uses local files only.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(MODEL_DIR))
    args = parser.parse_args()
    try:
        download_models(args.model_dir)
    except Exception:
        # Model/network errors may contain local paths or credentials; keep the
        # user-visible message fixed and do not print the underlying exception.
        print("Model download failed. Check network access and available disk space, then retry.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
