"""Run a YOLO training job with reproducible metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO

from litevit.training.config import load_config, resolve_paths


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return ""


def _sha1(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha1").hexdigest()


def build_metadata(root: Path, config: dict, save_dir: Path) -> dict:
    checksums = root / "data/processed/adverse16k/checksums.sha1"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    total_mb = None
    if torch.cuda.is_available():
        total_mb = round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1)
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(root),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "gpu_total_mb": total_mb,
        "peak_vram_mb": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            if torch.cuda.is_available()
            else None
        ),
        "config": config,
        "save_dir": str(save_dir),
        "dataset_checksums": str(checksums) if checksums.exists() else None,
        "dataset_checksums_sha1": _sha1(checksums) if checksums.exists() else None,
    }


def train(
    config_path: Path,
    root: Path | None = None,
    name: str | None = None,
    epochs: int | None = None,
    resume: bool = False,
    extra: dict | None = None,
) -> Path:
    root = Path.cwd() if root is None else root
    config = resolve_paths(load_config(config_path), root)
    run_name = name or config.get("name") or "baseline"
    if resume:
        last_weights = Path(config["project"]) / run_name / "weights/last.pt"
        if not last_weights.exists():
            raise FileNotFoundError(f"resume weights not found: {last_weights}")
        model = YOLO(str(last_weights))
        model.train(resume=True)
    else:
        train_args = {key: value for key, value in config.items() if key != "model"}
        train_args["name"] = run_name
        if epochs is not None:
            train_args["epochs"] = epochs
        if extra:
            train_args.update(extra)
        model = YOLO(str(config["model"]))
        model.train(**train_args)

    save_dir = Path(model.trainer.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "run_meta.json").write_text(
        json.dumps(build_metadata(root, config, save_dir), indent=2), encoding="utf-8"
    )
    (save_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    print(f"run directory: {save_dir}", flush=True)
    return save_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO detection model")
    parser.add_argument("--config", type=Path, default=Path("configs/train/baseline.yaml"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--name", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(
        config_path=args.config,
        root=args.root.resolve(),
        name=args.name,
        epochs=args.epochs,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
