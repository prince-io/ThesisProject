"""Run a YOLO training job with reproducible metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
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


def _gpu_used_mb() -> float | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        ).strip()
        return float(output.splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


class _VramTracker:
    """Track PyTorch peak memory from training callbacks."""

    def __init__(self) -> None:
        self.peak_allocated = 0.0
        self.peak_reserved = 0.0

    def __call__(self, _trainer) -> None:
        if not torch.cuda.is_available():
            return
        self.peak_allocated = max(self.peak_allocated, torch.cuda.max_memory_allocated())
        self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved())


def _start_gpu_sampler() -> tuple[subprocess.Popen | None, Path | None]:
    if _gpu_used_mb() is None:
        return None, None
    handle = tempfile.NamedTemporaryFile(prefix="vram_", suffix=".csv", delete=False)
    path = Path(handle.name)
    handle.close()
    try:
        with path.open("w") as output:
            process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-lms",
                    "200",
                ],
                stdout=output,
                stderr=subprocess.DEVNULL,
            )
        return process, path
    except Exception:  # noqa: BLE001
        path.unlink(missing_ok=True)
        return None, None


def _stop_gpu_sampler(process: subprocess.Popen | None, path: Path | None) -> float | None:
    if process is None or path is None:
        return None
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    values = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    path.unlink(missing_ok=True)
    return max(values) if values else None


def build_vram(tracker: _VramTracker, baseline_used: float | None, peak_used: float | None) -> dict:
    total_mb = None
    if torch.cuda.is_available():
        total_mb = round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1)
    delta = (
        round(peak_used - baseline_used, 1)
        if peak_used is not None and baseline_used is not None
        else None
    )
    return {
        "pytorch_peak_allocated_mb": round(tracker.peak_allocated / 1024**2, 1),
        "pytorch_peak_reserved_mb": round(tracker.peak_reserved / 1024**2, 1),
        "gpu_total_mb": total_mb,
        "gpu_used_baseline_mb": baseline_used,
        "gpu_used_peak_mb": peak_used,
        "gpu_used_peak_delta_mb": delta,
    }


def build_metadata(root: Path, config: dict, save_dir: Path, vram: dict) -> dict:
    checksums = root / "data/processed/adverse16k/checksums.sha1"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": " ".join(sys.argv),
        "git_commit": _git_commit(root),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "gpu_total_mb": vram.get("gpu_total_mb"),
        "training_peak_allocated_mb": vram.get("pytorch_peak_allocated_mb"),
        "training_peak_reserved_mb": vram.get("pytorch_peak_reserved_mb"),
        "gpu_used_peak_mb": vram.get("gpu_used_peak_mb"),
        "gpu_used_peak_delta_mb": vram.get("gpu_used_peak_delta_mb"),
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
    tracker = _VramTracker()
    baseline_used = _gpu_used_mb()
    sampler, sampler_path = _start_gpu_sampler()

    try:
        if resume:
            last_weights = Path(config["project"]) / run_name / "weights/last.pt"
            if not last_weights.exists():
                raise FileNotFoundError(f"resume weights not found: {last_weights}")
            model = YOLO(str(last_weights))
            for event in ("on_train_batch_end", "on_fit_epoch_end", "on_train_end"):
                model.add_callback(event, tracker)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            model.train(resume=True)
        else:
            train_args = {key: value for key, value in config.items() if key != "model"}
            train_args["name"] = run_name
            if epochs is not None:
                train_args["epochs"] = epochs
            if extra:
                train_args.update(extra)
            model = YOLO(str(config["model"]))
            for event in ("on_train_batch_end", "on_fit_epoch_end", "on_train_end"):
                model.add_callback(event, tracker)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            model.train(**train_args)
    finally:
        peak_used = _stop_gpu_sampler(sampler, sampler_path)
    vram = build_vram(tracker, baseline_used, peak_used)
    save_dir = Path(model.trainer.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "vram.json").write_text(json.dumps(vram, indent=2), encoding="utf-8")
    (save_dir / "run_meta.json").write_text(
        json.dumps(build_metadata(root, config, save_dir, vram), indent=2),
        encoding="utf-8",
    )
    (save_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    print(f"run directory: {save_dir}", flush=True)
    print(f"VRAM: {json.dumps(vram)}", flush=True)
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
