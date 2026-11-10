"""Run a YOLO training job with reproducible metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO

from litevit.models.build import EDGE_LAYER_INDICES, GATE_LAYER_INDICES, P5_LAYER_INDEX
from litevit.models.p2 import DETECT_LAYER_INDEX, P2_STRIDES
from litevit.training.config import load_config, resolve_paths
from litevit.training.trainer import (
    GateDetectionTrainer,
    GateEdgeCBAMTrainer,
    GateP2DetectionTrainer,
    GateP5DetectionTrainer,
    V1DetectionTrainer,
)


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


def verify_saved_architecture(save_dir: Path, arch: str) -> bool:
    """Fail the run if the saved checkpoint does not contain the requested architecture."""
    if arch == "baseline":
        return True
    if arch not in {"v1_p5", "gate_only", "gate_p2", "gate_p5", "gate_edge_cbam"}:
        raise ValueError(f"unsupported architecture: {arch}")
    checkpoint = save_dir / "weights/best.pt"
    if not checkpoint.exists():
        checkpoint = save_dir / "weights/last.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint found under {save_dir / 'weights'}")
    model = YOLO(str(checkpoint)).model
    layers = getattr(model, "model", None)
    if layers is None:
        raise RuntimeError("checkpoint has no detector layer sequence")
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError(f"{arch} checkpoint contains non-finite parameters")
    params = sum(parameter.numel() for parameter in model.parameters())
    if arch == "v1_p5":
        if not hasattr(layers[P5_LAYER_INDEX], "block"):
            raise RuntimeError(
                "V1 checkpoint does not contain a LiteViT block at the P5 layer; "
                "the architecture was discarded before training"
            )
        if params < 3_500_000:
            raise RuntimeError(
                f"V1 checkpoint has {params:,} parameters, which looks like the baseline"
            )
        block = layers[P5_LAYER_INDEX].block
        if block.qkv.weight.detach().abs().sum().item() == 0.0:
            raise RuntimeError(
                "V1 block weights are zero; training likely produced NaN "
                "and the checkpoint was sanitized"
            )
        return True
    if not all(hasattr(layers[index], "gate") for index in GATE_LAYER_INDICES):
        raise RuntimeError(
            f"{arch} checkpoint does not contain gates at P3/P4/P5; "
            "the architecture was discarded before training"
        )
    if arch == "gate_p2":
        if len(layers) != DETECT_LAYER_INDEX + 1:
            raise RuntimeError(
                f"P2 checkpoint has {len(layers)} layers, expected {DETECT_LAYER_INDEX + 1}"
            )
        if getattr(layers[-1], "nl", 0) != 4:
            raise RuntimeError("P2 checkpoint does not contain a four-level Detect head")
        strides = [float(value) for value in layers[-1].stride.tolist()]
        if strides != [float(value) for value in P2_STRIDES]:
            raise RuntimeError(f"P2 checkpoint strides are {strides}, expected {P2_STRIDES}")
        if params < 3_200_000 or params > 3_350_000:
            raise RuntimeError(
                f"gate+P2 checkpoint has {params:,} parameters, outside the expected range"
            )
        return True
    if arch == "gate_p5":
        if not hasattr(layers[P5_LAYER_INDEX], "block"):
            raise RuntimeError(
                "A+B checkpoint does not contain a LiteViT block at P5"
            )
        block = layers[P5_LAYER_INDEX].block
        if block.qkv.weight.detach().abs().sum().item() == 0.0:
            raise RuntimeError(
                "A+B block weights are zero; training likely produced NaN "
                "and the checkpoint was sanitized"
            )
        if params < 3_500_000 or params > 3_700_000:
            raise RuntimeError(
                f"A+B checkpoint has {params:,} parameters, outside the expected range"
            )
        return True
    if arch == "gate_edge_cbam":
        if not all(hasattr(layers[i], "edge") for i in EDGE_LAYER_INDICES):
            raise RuntimeError(
                "gate+edge+CBAM checkpoint does not contain edge branches at P3/P4"
            )
        if params < 3_150_000 or params > 3_300_000:
            raise RuntimeError(
                f"gate+edge+CBAM checkpoint has {params:,} parameters, outside the expected range"
            )
        return True
    if params < 3_150_000 or params > 3_250_000:
        raise RuntimeError(
            f"gate-only checkpoint has {params:,} parameters, outside the expected range"
        )
    return True


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


def build_metadata(
    root: Path, config: dict, save_dir: Path, vram: dict, arch_verified: bool
) -> dict:
    checksums = root / "data/processed/adverse16k/checksums.sha1"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": " ".join(sys.argv),
        "arch": config.get("arch", "baseline"),
        "arch_verified": arch_verified,
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
    arch = str(config.get("arch", "baseline"))
    trainer_cls = {
        "v1_p5": V1DetectionTrainer,
        "gate_only": GateDetectionTrainer,
        "gate_p2": GateP2DetectionTrainer,
        "gate_p5": GateP5DetectionTrainer,
        "gate_edge_cbam": GateEdgeCBAMTrainer,
    }.get(arch)
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
            resume_kwargs = {"resume": True}
            if trainer_cls is not None:
                resume_kwargs["trainer"] = trainer_cls
            model.train(**resume_kwargs)
        else:
            train_args = {
                key: value for key, value in config.items() if key not in {"model", "arch"}
            }
            train_args["name"] = run_name
            if epochs is not None:
                train_args["epochs"] = epochs
            if extra:
                train_args.update(extra)
            if arch not in {
                "baseline",
                "v1_p5",
                "gate_only",
                "gate_p2",
                "gate_p5",
                "gate_edge_cbam",
            }:
                raise ValueError(f"unsupported architecture: {arch}")
            model = YOLO(str(config["model"]))
            for event in ("on_train_batch_end", "on_fit_epoch_end", "on_train_end"):
                model.add_callback(event, tracker)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            train_kwargs = {"trainer": trainer_cls} if trainer_cls is not None else {}
            model.train(**train_kwargs, **train_args)
    except Exception:
        error_dir = Path(config["project"]) / run_name
        error_dir.mkdir(parents=True, exist_ok=True)
        (error_dir / "train_error.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
    finally:
        peak_used = _stop_gpu_sampler(sampler, sampler_path)
    vram = build_vram(tracker, baseline_used, peak_used)
    save_dir = Path(model.trainer.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    arch_verified = verify_saved_architecture(save_dir, arch)
    (save_dir / "vram.json").write_text(json.dumps(vram, indent=2), encoding="utf-8")
    (save_dir / "run_meta.json").write_text(
        json.dumps(build_metadata(root, config, save_dir, vram, arch_verified), indent=2),
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
