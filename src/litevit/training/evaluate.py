"""Evaluate a detector overall and on condition/source subsets."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


def _metric_dict(metrics) -> dict:
    names = metrics.names
    box = metrics.box
    per_class: dict[str, dict] = {}
    for class_index, ap50, ap in zip(
        box.ap_class_index, box.ap50, box.ap, strict=True
    ):
        per_class[names[int(class_index)]] = {
            "ap50": float(ap50),
            "ap50_95": float(ap),
        }
    for name in names.values():
        per_class.setdefault(name, {"ap50": None, "ap50_95": None})
    targets = {}
    if metrics.nt_per_class is not None:
        for index, name in names.items():
            targets[name] = int(metrics.nt_per_class[index])
    return {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "per_class": per_class,
        "targets_per_class": targets,
        "speed_ms": {key: float(value) for key, value in metrics.speed.items()},
        "results_dict": {key: float(value) for key, value in metrics.results_dict.items()},
    }


def _copy_figures(out_dir: Path, splits: tuple[str, ...], figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for split in splits:
        source_dir = out_dir / f"overall_{split}"
        for name in (
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "BoxPR_curve.png",
            "BoxF1_curve.png",
            "BoxP_curve.png",
            "BoxR_curve.png",
            "results.png",
        ):
            source = source_dir / name
            if source.exists():
                destination = figures_dir / f"{split}_{name}"
                shutil.copy2(source, destination)
                copied.append(str(destination))
    return copied


def _write_csvs(payload: dict, out_dir: Path) -> None:
    with (out_dir / "per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "class", "ap50", "ap50_95"])
        for split, metrics in payload["overall"].items():
            for name, values in metrics["per_class"].items():
                writer.writerow([split, name, values["ap50"], values["ap50_95"]])

    for group in ("per_condition", "per_source"):
        with (out_dir / f"{group}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["name", "split", "precision", "recall", "map50", "map50_95"])
            for name, splits in payload[group].items():
                for split, metrics in splits.items():
                    writer.writerow(
                        [
                            name,
                            split,
                            metrics["precision"],
                            metrics["recall"],
                            metrics["map50"],
                            metrics["map50_95"],
                        ]
                    )


def evaluate(
    weights: Path,
    data: Path,
    splits: tuple[str, ...],
    out_dir: Path,
    subsets_dir: Path | None = None,
    figures_dir: Path | None = None,
    imgsz: int = 640,
    batch: int = 4,
    device: int | str = 0,
    augment: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    payload: dict = {
        "weights": str(weights),
        "data": str(data),
        "splits": list(splits),
        "augment": augment,
        "overall": {},
        "per_condition": {},
        "per_source": {},
    }

    for split in splits:
        metrics = model.val(
            data=str(data),
            split=split,
            imgsz=imgsz,
            batch=batch,
            device=device,
            plots=True,
            augment=augment,
            project=str(out_dir.resolve()),
            name=f"overall_{split}",
            exist_ok=True,
            verbose=False,
        )
        payload["overall"][split] = _metric_dict(metrics)

    if subsets_dir is not None:
        subset_payload = json.loads(
            (subsets_dir / "subsets.json").read_text(encoding="utf-8")
        )
        for key, info in subset_payload.items():
            split, name = key.split("/", 1)
            if split not in splits or name == "all":
                continue
            metrics = model.val(
                data=info["yaml"],
                split="val",
                imgsz=imgsz,
                batch=batch,
                device=device,
                plots=True,
                augment=augment,
                project=str(out_dir.resolve()),
                name=f"{split}_{name}",
                exist_ok=True,
                verbose=False,
            )
            if name.startswith("cond_"):
                payload["per_condition"].setdefault(name[5:], {})[split] = _metric_dict(
                    metrics
                )
            elif name.startswith("source_"):
                payload["per_source"].setdefault(name[7:], {})[split] = _metric_dict(
                    metrics
                )

    first_split = payload["overall"][splits[0]]
    inference_ms = first_split["speed_ms"].get("inference", 0.0)
    payload["efficiency"] = {
        "params_m": round(sum(p.numel() for p in model.model.parameters()) / 1e6, 3),
        "gflops": round(float(get_flops(model.model, imgsz=imgsz)), 3),
        "latency_ms": inference_ms,
        "fps": round(1000.0 / inference_ms, 2) if inference_ms > 0 else None,
        "peak_vram_mb": (
            round(torch.cuda.max_memory_allocated() / 1024**2, 1)
            if torch.cuda.is_available()
            else None
        ),
    }
    payload["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["python"] = sys.version.split()[0]
    payload["torch"] = torch.__version__
    payload["ultralytics"] = ultralytics.__version__
    payload["gpu"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    )

    (out_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    _write_csvs(payload, out_dir)

    if figures_dir is not None:
        payload["figures"] = _copy_figures(out_dir, splits, figures_dir)
        (out_dir / "metrics.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a detector")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("configs/datasets/adverse16k.yaml"))
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subsets", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default=0)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()
    payload = evaluate(
        weights=args.weights,
        data=args.data,
        splits=tuple(args.splits),
        out_dir=args.out,
        subsets_dir=args.subsets,
        figures_dir=args.figures_dir,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        augment=args.augment,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
