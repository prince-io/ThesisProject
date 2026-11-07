"""Match predictions to ground truth and summarize errors by condition/source/class."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

from litevit.datasets.common import CLASS_NAMES, read_yolo_label

SIZE_BINS = (
    ("tiny", 0.0, 0.001),
    ("small", 0.001, 0.01),
    ("medium", 0.01, 0.05),
    ("large", 0.05, 1.01),
)


def _iou_matrix(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if len(first) == 0 or len(second) == 0:
        return np.zeros((len(first), len(second)))
    lt = np.maximum(first[:, None, :2], second[None, :, :2])
    rb = np.minimum(first[:, None, 2:], second[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_first = (first[:, 2] - first[:, 0]) * (first[:, 3] - first[:, 1])
    area_second = (second[:, 2] - second[:, 0]) * (second[:, 3] - second[:, 1])
    union = area_first[:, None] + area_second[None, :] - inter
    return np.divide(inter, np.maximum(union, 1e-9))


def _match(
    gt_boxes: np.ndarray,
    gt_classes: np.ndarray,
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    threshold: float,
    same_class: bool,
) -> tuple[list[tuple[int, int, float]], set[int]]:
    iou = _iou_matrix(gt_boxes, pred_boxes)
    candidates = []
    for gt_index in range(len(gt_boxes)):
        for pred_index in range(len(pred_boxes)):
            if iou[gt_index, pred_index] < threshold:
                continue
            if same_class and gt_classes[gt_index] != pred_classes[pred_index]:
                continue
            candidates.append((float(iou[gt_index, pred_index]), gt_index, pred_index))
    candidates.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches = []
    for iou_value, gt_index, pred_index in candidates:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append((gt_index, pred_index, iou_value))
    return matches, used_pred


def _gt_boxes(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    boxes = []
    classes = []
    for class_index, x_center, y_center, box_width, box_height in read_yolo_label(label_path):
        x1 = (x_center - box_width / 2) * width
        y1 = (y_center - box_height / 2) * height
        x2 = (x_center + box_width / 2) * width
        y2 = (y_center + box_height / 2) * height
        boxes.append([x1, y1, x2, y2])
        classes.append(class_index)
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(classes, dtype=int)


def _predict(
    model: YOLO, image_path: Path, conf: float, imgsz: int, device: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        iou=0.7,
        device=device,
        verbose=False,
        max_det=300,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,), dtype=int), np.zeros((0,))
    return (
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.cls.cpu().numpy().astype(int),
        result.boxes.conf.cpu().numpy(),
    )


def _size_bin(area: float) -> str:
    for name, low, high in SIZE_BINS:
        if low <= area < high:
            return name
    return "large"


def _bucket(stats: dict, key: str, gt: int, tp: int, fp: int, fn: int) -> None:
    entry = stats.setdefault(key, {"gt": 0, "tp": 0, "fp": 0, "fn": 0})
    entry["gt"] += gt
    entry["tp"] += tp
    entry["fp"] += fp
    entry["fn"] += fn


def _rates(entry: dict) -> dict:
    gt = entry["gt"]
    tp = entry["tp"]
    fp = entry["fp"]
    rates = {
        "gt": gt,
        "tp": tp,
        "fp": fp,
        "fn": entry["fn"],
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / gt if gt else 0.0,
    }
    if "ceiling_recall" in entry:
        rates["ceiling_recall"] = entry["ceiling_recall"]
    return rates


def _draw_overlay(
    image_path: Path,
    label_path: Path,
    pred_boxes: np.ndarray,
    pred_classes: np.ndarray,
    pred_conf: np.ndarray,
    width: int,
    height: int,
    output_path: Path,
) -> None:
    gt_boxes, gt_classes = _gt_boxes(label_path, width, height)
    with Image.open(image_path).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        for box, class_index in zip(gt_boxes, gt_classes, strict=True):
            draw.rectangle(box.tolist(), outline="lime", width=3)
            draw.text((box[0] + 2, box[1] + 2), f"GT {CLASS_NAMES[class_index]}", fill="lime")
        for box, class_index, confidence in zip(
            pred_boxes, pred_classes, pred_conf, strict=True
        ):
            draw.rectangle(box.tolist(), outline="red", width=2)
            draw.text(
                (box[0] + 2, box[1] + 12),
                f"{CLASS_NAMES[class_index]} {confidence:.2f}",
                fill="red",
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path, quality=85)


def analyze(
    weights: Path,
    manifest_path: Path,
    splits: tuple[str, ...],
    out_dir: Path,
    conf: float = 0.25,
    ceiling_conf: float = 0.001,
    iou_threshold: float = 0.5,
    imgsz: int = 640,
    device: int = 0,
    overlays: int = 5,
) -> dict:
    manifest_path = manifest_path.resolve()
    dataset_root = manifest_path.parent
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] in splits]
    model = YOLO(str(weights))
    out_dir.mkdir(parents=True, exist_ok=True)

    overall: dict[str, dict] = {}
    by_condition: dict[str, dict] = {}
    by_source: dict[str, dict] = {}
    by_class: dict[str, dict] = {}
    by_size: dict[str, dict] = {}
    ceiling: dict[str, dict] = {}
    ceiling_by_condition: dict[str, dict] = {}
    ceiling_by_source: dict[str, dict] = {}
    ceiling_by_class: dict[str, dict] = {}
    confusion: dict[str, Counter] = defaultdict(Counter)
    background_fp: Counter = Counter()
    fn_candidates: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    overlay_paths: list[str] = []

    for row in rows:
        split = row["split"]
        condition = row["condition"]
        source = row["source"]
        image_path = dataset_root / "images" / split / row["output_name"]
        label_path = dataset_root / "labels" / split / f"{Path(row['output_name']).stem}.txt"
        width = int(row["width"])
        height = int(row["height"])
        gt_boxes, gt_classes = _gt_boxes(label_path, width, height)

        pred_boxes, pred_classes, pred_conf = _predict(
            model, image_path, conf, imgsz, device
        )
        matches, _ = _match(
            gt_boxes, gt_classes, pred_boxes, pred_classes, iou_threshold, same_class=True
        )
        matched_gt = {match[0] for match in matches}
        tp = len(matches)
        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - tp
        _bucket(overall, split, len(gt_boxes), tp, fp, fn)
        _bucket(by_condition, f"{condition}/{split}", len(gt_boxes), tp, fp, fn)
        _bucket(by_source, f"{source}/{split}", len(gt_boxes), tp, fp, fn)

        for class_index in range(len(CLASS_NAMES)):
            gt_count = int((gt_classes == class_index).sum())
            pred_count = int((pred_classes == class_index).sum())
            class_tp = sum(
                1 for gt_index, _, _ in matches if gt_classes[gt_index] == class_index
            )
            if gt_count or pred_count:
                _bucket(
                    by_class,
                    f"{CLASS_NAMES[class_index]}/{split}",
                    gt_count,
                    class_tp,
                    pred_count - class_tp,
                    gt_count - class_tp,
                )

        for gt_index in range(len(gt_boxes)):
            area = (
                (gt_boxes[gt_index][2] - gt_boxes[gt_index][0])
                * (gt_boxes[gt_index][3] - gt_boxes[gt_index][1])
                / (width * height)
            )
            bin_name = _size_bin(area)
            matched = gt_index in matched_gt
            _bucket(
                by_size,
                f"{bin_name}/{split}",
                1,
                int(matched),
                0,
                int(not matched),
            )

        confusion_matches, confusion_used_pred = _match(
            gt_boxes, gt_classes, pred_boxes, pred_classes, iou_threshold, same_class=False
        )
        for gt_index, pred_index, _ in confusion_matches:
            confusion[CLASS_NAMES[int(gt_classes[gt_index])]][
                CLASS_NAMES[int(pred_classes[pred_index])]
            ] += 1
        for pred_index, pred_class in enumerate(pred_classes):
            if pred_index not in confusion_used_pred:
                background_fp[CLASS_NAMES[int(pred_class)]] += 1

        ceiling_pred_boxes, ceiling_pred_classes, _ = _predict(
            model, image_path, ceiling_conf, imgsz, device
        )
        ceiling_matches, _ = _match(
            gt_boxes,
            gt_classes,
            ceiling_pred_boxes,
            ceiling_pred_classes,
            iou_threshold,
            same_class=True,
        )
        ceiling_tp = len(ceiling_matches)
        ceiling_gt = len(gt_boxes)
        _bucket(ceiling, split, ceiling_gt, ceiling_tp, 0, ceiling_gt - ceiling_tp)
        _bucket(
            ceiling_by_condition,
            f"{condition}/{split}",
            ceiling_gt,
            ceiling_tp,
            0,
            ceiling_gt - ceiling_tp,
        )
        _bucket(
            ceiling_by_source,
            f"{source}/{split}",
            ceiling_gt,
            ceiling_tp,
            0,
            ceiling_gt - ceiling_tp,
        )
        for class_index in range(len(CLASS_NAMES)):
            gt_count = int((gt_classes == class_index).sum())
            class_tp = sum(
                1
                for gt_index, _, _ in ceiling_matches
                if gt_classes[gt_index] == class_index
            )
            if gt_count:
                _bucket(
                    ceiling_by_class,
                    f"{CLASS_NAMES[class_index]}/{split}",
                    gt_count,
                    class_tp,
                    0,
                    gt_count - class_tp,
                )

        fn_candidates[(split, condition)].append((fn, row))

    for (split, condition), candidates in fn_candidates.items():
        candidates.sort(key=lambda item: (-item[0], item[1]["output_name"]))
        for _, row in candidates[:overlays]:
            image_path = dataset_root / "images" / split / row["output_name"]
            label_path = (
                dataset_root / "labels" / split / f"{Path(row['output_name']).stem}.txt"
            )
            pred_boxes, pred_classes, pred_conf = _predict(
                model, image_path, conf, imgsz, device
            )
            output_path = (
                out_dir
                / "overlays"
                / f"{split}_{condition}_{Path(row['output_name']).stem}.jpg"
            )
            _draw_overlay(
                image_path,
                label_path,
                pred_boxes,
                pred_classes,
                pred_conf,
                int(row["width"]),
                int(row["height"]),
                output_path,
            )
            overlay_paths.append(str(output_path))

    def with_rates(group: dict[str, dict]) -> dict[str, dict]:
        return {key: _rates(value) for key, value in group.items()}

    ceiling_rates = with_rates(ceiling)
    for group, ceiling_group in (
        (by_condition, ceiling_by_condition),
        (by_source, ceiling_by_source),
        (by_class, ceiling_by_class),
    ):
        for key, entry in group.items():
            ceiling_entry = ceiling_group.get(key, {"gt": 0, "tp": 0})
            entry["ceiling_recall"] = (
                ceiling_entry["tp"] / ceiling_entry["gt"] if ceiling_entry["gt"] else 0.0
            )

    payload = {
        "weights": str(weights),
        "splits": list(splits),
        "thresholds": {
            "confidence": conf,
            "ceiling_confidence": ceiling_conf,
            "iou": iou_threshold,
            "imgsz": imgsz,
        },
        "overall": with_rates(overall),
        "by_condition": with_rates(by_condition),
        "by_source": with_rates(by_source),
        "by_class": with_rates(by_class),
        "by_size": with_rates(by_size),
        "ceiling": ceiling_rates,
        "ceiling_by_condition": with_rates(ceiling_by_condition),
        "ceiling_by_source": with_rates(ceiling_by_source),
        "ceiling_by_class": with_rates(ceiling_by_class),
        "confusion": {key: dict(value) for key, value in confusion.items()},
        "background_fp": dict(background_fp),
        "overlays": overlay_paths,
    }
    (out_dir / "error_analysis.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    for name in ("by_condition", "by_source", "by_class", "by_size"):
        with (out_dir / f"error_{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["key", "gt", "tp", "fp", "fn", "precision", "recall", "ceiling_recall"]
            )
            for key, entry in sorted(payload[name].items()):
                writer.writerow(
                    [
                        key,
                        entry["gt"],
                        entry["tp"],
                        entry["fp"],
                        entry["fn"],
                        entry["precision"],
                        entry["recall"],
                        entry.get("ceiling_recall", ""),
                    ]
                )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze detection errors")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/adverse16k/manifest.csv"),
    )
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--ceiling-conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--overlays", type=int, default=5)
    args = parser.parse_args()
    payload = analyze(
        weights=args.weights,
        manifest_path=args.manifest,
        splits=tuple(args.splits),
        out_dir=args.out,
        conf=args.conf,
        ceiling_conf=args.ceiling_conf,
        iou_threshold=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        overlays=args.overlays,
    )
    print(json.dumps(payload["overall"], indent=2))


if __name__ == "__main__":
    main()
