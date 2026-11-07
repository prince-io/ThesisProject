"""Verification gates for the adverse16k dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from litevit.datasets.common import (
    CLASS_NAMES,
    CONDITIONS,
    NUM_CLASSES,
    read_yolo_label,
    sha1_file,
)

SPLITS = ("train", "val", "test")
CLASS_COLORS = {0: "red", 1: "lime", 2: "cyan", 3: "yellow", 4: "magenta"}


def verify(
    root: Path,
    out_dir: Path,
    report_dir: Path,
    hash_images: bool = True,
    decode_images: bool = True,
    overlays: int = 2,
    require_independent: bool = False,
    write_checksums: bool = False,
) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    manifest_path = out_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_split: Counter = Counter()
    by_condition: Counter = Counter()
    by_source: Counter = Counter()
    by_split_condition: Counter = Counter()
    by_condition_source: Counter = Counter()
    class_instances: Counter = Counter()
    weather_missing = 0

    for row in rows:
        split = row["split"]
        condition = row["condition"]
        source = row["source"]
        by_split[split] += 1
        by_condition[condition] += 1
        by_source[source] += 1
        by_split_condition[f"{split}/{condition}"] += 1
        by_condition_source[f"{condition}/{source}"] += 1
        if source == "bdd" and (not row["weather"] or not row["timeofday"]):
            weather_missing += 1

        image_path = out_dir / "images" / split / row["output_name"]
        label_path = out_dir / "labels" / split / f"{Path(row['output_name']).stem}.txt"
        if not image_path.exists():
            errors.append(f"missing image: {image_path}")
            continue
        if require_independent:
            if image_path.is_symlink() or not image_path.is_file():
                errors.append(f"not a regular file: {image_path}")
            elif image_path.stat().st_nlink != 1:
                errors.append(
                    f"not independent (links={image_path.stat().st_nlink}): {image_path}"
                )
        if not label_path.exists():
            errors.append(f"missing label: {label_path}")
            continue
        labels = read_yolo_label(label_path)
        if not labels:
            errors.append(f"empty label: {label_path}")
        for class_index, x_center, y_center, width, height in labels:
            class_instances[CLASS_NAMES[class_index]] += 1
            if not 0 <= class_index < NUM_CLASSES:
                errors.append(f"bad class {class_index} in {label_path}")
            if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
                errors.append(f"bad center in {label_path}")
            if not (0.0 < width <= 1.0 and 0.0 < height <= 1.0):
                errors.append(f"bad size in {label_path}")
            if x_center - width / 2 < -1e-4 or x_center + width / 2 > 1.0001:
                errors.append(f"box exceeds x bounds in {label_path}")
            if y_center - height / 2 < -1e-4 or y_center + height / 2 > 1.0001:
                errors.append(f"box exceeds y bounds in {label_path}")
        if decode_images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"cannot decode {image_path}: {exc}")

    if len(rows) != 16000:
        errors.append(f"total images {len(rows)} != 16000")
    for split, expected in (("train", 12800), ("val", 1600), ("test", 1600)):
        if by_split[split] != expected:
            errors.append(f"split {split}: {by_split[split]} != {expected}")
    for condition in CONDITIONS:
        if by_condition[condition] != 4000:
            errors.append(f"condition {condition}: {by_condition[condition]} != 4000")
        for split, expected in (("train", 3200), ("val", 400), ("test", 400)):
            actual = by_split_condition[f"{split}/{condition}"]
            if actual != expected:
                errors.append(
                    f"{condition}/{split}: {actual} != {expected}"
                )

    if weather_missing:
        errors.append(f"{weather_missing} BDD samples missing weather/timeofday")

    duplicate_report: dict[str, list[dict[str, str]]] = {}
    checksums_path: Path | None = None
    if hash_images:
        hashes: dict[str, list[dict[str, str]]] = defaultdict(list)
        checksum_rows: list[tuple[str, str]] = []
        for row in rows:
            image_path = out_dir / "images" / row["split"] / row["output_name"]
            if image_path.exists():
                digest = sha1_file(image_path)
                hashes[digest].append(row)
                relative = f"images/{row['split']}/{row['output_name']}"
                checksum_rows.append((relative, digest))
        for digest, items in hashes.items():
            if len(items) > 1:
                duplicate_report[digest] = [
                    {"output_name": item["output_name"], "split": item["split"]}
                    for item in items
                ]
        if duplicate_report:
            errors.append(f"{len(duplicate_report)} duplicate image hashes found")
        if write_checksums:
            checksums_path = out_dir / "checksums.sha1"
            checksum_rows.sort(key=lambda item: item[0])
            checksums_path.write_text(
                "\n".join(f"{digest}  {relative}" for relative, digest in checksum_rows) + "\n",
                encoding="utf-8",
            )

    report_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = report_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    if overlays:
        seen: Counter = Counter()
        for row in rows:
            key = (row["source"], row["condition"])
            if seen[key] >= overlays:
                continue
            seen[key] += 1
            image_path = out_dir / "images" / row["split"] / row["output_name"]
            label_path = out_dir / "labels" / row["split"] / f"{Path(row['output_name']).stem}.txt"
            if not image_path.exists() or not label_path.exists():
                continue
            with Image.open(image_path).convert("RGB") as image:
                draw = ImageDraw.Draw(image)
                for class_index, x_center, y_center, width, height in read_yolo_label(
                    label_path
                ):
                    x1 = (x_center - width / 2) * image.width
                    y1 = (y_center - height / 2) * image.height
                    x2 = (x_center + width / 2) * image.width
                    y2 = (y_center + height / 2) * image.height
                    draw.rectangle(
                        [x1, y1, x2, y2],
                        outline=CLASS_COLORS.get(class_index, "white"),
                        width=2,
                    )
                output_path = (
                    overlay_dir
                    / f"{row['source']}_{row['condition']}_{row['output_name']}.jpg"
                )
                image.save(output_path, quality=85)

    report = {
        "total_images": len(rows),
        "by_split": dict(by_split),
        "by_condition": dict(by_condition),
        "by_source": dict(by_source),
        "by_split_condition": dict(by_split_condition),
        "by_condition_source": dict(by_condition_source),
        "class_instances": dict(class_instances),
        "duplicate_hashes": duplicate_report,
        "independent_required": require_independent,
        "checksums_written": str(checksums_path) if checksums_path else None,
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }
    (report_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the adverse16k dataset")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("data/processed/adverse16k"))
    parser.add_argument("--report", type=Path, default=Path("runs/dataset_checks/adverse16k"))
    parser.add_argument("--no-hash", action="store_true")
    parser.add_argument("--no-decode", action="store_true")
    parser.add_argument(
        "--require-independent",
        action="store_true",
        help="require every processed image to be a regular file with link count 1",
    )
    parser.add_argument(
        "--checksums",
        action="store_true",
        help="write checksums.sha1 for the processed images",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out if args.out.is_absolute() else root / args.out
    report_dir = args.report if args.report.is_absolute() else root / args.report
    report = verify(
        root,
        out_dir,
        report_dir,
        hash_images=not args.no_hash,
        decode_images=not args.no_decode,
        require_independent=args.require_independent,
        write_checksums=args.checksums,
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
