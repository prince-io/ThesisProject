"""Audit the adverse16k dataset for distribution, leakage, and label quality."""

from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from litevit.datasets.common import (  # noqa: E402
    ACDC_MAP,
    BDD_MAP,
    CLASS_NAMES,
    DAWN_MAP,
    RTTS_MAP,
    dhash_image,
    read_yolo_label,
)


def _percentiles(values: list[float], points: tuple[int, ...] = (0, 10, 50, 90, 100)) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    result = {}
    for point in points:
        index = round((len(ordered) - 1) * point / 100)
        result[f"p{point}"] = ordered[index]
    result["mean"] = sum(ordered) / len(ordered)
    return result


def _dropped_and_merged(root: Path, rows: list[dict]) -> tuple[dict, dict]:
    dropped: Counter = Counter()
    merged: Counter = Counter()

    for condition in ("fog", "night", "rain", "snow"):
        for split in ("train", "val"):
            path = (
                root
                / "ref/acdc/labels"
                / condition
                / f"instancesonly_{condition}_{split}_gt_detection.json"
            )
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            categories = {item["id"]: item["name"] for item in payload["categories"]}
            image_condition = {
                image["id"]: image["file_name"].split("/")[0] for image in payload["images"]
            }
            for annotation in payload["annotations"]:
                if image_condition.get(annotation["image_id"]) != condition:
                    continue
                name = categories[annotation["category_id"]]
                if name not in ACDC_MAP:
                    dropped[f"acdc/{condition}/{name}"] += 1
                elif name in {"truck", "rider"}:
                    merged[f"acdc/{condition}/{name}"] += 1

    for folder in ("Fog", "Rain", "Snow"):
        for xml_path in (root / "ref/dawn" / folder / f"{folder}_PASCAL_VOC").glob("*.xml"):
            for obj in ET.parse(xml_path).getroot().findall("object"):
                name = (obj.findtext("name") or "").strip().lower()
                condition = folder.lower()
                if name not in DAWN_MAP:
                    dropped[f"dawn/{condition}/{name}"] += 1
                elif name == "truck":
                    merged[f"dawn/{condition}/{name}"] += 1

    for split in ("train", "val"):
        path = root / "ref/rtts/RTTS/annotations_json" / f"rtts_{split}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        categories = {item["id"]: item["name"] for item in payload["categories"]}
        for annotation in payload["annotations"]:
            name = categories[annotation["category_id"]]
            if name not in RTTS_MAP:
                dropped[f"rtts/fog/{name}"] += 1

    bdd_zip = root / "ref/bdd100k/bdd100k_det_20_labels_trainval.zip"
    if bdd_zip.exists():
        with zipfile.ZipFile(bdd_zip) as archive:
            for row in rows:
                if row["source"] != "bdd":
                    continue
                entry = (
                    f"bdd100k/labels/100k/{row['source_split']}/{row['image_id']}.json"
                )
                payload = json.loads(archive.read(entry))
                frames = payload.get("frames") or []
                if not frames:
                    continue
                for obj in frames[0].get("objects", []):
                    name = obj.get("category", "")
                    if name not in BDD_MAP:
                        dropped[f"bdd/{row['condition']}/{name}"] += 1
                    elif name == "truck":
                        merged[f"bdd/{row['condition']}/{name}"] += 1

    return dict(dropped), dict(merged)


def _near_duplicates(rows: list[dict], out_dir: Path, threshold: int = 6) -> dict:
    dawn = [row for row in rows if row["source"] == "dawn"]
    hashes = []
    for row in dawn:
        image_path = out_dir / "images" / row["split"] / row["output_name"]
        if image_path.exists():
            hashes.append((row, dhash_image(image_path)))
    pairs = []
    cross_split = []
    for index, (row_a, hash_a) in enumerate(hashes):
        for row_b, hash_b in hashes[index + 1 :]:
            if (hash_a ^ hash_b).bit_count() <= threshold:
                pairs.append((row_a["output_name"], row_b["output_name"]))
                if row_a["split"] != row_b["split"]:
                    cross_split.append(
                        (row_a["output_name"], row_a["split"], row_b["output_name"], row_b["split"])
                    )
    return {
        "threshold_hamming": threshold,
        "pairs": len(pairs),
        "cross_split_pairs": len(cross_split),
        "examples": cross_split[:20],
    }


def _plots(
    class_by_condition: dict[str, Counter],
    source_by_condition: Counter,
    box_areas: list[float],
    objects_per_image: list[int],
    plot_dir: Path,
) -> list[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    conditions = list(class_by_condition)
    bottom = [0] * len(conditions)
    plt.figure(figsize=(8, 4))
    for name in CLASS_NAMES:
        values = [class_by_condition[c][name] for c in conditions]
        plt.bar(conditions, values, bottom=bottom, label=name)
        bottom = [a + b for a, b in zip(bottom, values, strict=True)]
    plt.ylabel("instances")
    plt.title("Class instances by condition")
    plt.legend()
    plt.tight_layout()
    path = plot_dir / "class_by_condition.png"
    plt.savefig(path)
    plt.close()
    paths.append(str(path))

    pairs = sorted({(condition, source) for (condition, source) in source_by_condition})
    conditions = sorted({condition for condition, _ in pairs})
    sources = sorted({source for _, source in pairs})
    bottom = [0] * len(conditions)
    plt.figure(figsize=(8, 4))
    for source in sources:
        values = [source_by_condition[(condition, source)] for condition in conditions]
        plt.bar(conditions, values, bottom=bottom, label=source)
        bottom = [a + b for a, b in zip(bottom, values, strict=True)]
    plt.ylabel("images")
    plt.title("Source mix by condition")
    plt.legend()
    plt.tight_layout()
    path = plot_dir / "source_by_condition.png"
    plt.savefig(path)
    plt.close()
    paths.append(str(path))

    plt.figure(figsize=(8, 4))
    plt.hist([area for area in box_areas if area > 0], bins=50, log=True)
    plt.xlabel("relative box area")
    plt.ylabel("count (log)")
    plt.title("Box area distribution")
    plt.tight_layout()
    path = plot_dir / "box_area.png"
    plt.savefig(path)
    plt.close()
    paths.append(str(path))

    plt.figure(figsize=(8, 4))
    plt.hist(objects_per_image, bins=50, log=True)
    plt.xlabel("objects per image")
    plt.ylabel("count (log)")
    plt.title("Objects per image")
    plt.tight_layout()
    path = plot_dir / "objects_per_image.png"
    plt.savefig(path)
    plt.close()
    paths.append(str(path))
    return paths


def audit(root: Path, out_dir: Path, report_dir: Path) -> dict:
    with (out_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    by_split: Counter = Counter()
    by_condition: Counter = Counter()
    by_source: Counter = Counter()
    source_by_condition: Counter = Counter()
    class_by_split: dict[str, Counter] = defaultdict(Counter)
    class_by_condition: dict[str, Counter] = defaultdict(Counter)
    class_by_source: dict[str, Counter] = defaultdict(Counter)
    resolutions: Counter = Counter()
    objects_per_image: list[int] = []
    box_areas: list[float] = []
    sequence_splits: dict[tuple[str, str], set[str]] = defaultdict(set)

    for row in rows:
        split = row["split"]
        condition = row["condition"]
        source = row["source"]
        by_split[split] += 1
        by_condition[condition] += 1
        by_source[source] += 1
        source_by_condition[(condition, source)] += 1
        resolutions[f"{row['width']}x{row['height']}"] += 1
        objects_per_image.append(int(row["num_boxes"]))
        if row["sequence"]:
            sequence_splits[(source, row["sequence"])].add(split)
        label_path = out_dir / "labels" / split / f"{Path(row['output_name']).stem}.txt"
        for class_index, _x, _y, width, height in read_yolo_label(label_path):
            class_name = CLASS_NAMES[class_index]
            class_by_split[split][class_name] += 1
            class_by_condition[condition][class_name] += 1
            class_by_source[source][class_name] += 1
            box_areas.append(width * height)

    sequence_leakage = {
        f"{source}/{sequence}": sorted(splits)
        for (source, sequence), splits in sequence_splits.items()
        if len(splits) > 1
    }
    dropped, merged = _dropped_and_merged(root, rows)
    near_duplicates = _near_duplicates(rows, out_dir)

    report_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = _plots(
        class_by_condition,
        source_by_condition,
        box_areas,
        objects_per_image,
        report_dir / "plots",
    )

    report = {
        "total_images": len(rows),
        "by_split": dict(by_split),
        "by_condition": dict(by_condition),
        "by_source": dict(by_source),
        "source_by_condition": {
            f"{condition}/{source}": count
            for (condition, source), count in sorted(source_by_condition.items())
        },
        "class_by_split": {k: dict(v) for k, v in class_by_split.items()},
        "class_by_condition": {k: dict(v) for k, v in class_by_condition.items()},
        "class_by_source": {k: dict(v) for k, v in class_by_source.items()},
        "objects_per_image": _percentiles([float(v) for v in objects_per_image]),
        "box_area_relative": _percentiles(box_areas),
        "resolutions": dict(resolutions),
        "sequence_leakage": sequence_leakage,
        "dropped_annotations": dropped,
        "merged_annotations": merged,
        "dawn_near_duplicates": near_duplicates,
        "plots": plot_paths,
    }
    (report_dir / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# adverse16k audit",
        "",
        f"- total images: {report['total_images']}",
        f"- by split: {report['by_split']}",
        f"- by condition: {report['by_condition']}",
        f"- by source: {report['by_source']}",
        f"- sequence leakage: {len(sequence_leakage)}",
        "- duplicate hashes: checked by verify.py",
        f"- DAWN near-duplicate pairs: {near_duplicates['pairs']} "
        f"(cross-split: {near_duplicates['cross_split_pairs']})",
        f"- objects/image: {report['objects_per_image']}",
        f"- box area: {report['box_area_relative']}",
        f"- resolutions: {report['resolutions']}",
        "",
        "## Dropped annotations (not in the 5-class target)",
        "",
    ]
    for key, count in sorted(dropped.items()):
        lines.append(f"- {key}: {count}")
    lines += ["", "## Merged annotations", ""]
    for key, count in sorted(merged.items()):
        lines.append(f"- {key}: {count}")
    lines += ["", "## Plots", ""]
    for path in plot_paths:
        lines.append(f"- {path}")
    (report_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the adverse16k dataset")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("data/processed/adverse16k"))
    parser.add_argument(
        "--report", type=Path, default=Path("runs/dataset_checks/adverse16k")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out if args.out.is_absolute() else root / args.out
    report_dir = args.report if args.report.is_absolute() else root / args.report
    report = audit(root, out_dir, report_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
