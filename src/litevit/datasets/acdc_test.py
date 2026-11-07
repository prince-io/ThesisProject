"""Prepare and verify the unlabeled ACDC test set for demos and final testing."""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from litevit.datasets.common import sha1_file

ACDC_TEST_CONDITIONS = ("fog", "night", "rain", "snow")


def prepare_acdc_test(root: Path, out_dir: Path, force: bool = False) -> dict:
    """Copy the 2,000 unlabeled ACDC test images into an independent directory."""
    if out_dir.exists():
        if not force:
            raise FileExistsError(f"{out_dir} exists; pass --force to rebuild")
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels_info").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    counts: Counter = Counter()
    categories: list[dict] = []
    for condition in ACDC_TEST_CONDITIONS:
        info_path = (
            root / "labels" / condition / f"instancesonly_{condition}_test_image_info.json"
        )
        payload = json.loads(info_path.read_text(encoding="utf-8"))
        if not categories:
            categories = payload.get("categories", [])
        shutil.copy2(info_path, out_dir / "labels_info" / info_path.name)
        for image in payload["images"]:
            file_name = image["file_name"].replace("\\", "/")
            source = root / "images" / file_name
            if not source.exists():
                raise FileNotFoundError(source)
            destination = out_dir / "images" / file_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            rows.append(
                {
                    "output_name": file_name,
                    "condition": condition,
                    "sequence": Path(file_name).parent.name,
                    "image_id": Path(file_name).stem,
                    "width": image["width"],
                    "height": image["height"],
                    "source_path": str(source),
                }
            )
            counts[condition] += 1

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "output_name",
                "condition",
                "sequence",
                "image_id",
                "width",
                "height",
                "source_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    stats = {
        "total_images": len(rows),
        "by_condition": dict(counts),
        "labeled": False,
        "categories": categories,
        "note": "ACDC test ground truth is withheld; use for demos and qualitative testing.",
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# ACDC test (unlabeled)\n\n"
        "2,000 anonymized ACDC adverse-condition test images (500 per condition).\n"
        "Annotations are withheld by the dataset authors, so this set is for demos,\n"
        "qualitative testing, and final inference only. Eight Cityscapes categories\n"
        "(person, rider, car, truck, bus, train, motorcycle, bicycle) are defined in\n"
        "`labels_info/` but no boxes are provided.\n",
        encoding="utf-8",
    )
    return stats


def verify_acdc_test(
    out_dir: Path,
    report_dir: Path,
    decode_images: bool = True,
    hash_images: bool = True,
) -> dict:
    errors: list[str] = []
    manifest_path = out_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    counts: Counter = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        counts[row["condition"]] += 1
        image_path = out_dir / "images" / row["output_name"]
        if not image_path.exists():
            errors.append(f"missing image: {image_path}")
            continue
        if image_path.is_symlink() or image_path.stat().st_nlink != 1:
            errors.append(f"not independent: {image_path}")
        if decode_images:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"cannot decode {image_path}: {exc}")
        if hash_images:
            hashes[sha1_file(image_path)].append(row["output_name"])

    duplicates = {
        digest: names for digest, names in hashes.items() if len(names) > 1
    }
    if duplicates:
        errors.append(f"{len(duplicates)} duplicate image hashes found")
    if len(rows) != 2000:
        errors.append(f"total images {len(rows)} != 2000")
    for condition in ACDC_TEST_CONDITIONS:
        if counts[condition] != 500:
            errors.append(f"{condition}: {counts[condition]} != 500")
    if (out_dir / "labels").exists():
        errors.append("unexpected labels directory in unlabeled set")

    report = {
        "total_images": len(rows),
        "by_condition": dict(counts),
        "duplicate_hashes": duplicates,
        "errors": errors,
        "passed": not errors,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
