"""ACDC COCO detection loader for adverse-condition train/val images."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from litevit.datasets.common import ACDC_MAP, Box, Sample

_ACDC_CONDITIONS = ("fog", "night", "rain", "snow")
_ACDC_SPLITS = ("train", "val")


def load_acdc(root: Path) -> list[Sample]:
    """Load ACDC adverse train/val samples, deduplicating the combined rain file."""
    samples: list[Sample] = []
    seen_images: set[str] = set()
    for condition in _ACDC_CONDITIONS:
        for split in _ACDC_SPLITS:
            label_path = (
                root
                / "labels"
                / condition
                / f"instancesonly_{condition}_{split}_gt_detection.json"
            )
            if not label_path.exists():
                continue
            payload = json.loads(label_path.read_text(encoding="utf-8"))
            categories = {item["id"]: item["name"] for item in payload["categories"]}
            images = {
                image["id"]: image
                for image in payload["images"]
                if image["file_name"].replace("\\", "/").split("/")[0] == condition
            }
            annotations: dict[int, list[dict]] = defaultdict(list)
            for annotation in payload["annotations"]:
                if annotation["image_id"] in images:
                    annotations[annotation["image_id"]].append(annotation)
            for image_id, image in images.items():
                file_name = image["file_name"].replace("\\", "/")
                image_path = root / "images" / file_name
                if not image_path.exists():
                    continue
                key = str(image_path)
                if key in seen_images:
                    continue
                width = int(image["width"])
                height = int(image["height"])
                boxes: list[Box] = []
                for annotation in annotations.get(image_id, []):
                    class_index = ACDC_MAP.get(categories.get(annotation["category_id"], ""))
                    if class_index is None:
                        continue
                    x, y, box_width, box_height = annotation["bbox"]
                    box = Box(class_index, x, y, x + box_width, y + box_height).clipped(
                        width, height
                    )
                    if box is not None:
                        boxes.append(box)
                if not boxes:
                    continue
                seen_images.add(key)
                samples.append(
                    Sample(
                        source="acdc",
                        source_split=split,
                        condition=condition,
                        image_path=image_path,
                        width=width,
                        height=height,
                        boxes=boxes,
                        image_id=Path(file_name).stem,
                        sequence_id=Path(file_name).parent.name,
                    )
                )
    return samples
