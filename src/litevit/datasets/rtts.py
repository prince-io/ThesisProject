"""RTTS (RESIDE real-world foggy detection) loader."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from litevit.datasets.common import RTTS_MAP, Box, Sample


def load_rtts(root: Path) -> list[Sample]:
    """Load RTTS train/val COCO annotations (5 classes, fog only)."""
    samples: list[Sample] = []
    for split in ("train", "val"):
        label_path = root / "annotations_json" / f"rtts_{split}.json"
        if not label_path.exists():
            continue
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        categories = {item["id"]: item["name"] for item in payload["categories"]}
        annotations: dict[int, list[dict]] = defaultdict(list)
        for annotation in payload["annotations"]:
            annotations[annotation["image_id"]].append(annotation)
        for image in payload["images"]:
            file_name = image["file_name"].replace("\\", "/")
            image_path = root / file_name
            if not image_path.exists():
                continue
            width = int(image["width"])
            height = int(image["height"])
            boxes: list[Box] = []
            for annotation in annotations.get(image["id"], []):
                class_index = RTTS_MAP.get(categories.get(annotation["category_id"], ""))
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
            samples.append(
                Sample(
                    source="rtts",
                    source_split=split,
                    condition="fog",
                    image_path=image_path,
                    width=width,
                    height=height,
                    boxes=boxes,
                    image_id=Path(file_name).stem,
                )
            )
    return samples
