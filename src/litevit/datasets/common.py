"""Shared dataset types, class mappings, and serialization helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

CLASS_NAMES = ("person", "car", "bus", "bicycle", "motorcycle")
CLASS_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

CONDITIONS = ("fog", "rain", "snow", "night")

# Truck is merged into car. Rider is merged into person. Train is dropped.
ACDC_MAP = {
    "person": 0,
    "rider": 0,
    "car": 1,
    "truck": 1,
    "bus": 2,
    "bicycle": 3,
    "motorcycle": 4,
}
DAWN_MAP = {
    "person": 0,
    "car": 1,
    "truck": 1,
    "bus": 2,
    "bicycle": 3,
    "motorcycle": 4,
}
BDD_MAP = {
    "pedestrian": 0,
    "person": 0,
    "rider": 0,
    "car": 1,
    "truck": 1,
    "bus": 2,
    "bicycle": 3,
    "bike": 3,
    "motorcycle": 4,
    "motorbike": 4,
    "motor": 4,
}
RTTS_MAP = {
    "person": 0,
    "car": 1,
    "bus": 2,
    "bicycle": 3,
    "motorbike": 4,
}


@dataclass(frozen=True)
class Box:
    """A detection box in absolute pixel coordinates."""

    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    def clipped(self, width: int, height: int) -> Box | None:
        x1 = min(max(self.x1, 0.0), float(width))
        y1 = min(max(self.y1, 0.0), float(height))
        x2 = min(max(self.x2, 0.0), float(width))
        y2 = min(max(self.y2, 0.0), float(height))
        if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
            return None
        return Box(self.cls, x1, y1, x2, y2)

    def to_yolo(self, width: int, height: int) -> tuple[float, float, float, float]:
        x_center = ((self.x1 + self.x2) / 2.0) / width
        y_center = ((self.y1 + self.y2) / 2.0) / height
        box_width = (self.x2 - self.x1) / width
        box_height = (self.y2 - self.y1) / height
        return (
            min(max(x_center, 0.0), 1.0),
            min(max(y_center, 0.0), 1.0),
            min(max(box_width, 0.0), 1.0),
            min(max(box_height, 0.0), 1.0),
        )


@dataclass
class Sample:
    """A single image and its normalized annotations."""

    source: str
    source_split: str
    condition: str
    image_path: Path
    width: int
    height: int
    boxes: list[Box]
    image_id: str
    weather: str = ""
    timeofday: str = ""
    sequence_id: str = ""
    split_hint: str = ""
    output_name: str = ""

    @property
    def num_boxes(self) -> int:
        return len(self.boxes)

    def class_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for box in self.boxes:
            counts[box.cls] = counts.get(box.cls, 0) + 1
        return counts


def bdd_condition(weather: str, timeofday: str) -> str | None:
    """Assign a BDD100K image to exactly one benchmark condition."""
    weather = (weather or "").strip().lower()
    timeofday = (timeofday or "").strip().lower()
    if weather == "foggy":
        return "fog"
    if weather == "snowy":
        return "snow"
    if weather == "rainy":
        return "rain"
    if timeofday == "night" and weather in {
        "clear",
        "partly cloudy",
        "overcast",
        "undefined",
        "",
    }:
        return "night"
    return None


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def link_or_copy(source: Path, destination: Path, copy_only: bool = False) -> None:
    """Hardlink by default, or copy when an independent dataset is requested."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if not copy_only:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def write_yolo_label(path: Path, boxes: list[Box], width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for box in boxes:
        x_center, y_center, box_width, box_height = box.to_yolo(width, height)
        lines.append(
            f"{box.cls} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_yolo_label(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        rows.append((int(parts[0]), *(float(value) for value in parts[1:5])))
    return rows


def write_coco(
    samples: list[Sample], output_path: Path, category_names: tuple[str, ...] = CLASS_NAMES
) -> None:
    images = []
    annotations = []
    annotation_id = 1
    for image_id, sample in enumerate(samples, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": sample.output_name,
                "width": sample.width,
                "height": sample.height,
            }
        )
        for box in sample.boxes:
            width = box.x2 - box.x1
            height = box.y2 - box.y1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": box.cls + 1,
                    "bbox": [box.x1, box.y1, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": index + 1, "name": name} for index, name in enumerate(category_names)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload), encoding="utf-8")


def dhash_image(path: Path, size: int = 8) -> int:
    """Compute a 64-bit difference hash for near-duplicate grouping."""
    with Image.open(path).convert("L").resize((size + 1, size)) as image:
        pixels = list(image.get_flattened_data())
    bits = 0
    for row in range(size):
        for col in range(size):
            left = pixels[row * (size + 1) + col]
            right = pixels[row * (size + 1) + col + 1]
            bits = (bits << 1) | int(left > right)
    return bits


def sha1_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()
