"""BDD100K loader: 2020 detection boxes joined with 2018 weather/time attributes."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import ijson

from litevit.datasets.common import BDD_MAP, Box, Sample, bdd_condition, image_size


def load_bdd_attributes(labels_dir: Path) -> dict[str, tuple[str, str]]:
    """Stream the deprecated 2018 release for frame-level weather/time attributes."""
    attributes: dict[str, tuple[str, str]] = {}
    for split in ("train", "val"):
        path = labels_dir / f"bdd100k_labels_images_{split}.json"
        if not path.exists():
            continue
        with path.open("rb") as handle:
            for frame in ijson.items(handle, "item"):
                name = frame.get("name", "")
                stem = name[:-4] if name.lower().endswith(".jpg") else name
                frame_attributes = frame.get("attributes") or {}
                attributes[stem] = (
                    frame_attributes.get("weather", ""),
                    frame_attributes.get("timeofday", ""),
                )
    return attributes


def build_image_index(images_root: Path) -> dict[str, Path]:
    """Index every 100k image by stem; the local copy nests splits in subfolders."""
    index: dict[str, Path] = {}
    for path in (images_root / "100k").rglob("*.jpg"):
        index.setdefault(path.stem, path)
    return index


def load_bdd(
    image_index: dict[str, Path],
    det20_zip: Path,
    attributes: dict[str, tuple[str, str]],
    conditions: set[str],
) -> list[Sample]:
    """Parse Detection-2020 per-image labels for the requested conditions."""
    samples: list[Sample] = []
    with zipfile.ZipFile(det20_zip) as archive:
        for entry in archive.namelist():
            if not entry.endswith(".json"):
                continue
            parts = entry.split("/")
            if len(parts) < 4 or parts[-3] != "100k":
                continue
            split = parts[-2]
            if split not in {"train", "val"}:
                continue
            stem = parts[-1][:-5]
            frame_attributes = attributes.get(stem)
            if frame_attributes is None:
                continue
            condition = bdd_condition(*frame_attributes)
            if condition not in conditions:
                continue
            image_path = image_index.get(stem)
            if image_path is None:
                continue
            payload = json.loads(archive.read(entry))
            frames = payload.get("frames") or []
            if not frames:
                continue
            width, height = image_size(image_path)
            boxes: list[Box] = []
            for obj in frames[0].get("objects", []):
                class_index = BDD_MAP.get(obj.get("category", ""))
                if class_index is None:
                    continue
                box2d = obj.get("box2d")
                if not box2d:
                    continue
                box = Box(
                    class_index,
                    float(box2d["x1"]),
                    float(box2d["y1"]),
                    float(box2d["x2"]),
                    float(box2d["y2"]),
                ).clipped(width, height)
                if box is not None:
                    boxes.append(box)
            if not boxes:
                continue
            samples.append(
                Sample(
                    source="bdd",
                    source_split=split,
                    condition=condition,
                    image_path=image_path,
                    width=width,
                    height=height,
                    boxes=boxes,
                    image_id=stem,
                    weather=frame_attributes[0],
                    timeofday=frame_attributes[1],
                    sequence_id=stem,
                )
            )
    return samples
