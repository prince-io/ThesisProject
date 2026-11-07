"""DAWN (Detection in Adverse Weather Nature) loader."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from litevit.datasets.common import DAWN_MAP, Box, Sample, sha1_file


def _text(element: ET.Element | None, tag: str) -> str:
    if element is None:
        return ""
    return (element.findtext(tag) or "").strip()


def load_dawn(root: Path) -> list[Sample]:
    """Parse DAWN Pascal VOC annotations for fog, rain, and snow."""
    samples: list[Sample] = []
    seen_hashes: set[str] = set()
    for folder in ("Fog", "Rain", "Snow"):
        condition = folder.lower()
        annotations_dir = root / folder / f"{folder}_PASCAL_VOC"
        if not annotations_dir.exists():
            continue
        for xml_path in sorted(annotations_dir.glob("*.xml")):
            image_path = root / folder / f"{xml_path.stem}.jpg"
            if not image_path.exists():
                continue
            digest = sha1_file(image_path)
            if digest in seen_hashes:
                continue
            root_element = ET.parse(xml_path).getroot()
            size = root_element.find("size")
            width = int(float(_text(size, "width") or 0))
            height = int(float(_text(size, "height") or 0))
            if width <= 0 or height <= 0:
                continue
            boxes: list[Box] = []
            for obj in root_element.findall("object"):
                class_index = DAWN_MAP.get(_text(obj, "name").lower())
                if class_index is None:
                    continue
                bndbox = obj.find("bndbox")
                if bndbox is None:
                    continue
                box = Box(
                    class_index,
                    float(_text(bndbox, "xmin")),
                    float(_text(bndbox, "ymin")),
                    float(_text(bndbox, "xmax")),
                    float(_text(bndbox, "ymax")),
                ).clipped(width, height)
                if box is not None:
                    boxes.append(box)
            if not boxes:
                continue
            seen_hashes.add(digest)
            samples.append(
                Sample(
                    source="dawn",
                    source_split="unsplit",
                    condition=condition,
                    image_path=image_path,
                    width=width,
                    height=height,
                    boxes=boxes,
                    image_id=xml_path.stem,
                )
            )
    return samples
