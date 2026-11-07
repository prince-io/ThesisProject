from pathlib import Path

from PIL import Image, ImageDraw

from litevit.datasets.build import SOURCE_TARGETS, _assign_sequence_splits
from litevit.datasets.common import (
    ACDC_MAP,
    BDD_MAP,
    CLASS_NAMES,
    DAWN_MAP,
    RTTS_MAP,
    Box,
    Sample,
    bdd_condition,
    dhash_image,
)


def test_class_maps_merge_truck_and_rider() -> None:
    for mapping in (ACDC_MAP, DAWN_MAP, BDD_MAP):
        assert mapping["truck"] == mapping["car"]
        assert "train" not in mapping
    assert ACDC_MAP["rider"] == ACDC_MAP["person"]
    assert BDD_MAP["rider"] == BDD_MAP["pedestrian"]


def test_bdd_legacy_aliases() -> None:
    assert BDD_MAP["person"] == BDD_MAP["pedestrian"]
    assert BDD_MAP["bike"] == BDD_MAP["bicycle"]
    assert BDD_MAP["motor"] == BDD_MAP["motorcycle"]


def test_rtts_five_classes() -> None:
    assert set(RTTS_MAP.values()) == set(range(len(CLASS_NAMES)))


def test_sequence_assignment_keeps_sequences_whole() -> None:
    samples = [
        Sample(
            source="acdc",
            source_split="train",
            condition="fog",
            image_path=Path("x.jpg"),
            width=10,
            height=10,
            boxes=[],
            image_id=f"img{index}",
            sequence_id=f"seq{index % 3}",
        )
        for index in range(12)
    ]
    mapping = _assign_sequence_splits(samples, 42)
    assert set(mapping.values()) <= {"train", "val", "test"}
    assert len(mapping) == 3
    assert mapping == _assign_sequence_splits(samples, 42)


def test_dhash_identical_and_inverted_patterns(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    inverted = tmp_path / "inverted.png"
    for path, flip in ((first, False), (second, False), (inverted, True)):
        image = Image.new("L", (32, 24), 0)
        draw = ImageDraw.Draw(image)
        for start in range(0, 32, 4):
            if bool((start // 4) % 2) ^ flip:
                draw.rectangle([start, 0, start + 3, 23], fill=255)
        image.save(path)
    assert dhash_image(first) == dhash_image(second)
    assert dhash_image(first) != dhash_image(inverted)


def test_bdd_condition_is_exclusive() -> None:
    assert bdd_condition("foggy", "night") == "fog"
    assert bdd_condition("snowy", "daytime") == "snow"
    assert bdd_condition("rainy", "night") == "rain"
    assert bdd_condition("clear", "night") == "night"
    assert bdd_condition("overcast", "night") == "night"
    assert bdd_condition("clear", "daytime") is None
    assert bdd_condition("undefined", "dawn/dusk") is None


def test_box_clip_and_yolo() -> None:
    box = Box(1, -10, -5, 50, 40).clipped(100, 100)
    assert box is not None
    assert (box.x1, box.y1, box.x2, box.y2) == (0.0, 0.0, 50.0, 40.0)
    x_center, y_center, width, height = box.to_yolo(100, 100)
    assert abs(x_center - 0.25) < 1e-9
    assert abs(y_center - 0.2) < 1e-9
    assert abs(width - 0.5) < 1e-9
    assert abs(height - 0.4) < 1e-9


def test_degenerate_box_is_dropped() -> None:
    assert Box(1, 10, 10, 10, 30).clipped(100, 100) is None
    assert Box(1, 10, 10, 30, 10).clipped(100, 100) is None


def test_source_targets_sum_to_balanced_split() -> None:
    totals = {"train": 0, "val": 0, "test": 0}
    for condition in ("fog", "rain", "snow", "night"):
        train = val = test = 0
        for (group_condition, _source), (n_train, n_val, n_test) in SOURCE_TARGETS.items():
            if group_condition != condition:
                continue
            train += n_train
            val += n_val
            test += n_test
        assert (train, val, test) == (3200, 400, 400)
        totals["train"] += train
        totals["val"] += val
        totals["test"] += test
    assert totals == {"train": 12800, "val": 1600, "test": 1600}
