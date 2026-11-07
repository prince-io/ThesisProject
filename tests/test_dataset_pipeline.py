from litevit.datasets.build import SOURCE_TARGETS
from litevit.datasets.common import (
    ACDC_MAP,
    BDD_MAP,
    CLASS_NAMES,
    DAWN_MAP,
    RTTS_MAP,
    Box,
    bdd_condition,
)


def test_class_maps_merge_truck_and_rider() -> None:
    for mapping in (ACDC_MAP, DAWN_MAP, BDD_MAP):
        assert mapping["truck"] == mapping["car"]
        assert "train" not in mapping
    assert ACDC_MAP["rider"] == ACDC_MAP["person"]
    assert BDD_MAP["rider"] == BDD_MAP["pedestrian"]


def test_rtts_five_classes() -> None:
    assert set(RTTS_MAP.values()) == set(range(len(CLASS_NAMES)))


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
