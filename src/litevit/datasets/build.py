"""Build the balanced 16k adverse-weather detection dataset."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from litevit.datasets.acdc import load_acdc
from litevit.datasets.bdd import build_image_index, load_bdd, load_bdd_attributes
from litevit.datasets.common import (
    CLASS_NAMES,
    CONDITIONS,
    NUM_CLASSES,
    Sample,
    image_size,
    link_or_copy,
    safe_token,
    write_coco,
    write_yolo_label,
)
from litevit.datasets.dawn import load_dawn
from litevit.datasets.rtts import load_rtts

SPLITS = ("train", "val", "test")

# (condition, source) -> (train, val, test); each condition sums to 3200/400/400.
SOURCE_TARGETS: dict[tuple[str, str], tuple[int, int, int]] = {
    ("fog", "rtts"): (2560, 320, 320),
    ("fog", "acdc"): (400, 50, 50),
    ("fog", "dawn"): (240, 30, 30),
    ("rain", "bdd"): (2640, 330, 330),
    ("rain", "acdc"): (400, 50, 50),
    ("rain", "dawn"): (160, 20, 20),
    ("snow", "bdd"): (2637, 330, 329),
    ("snow", "acdc"): (400, 50, 50),
    ("snow", "dawn"): (163, 20, 21),
    ("night", "bdd"): (2800, 347, 347),
    ("night", "acdc"): (400, 53, 53),
}

CONDITION_TARGETS = {"train": 3200, "val": 400, "test": 400}
FILLERS = {"fog": "rtts", "rain": "bdd", "snow": "bdd", "night": "bdd"}


def _rng(condition: str, source: str, seed: int) -> np.random.Generator:
    token = zlib.crc32(f"{condition}:{source}".encode()) % 10000
    return np.random.default_rng(seed + token)


def _take_up_to(items: list[Sample], count: int, rng: np.random.Generator) -> list[Sample]:
    count = min(count, len(items))
    if count <= 0:
        return []
    indices = rng.choice(len(items), size=count, replace=False)
    return [items[int(index)] for index in indices]


def _assign_sequence_splits(samples: list[Sample], seed: int) -> dict[str, str]:
    """Assign whole sequences to train/val/test to prevent frame leakage."""
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        groups[sample.sequence_id or sample.image_id].append(sample)
    if not groups:
        return {}
    total = sum(len(items) for items in groups.values())
    targets = {"train": 0.8 * total, "val": 0.1 * total, "test": 0.1 * total}
    assigned = {split: 0 for split in SPLITS}
    rng = np.random.default_rng(seed + zlib.crc32(b"sequence") % 10000)
    order = list(groups)
    rng.shuffle(order)
    order.sort(key=lambda sequence: len(groups[sequence]), reverse=True)
    split_map: dict[str, str] = {}
    for sequence in order:
        split = max(
            SPLITS,
            key=lambda candidate: (targets[candidate] - assigned[candidate])
            / max(targets[candidate], 1.0),
        )
        split_map[sequence] = split
        assigned[split] += len(groups[sequence])
    return split_map


def _allocate_source(
    condition: str,
    source: str,
    samples: list[Sample],
    targets: tuple[int, int, int],
    seed: int,
    allow_short: bool,
) -> list[tuple[Sample, str]]:
    n_train, n_val, n_test = targets
    rng = _rng(condition, source, seed)
    if source in {"acdc", "rtts", "dawn"}:
        train = _take_up_to(
            [sample for sample in samples if sample.split_hint == "train"], n_train, rng
        )
        val = _take_up_to(
            [sample for sample in samples if sample.split_hint == "val"], n_val, rng
        )
        test = _take_up_to(
            [sample for sample in samples if sample.split_hint == "test"], n_test, rng
        )
    else:
        train_pool = [sample for sample in samples if sample.source_split == "train"]
        holdout_pool = [
            sample for sample in samples if sample.source_split in {"val", "test"}
        ]
        train = _take_up_to(train_pool, n_train, rng)
        val = _take_up_to(holdout_pool, n_val, rng)
        remaining = [
            sample for sample in holdout_pool if id(sample) not in {id(item) for item in val}
        ]
        test = _take_up_to(remaining, n_test, rng)

    selected = (
        [(sample, "train") for sample in train]
        + [(sample, "val") for sample in val]
        + [(sample, "test") for sample in test]
    )
    if not allow_short:
        actual = {"train": len(train), "val": len(val), "test": len(test)}
        for split, wanted in (("train", n_train), ("val", n_val), ("test", n_test)):
            if actual[split] < wanted:
                raise RuntimeError(
                    f"{condition}/{source}: {split} needs {wanted}, only {actual[split]} available"
                )
    return selected


def _prepare_samples(selected: list[tuple[Sample, str]]) -> None:
    used_names: set[str] = set()
    for sample, _ in selected:
        if sample.width <= 0 or sample.height <= 0:
            sample.width, sample.height = image_size(sample.image_path)
        boxes = []
        for box in sample.boxes:
            clipped = box.clipped(sample.width, sample.height)
            if clipped is not None:
                boxes.append(clipped)
        if not boxes:
            raise RuntimeError(f"no valid boxes remain for {sample.image_path}")
        sample.boxes = boxes
        stem = safe_token(f"{sample.source}_{sample.condition}_{sample.image_id}")
        suffix = sample.image_path.suffix.lower()
        output_name = f"{stem}{suffix}"
        counter = 1
        while output_name in used_names:
            counter += 1
            output_name = f"{stem}_{counter}{suffix}"
        used_names.add(output_name)
        sample.output_name = output_name


def _counts(selected: list[tuple[Sample, str]]) -> dict:
    by_split: Counter = Counter()
    by_condition: Counter = Counter()
    by_source: Counter = Counter()
    by_split_condition: Counter = Counter()
    by_split_source: Counter = Counter()
    classes_by_split: dict[str, Counter] = {split: Counter() for split in SPLITS}
    classes_by_condition: dict[str, Counter] = {
        condition: Counter() for condition in CONDITIONS
    }
    for sample, split in selected:
        by_split[split] += 1
        by_condition[sample.condition] += 1
        by_source[sample.source] += 1
        by_split_condition[f"{split}/{sample.condition}"] += 1
        by_split_source[f"{split}/{sample.source}"] += 1
        for box in sample.boxes:
            classes_by_split[split][CLASS_NAMES[box.cls]] += 1
            classes_by_condition[sample.condition][CLASS_NAMES[box.cls]] += 1
    return {
        "total_images": len(selected),
        "by_split": dict(by_split),
        "by_condition": dict(by_condition),
        "by_source": dict(by_source),
        "by_split_condition": dict(by_split_condition),
        "by_split_source": dict(by_split_source),
        "instances_by_split": {k: dict(v) for k, v in classes_by_split.items()},
        "instances_by_condition": {k: dict(v) for k, v in classes_by_condition.items()},
    }


def build(
    root: Path,
    out_dir: Path,
    seed: int = 42,
    force: bool = False,
    copy_images: bool = False,
) -> dict:
    started = time.time()
    if out_dir.exists():
        if not force:
            raise FileExistsError(f"{out_dir} exists; pass --force to rebuild")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dawn_root = root / "ref" / "dawn"
    acdc_root = root / "ref" / "acdc"
    bdd_root = root / "ref" / "bdd100k"
    bdd_images = bdd_root / "bdd100k" / "bdd100k" / "images"
    bdd_2018_labels = bdd_root / "bdd100k_labels_release" / "bdd100k" / "labels"
    bdd_2020_zip = bdd_root / "bdd100k_det_20_labels_trainval.zip"
    rtts_root = root / "ref" / "rtts" / "RTTS"

    required = [
        dawn_root,
        acdc_root,
        bdd_images,
        bdd_2018_labels,
        bdd_2020_zip,
        rtts_root,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing raw data:\n  " + "\n  ".join(missing))

    print("[1/5] loading DAWN", flush=True)
    dawn = load_dawn(dawn_root)
    print(f"      dawn samples: {len(dawn)}", flush=True)

    print("[2/5] loading ACDC", flush=True)
    acdc = load_acdc(acdc_root)
    print(f"      acdc samples: {len(acdc)}", flush=True)

    print("[3/5] loading RTTS", flush=True)
    rtts = load_rtts(rtts_root)
    print(f"      rtts samples: {len(rtts)}", flush=True)

    print("[4/5] loading BDD100K (streaming attributes + 2020 labels)", flush=True)
    attributes = load_bdd_attributes(bdd_2018_labels)
    print(f"      BDD frame attributes: {len(attributes)}", flush=True)
    image_index = build_image_index(bdd_images)
    print(f"      BDD image index: {len(image_index)}", flush=True)
    bdd = load_bdd(image_index, bdd_2020_zip, attributes, {"rain", "snow", "night"})
    print(f"      bdd samples (rain/snow/night): {len(bdd)}", flush=True)

    all_samples = dawn + acdc + rtts + bdd
    for source in ("acdc", "rtts", "dawn"):
        source_samples = [sample for sample in all_samples if sample.source == source]
        split_map = _assign_sequence_splits(source_samples, seed)
        for sample in source_samples:
            sample.split_hint = split_map[sample.sequence_id or sample.image_id]
    groups: dict[tuple[str, str], list[Sample]] = {}
    for sample in all_samples:
        groups.setdefault((sample.condition, sample.source), []).append(sample)

    print("[5/5] allocating 16k balanced split", flush=True)
    selected: list[tuple[Sample, str]] = []
    for condition in CONDITIONS:
        filler = FILLERS[condition]
        sources = [
            source for (group_condition, source) in SOURCE_TARGETS if group_condition == condition
        ]
        ordered = [source for source in sources if source != filler] + [filler]
        actual = {split: 0 for split in SPLITS}
        for source in ordered:
            group = groups.get((condition, source), [])
            if source == filler:
                desired = tuple(
                    CONDITION_TARGETS[split] - actual[split] for split in SPLITS
                )
            elif source in {"acdc", "dawn"}:
                desired = tuple(
                    sum(1 for sample in group if sample.split_hint == split)
                    for split in SPLITS
                )
            else:
                desired = SOURCE_TARGETS.get((condition, source), (0, 0, 0))
            pools = Counter(
                sample.split_hint
                if source in {"acdc", "rtts", "dawn"}
                else sample.source_split
                for sample in group
            )
            print(
                f"      {condition}/{source}: pools {dict(pools)} targets {desired}",
                flush=True,
            )
            allocated = _allocate_source(
                condition,
                source,
                group,
                desired,
                seed,
                allow_short=source != filler,
            )
            selected.extend(allocated)
            for _, split in allocated:
                actual[split] += 1
        if actual != CONDITION_TARGETS:
            raise RuntimeError(f"{condition}: allocated {actual} != {CONDITION_TARGETS}")
    print(f"      selected total: {len(selected)}", flush=True)

    _prepare_samples(selected)
    stats = _counts(selected)

    for split in SPLITS:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "coco" / split).parent.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "output_name",
                "source",
                "source_split",
                "condition",
                "split",
                "image_id",
                "sequence",
                "weather",
                "timeofday",
                "width",
                "height",
                "num_boxes",
                "class_counts",
                "original_path",
            ]
        )
        for sample, split in selected:
            destination = out_dir / "images" / split / sample.output_name
            link_or_copy(sample.image_path, destination, copy_only=copy_images)
            label_path = out_dir / "labels" / split / f"{Path(sample.output_name).stem}.txt"
            write_yolo_label(label_path, sample.boxes, sample.width, sample.height)
            writer.writerow(
                [
                    sample.output_name,
                    sample.source,
                    sample.source_split,
                    sample.condition,
                    split,
                    sample.image_id,
                    sample.sequence_id,
                    sample.weather,
                    sample.timeofday,
                    sample.width,
                    sample.height,
                    sample.num_boxes,
                    json.dumps(
                        {CLASS_NAMES[k]: v for k, v in sorted(sample.class_counts().items())}
                    ),
                    str(sample.image_path),
                ]
            )

    for split in SPLITS:
        split_samples = [sample for sample, sample_split in selected if sample_split == split]
        write_coco(split_samples, out_dir / "coco" / f"{split}.json")

    stats["seed"] = seed
    stats["copy_images"] = copy_images
    stats["class_names"] = list(CLASS_NAMES)
    stats["num_classes"] = NUM_CLASSES
    stats["elapsed_seconds"] = round(time.time() - started, 2)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    dataset_yaml = (
        f"path: {out_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n"
        + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_NAMES))
    )
    (out_dir / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")

    print(json.dumps(stats, indent=2), flush=True)
    print(f"dataset written to {out_dir}", flush=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the adverse16k dataset")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("data/processed/adverse16k"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="copy images instead of hardlinking (self-contained dataset)",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = args.out if args.out.is_absolute() else root / args.out
    build(root, out_dir, seed=args.seed, force=args.force, copy_images=args.copy_images)


if __name__ == "__main__":
    main()
