"""Build per-condition and per-source evaluation subsets from the manifest."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from litevit.datasets.common import CLASS_NAMES


def _yaml_payload(root: Path, list_path: Path) -> str:
    names = "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_NAMES))
    return (
        f"path: {root}\n"
        "train: images/train\n"
        f"val: {list_path}\n"
        f"test: {list_path}\n"
        "names:\n"
        f"{names}"
    )


def build_subsets(manifest_path: Path, out_dir: Path) -> dict:
    manifest_path = manifest_path.resolve()
    dataset_root = manifest_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        split = row["split"]
        groups[(split, "all")].append(row)
        groups[(split, f"cond_{row['condition']}")].append(row)
        groups[(split, f"source_{row['source']}")].append(row)

    manifest: dict[str, dict] = {}
    for (split, key), items in sorted(groups.items()):
        list_path = (out_dir / f"{split}_{key}.txt").resolve()
        with list_path.open("w", encoding="utf-8") as handle:
            for row in items:
                image_path = (dataset_root / "images" / split / row["output_name"]).resolve()
                handle.write(f"{image_path}\n")
        yaml_path = (out_dir / f"{split}_{key}.yaml").resolve()
        yaml_path.write_text(_yaml_payload(dataset_root, list_path), encoding="utf-8")
        manifest[f"{split}/{key}"] = {
            "count": len(items),
            "list": str(list_path),
            "yaml": str(yaml_path),
        }

    (out_dir / "subsets.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evaluation subsets")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/adverse16k/manifest.csv"),
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/processed/adverse16k/subsets")
    )
    args = parser.parse_args()
    manifest = build_subsets(args.manifest, args.out)
    counts = {key: value["count"] for key, value in manifest.items()}
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
