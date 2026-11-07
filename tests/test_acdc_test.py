import json
from pathlib import Path

from PIL import Image

from litevit.datasets.acdc_test import prepare_acdc_test


def _make_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), (10, 20, 30)).save(path)


def test_prepare_acdc_test_copies_and_manifests(tmp_path: Path) -> None:
    root = tmp_path / "acdc"
    for condition in ("fog", "night", "rain", "snow"):
        file_name = f"{condition}/test/seq1/frame1_rgb_anon.png"
        _make_image(root / "images" / file_name)
        info = {
            "images": [{"file_name": file_name, "height": 24, "width": 32, "id": 0}],
            "categories": [{"id": 1, "name": "car"}],
        }
        info_path = (
            root / "labels" / condition / f"instancesonly_{condition}_test_image_info.json"
        )
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps(info), encoding="utf-8")

    out = tmp_path / "out"
    stats = prepare_acdc_test(root, out)
    assert stats["total_images"] == 4
    assert stats["by_condition"] == {"fog": 1, "night": 1, "rain": 1, "snow": 1}
    assert len(list((out / "images").rglob("*.png"))) == 4
    assert len(list((out / "labels_info").glob("*.json"))) == 4
    manifest_lines = (out / "manifest.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(manifest_lines) == 5
