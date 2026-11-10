import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from litevit.training.config import load_config, resolve_paths
from litevit.training.evaluate import _metric_dict
from litevit.training.subsets import build_subsets


def test_load_config_and_resolve_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "model: yolov8n.pt\n"
        "data: configs/datasets/adverse16k.yaml\n"
        "epochs: 10\n"
        "imgsz: 640\n"
        "batch: 4\n"
        "cls_pw: 0.5\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config["cls_pw"] == 0.5
    resolved = resolve_paths(config, tmp_path)
    assert resolved["data"] == str(
        (tmp_path / "configs/datasets/adverse16k.yaml").resolve()
    )
    assert resolved["project"] == str((tmp_path / "runs/train").resolve())


def test_load_config_rejects_bad_cls_pw(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "model: yolov8n.pt\n"
        "data: data.yaml\n"
        "epochs: 1\n"
        "imgsz: 640\n"
        "batch: 4\n"
        "cls_pw: 2.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(config_path)


def test_load_config_accepts_v1_arch(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "model: yolov8n.pt\n"
        "arch: v1_p5\n"
        "data: data.yaml\n"
        "epochs: 1\n"
        "imgsz: 640\n"
        "batch: 4\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config["arch"] == "v1_p5"


def test_load_config_accepts_gate_arch(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "model: yolov8n.pt\n"
        "arch: gate_only\n"
        "data: data.yaml\n"
        "epochs: 1\n"
        "imgsz: 640\n"
        "batch: 4\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config["arch"] == "gate_only"


def test_load_config_accepts_gate_p5_arch(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "model: yolov8n.pt\n"
        "arch: gate_p5\n"
        "data: data.yaml\n"
        "epochs: 1\n"
        "imgsz: 640\n"
        "batch: 4\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config["arch"] == "gate_p5"


def test_load_config_accepts_gate_p2_arch(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "model: yolov8n.pt\n"
        "arch: gate_p2\n"
        "data: data.yaml\n"
        "epochs: 1\n"
        "imgsz: 640\n"
        "batch: 4\n",
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config["arch"] == "gate_p2"


def test_load_config_rejects_bad_arch(tmp_path: Path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "model: yolov8n.pt\n"
        "arch: unknown\n"
        "data: data.yaml\n"
        "epochs: 1\n"
        "imgsz: 640\n"
        "batch: 4\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(config_path)


def test_build_subsets(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    rows = [
        {"output_name": "a.jpg", "source": "acdc", "condition": "fog", "split": "val"},
        {"output_name": "b.jpg", "source": "acdc", "condition": "fog", "split": "val"},
        {"output_name": "c.jpg", "source": "bdd", "condition": "rain", "split": "val"},
        {"output_name": "d.jpg", "source": "bdd", "condition": "rain", "split": "test"},
    ]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["output_name", "source", "condition", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)
    out_dir = tmp_path / "subsets"
    payload = build_subsets(manifest, out_dir)
    assert payload["val/cond_fog"]["count"] == 2
    assert payload["val/source_bdd"]["count"] == 1
    assert payload["test/cond_rain"]["count"] == 1
    list_path = Path(payload["val/cond_fog"]["list"])
    assert len(list_path.read_text(encoding="utf-8").splitlines()) == 2
    yaml_path = Path(payload["val/cond_fog"]["yaml"])
    assert "names:" in yaml_path.read_text(encoding="utf-8")


def test_metric_dict() -> None:
    box = SimpleNamespace(
        ap_class_index=[0, 1],
        ap50=[0.7, 0.8],
        ap=[0.5, 0.6],
        mp=0.5,
        mr=0.4,
        map50=0.75,
        map=0.55,
    )
    metrics = SimpleNamespace(
        names={0: "person", 1: "car", 2: "bus"},
        box=box,
        nt_per_class=[10, 20, 0],
        speed={"inference": 10.0},
        results_dict={"fitness": 0.55},
    )
    payload = _metric_dict(metrics)
    assert payload["map50"] == 0.75
    assert payload["per_class"]["person"]["ap50"] == 0.7
    assert payload["per_class"]["bus"]["ap50"] is None
    assert payload["targets_per_class"]["car"] == 20
    assert payload["speed_ms"]["inference"] == 10.0
