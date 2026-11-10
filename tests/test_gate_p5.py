from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ultralytics import YOLO

from litevit.models import FeatureWithBlock, FeatureWithGate, build_gate_p5
from litevit.training.trainer import GateP5DetectionTrainer


def _assert_same_output(left, right, atol: float = 1e-5) -> None:
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_same_output(left[key], right[key], atol=atol)
        return
    if isinstance(left, (tuple, list)):
        assert isinstance(right, (tuple, list))
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_same_output(a, b, atol=atol)
        return
    assert isinstance(left, torch.Tensor)
    assert isinstance(right, torch.Tensor)
    assert left.shape == right.shape
    assert torch.allclose(left, right, atol=atol)


def test_gate_p5_trainer_builds_combined_model() -> None:
    weights = Path("yolov8n.pt")
    if not weights.exists():
        pytest.skip("yolov8n.pt not available")
    trainer = object.__new__(GateP5DetectionTrainer)
    trainer.args = SimpleNamespace(model=str(weights), pretrained=True, cls_remap=True)
    trainer.data = {
        "nc": 5,
        "names": {0: "person", 1: "car", 2: "bus", 3: "bicycle", 4: "motorcycle"},
        "channels": 3,
    }
    base = YOLO(str(weights)).model
    model = trainer.get_model(cfg=base.yaml, weights=base, verbose=False)
    assert isinstance(model.model[9], FeatureWithBlock)
    for index in (15, 18, 21):
        assert isinstance(model.model[index], FeatureWithGate)
    assert model.model[-1].nc == 5
    assert sum(p.numel() for p in model.parameters()) == 3_583_951


def test_build_gate_p5_starts_as_baseline() -> None:
    weights = Path("yolov8n.pt")
    if not weights.exists():
        pytest.skip("yolov8n.pt not available")
    baseline = YOLO(str(weights)).model.eval()
    combined = build_gate_p5(weights).model.eval()
    x = torch.randn(1, 3, 160, 160)
    with torch.no_grad():
        y_base = baseline(x)
        y_combined = combined(x)
    _assert_same_output(y_base, y_combined)
