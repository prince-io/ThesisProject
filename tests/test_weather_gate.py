from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from ultralytics import YOLO

from litevit.models import FeatureWithGate, WeatherGate, build_gate_only
from litevit.training.trainer import GateDetectionTrainer


def test_weather_gate_is_identity_at_init() -> None:
    for channels in (64, 128, 256):
        gate = WeatherGate(channels).eval()
        x = torch.randn(2, channels, 8, 8)
        with torch.no_grad():
            y = gate(x)
        assert torch.allclose(x, y, atol=1e-6)


def test_weather_gate_parameter_budget() -> None:
    gates = [WeatherGate(channels) for channels in (64, 128, 256)]
    params = sum(p.numel() for gate in gates for p in gate.parameters())
    assert params == 43_568


def test_feature_with_gate_preserves_metadata() -> None:
    base = nn.Conv2d(4, 4, kernel_size=1)
    base.i = 15
    base.f = -1
    base.type = "Dummy"
    wrapper = FeatureWithGate(base, WeatherGate(4))
    x = torch.randn(1, 4, 5, 5)
    with torch.no_grad():
        y = wrapper(x)
    assert y.shape == x.shape
    assert wrapper.i == 15
    assert wrapper.f == -1
    assert wrapper.type == "Dummy"


def test_gate_trainer_builds_wrapped_five_class_model() -> None:
    weights = Path("yolov8n.pt")
    if not weights.exists():
        pytest.skip("yolov8n.pt not available")
    trainer = object.__new__(GateDetectionTrainer)
    trainer.args = SimpleNamespace(model=str(weights), pretrained=True, cls_remap=True)
    trainer.data = {
        "nc": 5,
        "names": {0: "person", 1: "car", 2: "bus", 3: "bicycle", 4: "motorcycle"},
        "channels": 3,
    }
    base = YOLO(str(weights)).model
    model = trainer.get_model(cfg=base.yaml, weights=base, verbose=False)
    for index in (15, 18, 21):
        assert isinstance(model.model[index], FeatureWithGate)
    assert model.model[-1].nc == 5
    assert 3_150_000 < sum(p.numel() for p in model.parameters()) < 3_250_000


def test_build_gate_only_wraps_neck_layers() -> None:
    weights = Path("yolov8n.pt")
    if not weights.exists():
        pytest.skip("yolov8n.pt not available")
    model = build_gate_only(weights).model
    for index in (15, 18, 21):
        assert isinstance(model.model[index], FeatureWithGate)


def test_weather_gate_supports_half_inference() -> None:
    gate = WeatherGate(16).eval().half()
    x = torch.randn(1, 16, 8, 8, dtype=torch.float16)
    with torch.no_grad():
        y = gate(x)
    assert y.dtype == torch.float16
    assert torch.isfinite(y).all()
