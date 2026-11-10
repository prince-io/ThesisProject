from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from ultralytics import YOLO

from litevit.models import (
    EdgeEnhance,
    FeatureWithEdge,
    FeatureWithGate,
    GateWithSpatial,
    SpatialAttention,
    build_gate_edge_cbam,
)
from litevit.training.trainer import GateEdgeCBAMTrainer


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


def test_edge_enhance_is_identity_at_init() -> None:
    for channels in (64, 128):
        edge = EdgeEnhance(channels).eval()
        x = torch.randn(2, channels, 8, 8)
        with torch.no_grad():
            y = edge(x)
        assert y.shape == x.shape
        assert torch.allclose(x, y, atol=1e-6)


def test_spatial_attention_is_identity_at_init() -> None:
    spatial = SpatialAttention().eval()
    x = torch.randn(2, 16, 8, 8)
    with torch.no_grad():
        y = spatial(x)
    assert torch.allclose(x, y, atol=1e-6)


def test_gate_with_spatial_is_identity_at_init() -> None:
    block = GateWithSpatial(16).eval()
    x = torch.randn(2, 16, 8, 8)
    with torch.no_grad():
        y = block(x)
    assert torch.allclose(x, y, atol=1e-6)


def test_feature_with_edge_preserves_metadata() -> None:
    base = nn.Conv2d(4, 4, kernel_size=1)
    base.i = 4
    base.f = -1
    base.type = "Dummy"
    wrapper = FeatureWithEdge(base, EdgeEnhance(4))
    x = torch.randn(1, 4, 5, 5)
    with torch.no_grad():
        y = wrapper(x)
    assert y.shape == x.shape
    assert wrapper.i == 4
    assert wrapper.f == -1
    assert wrapper.type == "Dummy"


def test_gate_edge_cbam_trainer_builds_model() -> None:
    weights = Path("yolov8n.pt")
    if not weights.exists():
        pytest.skip("yolov8n.pt not available")
    trainer = object.__new__(GateEdgeCBAMTrainer)
    trainer.args = SimpleNamespace(model=str(weights), pretrained=True, cls_remap=True)
    trainer.data = {
        "nc": 5,
        "names": {0: "person", 1: "car", 2: "bus", 3: "bicycle", 4: "motorcycle"},
        "channels": 3,
    }
    base = YOLO(str(weights)).model
    model = trainer.get_model(cfg=base.yaml, weights=base, verbose=False)
    for index in (4, 6):
        assert isinstance(model.model[index], FeatureWithEdge)
    for index in (15, 18, 21):
        assert isinstance(model.model[index], FeatureWithGate)
    assert model.model[-1].nc == 5
    assert sum(p.numel() for p in model.parameters()) == 3_203_704


def test_build_gate_edge_cbam_starts_as_baseline() -> None:
    weights = Path("yolov8n.pt")
    if not weights.exists():
        pytest.skip("yolov8n.pt not available")
    baseline = YOLO(str(weights)).model.eval()
    combined = build_gate_edge_cbam(weights).model.eval()
    x = torch.randn(1, 3, 160, 160)
    with torch.no_grad():
        y_base = baseline(x)
        y_combined = combined(x)
    _assert_same_output(y_base, y_combined)
