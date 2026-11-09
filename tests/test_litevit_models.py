from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from ultralytics import YOLO

from litevit.models import (
    FeatureWithBlock,
    LayerNorm2d,
    LinearAttention2d,
    LiteViTBlock,
    build_v1_p5,
)
from litevit.training.train import verify_saved_architecture
from litevit.training.trainer import V1DetectionTrainer


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


def test_layer_norm_2d_shape_and_normalization() -> None:
    x = torch.randn(2, 8, 4, 5)
    y = LayerNorm2d(8)(x)
    assert y.shape == x.shape
    assert torch.allclose(y.mean(dim=1), torch.zeros_like(y.mean(dim=1)), atol=1e-5)


def test_linear_attention_shape_and_finite() -> None:
    x = torch.randn(2, 3 * 16, 5, 7)
    y = LinearAttention2d(16, heads=4)(x)
    assert y.shape == (2, 16, 5, 7)
    assert torch.isfinite(y).all()


def test_litevit_block_is_identity_at_init() -> None:
    block = LiteViTBlock(32, heads=4).eval()
    x = torch.randn(2, 32, 8, 9)
    with torch.no_grad():
        y = block(x)
    assert torch.allclose(x, y, atol=1e-6)


def test_litevit_block_fp16_forward_stays_finite() -> None:
    block = LiteViTBlock(32, heads=4).eval()
    x = torch.randn(2, 32, 8, 9, dtype=torch.float16)
    with torch.no_grad():
        y = block(x)
    assert y.dtype == torch.float16
    assert torch.isfinite(y).all()
    assert torch.allclose(x.float(), y.float(), atol=1e-3)


def test_litevit_block_parameter_budget() -> None:
    block = LiteViTBlock(256, heads=4)
    params = sum(p.numel() for p in block.parameters())
    assert 400_000 < params < 700_000


def test_feature_wrapper_preserves_metadata() -> None:
    base = nn.Conv2d(4, 4, kernel_size=1)
    base.i = 3
    base.f = -1
    base.type = "Dummy"
    wrapper = FeatureWithBlock(base, LiteViTBlock(4, heads=2))
    x = torch.randn(1, 4, 5, 5)
    with torch.no_grad():
        y = wrapper(x)
    assert y.shape == x.shape
    assert wrapper.i == 3
    assert wrapper.f == -1
    assert wrapper.type == "Dummy"


def test_build_v1_p5_starts_as_baseline() -> None:
    weights = Path("yolov8n.pt")
    if not weights.exists():
        pytest.skip("yolov8n.pt not available")
    baseline = YOLO(str(weights)).model.eval()
    wrapped = build_v1_p5(weights).model.eval()
    x = torch.randn(1, 3, 160, 160)
    with torch.no_grad():
        y_base = baseline(x)
        y_wrapped = wrapped(x)
    _assert_same_output(y_base, y_wrapped)


def test_v1_trainer_builds_wrapped_five_class_model() -> None:
    weights = Path("yolov8n.pt")
    if not weights.exists():
        pytest.skip("yolov8n.pt not available")
    trainer = object.__new__(V1DetectionTrainer)
    trainer.args = SimpleNamespace(model=str(weights), pretrained=True, cls_remap=True)
    trainer.data = {
        "nc": 5,
        "names": {0: "person", 1: "car", 2: "bus", 3: "bicycle", 4: "motorcycle"},
        "channels": 3,
    }
    base = YOLO(str(weights)).model
    model = trainer.get_model(cfg=base.yaml, weights=base, verbose=False)
    assert isinstance(model.model[9], FeatureWithBlock)
    assert model.model[-1].nc == 5
    assert 3_500_000 < sum(p.numel() for p in model.parameters()) < 3_800_000


def test_architecture_verification_rejects_baseline(tmp_path: Path) -> None:
    weights = Path("yolov8n.pt")
    if not weights.exists():
        pytest.skip("yolov8n.pt not available")
    graph_dir = tmp_path / "weights"
    graph_dir.mkdir()
    YOLO(str(weights)).save(str(graph_dir / "best.pt"))
    with pytest.raises(RuntimeError):
        verify_saved_architecture(tmp_path, "v1_p5")
