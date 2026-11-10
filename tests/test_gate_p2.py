from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from ultralytics import YOLO

from litevit.models import add_p2_head, build_gate_only, build_gate_p2
from litevit.models.gates import FeatureWithGate
from litevit.training.train import verify_saved_architecture
from litevit.training.trainer import GateP2DetectionTrainer


def _make_trainer_model():
    weights = Path("yolov8n.pt")
    if not weights.exists():
        pytest.skip("yolov8n.pt not available")
    trainer = object.__new__(GateP2DetectionTrainer)
    trainer.args = SimpleNamespace(model=str(weights), pretrained=True, cls_remap=True)
    trainer.data = {
        "nc": 5,
        "names": {0: "person", 1: "car", 2: "bus", 3: "bicycle", 4: "motorcycle"},
        "channels": 3,
    }
    base = YOLO(str(weights)).model
    return trainer.get_model(cfg=base.yaml, weights=base, verbose=False)


def test_gate_p2_trainer_builds_four_level_model() -> None:
    model = _make_trainer_model()
    assert len(model.model) == 26
    detect = model.model[-1]
    assert detect.nl == 4
    assert detect.stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    for index in (15, 18, 21):
        assert isinstance(model.model[index], FeatureWithGate)
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_270_388


def test_gate_p2_preserves_p3_p4_p5_branches() -> None:
    weights = Path("yolov8n.pt")
    if not weights.exists():
        pytest.skip("yolov8n.pt not available")
    base = build_gate_only(weights).model
    p2 = build_gate_p2(weights).model
    base_detect = base.model[-1]
    p2_detect = p2.model[-1]
    for index in range(3):
        base_state = base_detect.cv2[index].state_dict()
        p2_state = p2_detect.cv2[index + 1].state_dict()
        assert base_state.keys() == p2_state.keys()
        assert all(torch.equal(base_state[key], p2_state[key]) for key in base_state)
        base_state = base_detect.cv3[index].state_dict()
        p2_state = p2_detect.cv3[index + 1].state_dict()
        assert base_state.keys() == p2_state.keys()
        assert all(torch.equal(base_state[key], p2_state[key]) for key in base_state)


def test_gate_p2_forward_shapes() -> None:
    model = _make_trainer_model()
    x = torch.zeros(1, 3, 160, 160)
    model.eval()
    with torch.no_grad():
        output = model(x)
    assert output[0].shape == (1, 9, 2_125)

    model.train()
    with torch.no_grad():
        output = model(x)
    assert output["boxes"].shape == (1, 64, 2_125)
    assert output["scores"].shape == (1, 5, 2_125)


def test_gate_p2_head_is_idempotent() -> None:
    weights = Path("yolov8n.pt")
    if not weights.exists():
        pytest.skip("yolov8n.pt not available")
    model = YOLO(str(weights)).model
    add_p2_head(model)
    add_p2_head(model)
    assert len(model.model) == 26
    assert model.model[-1].nl == 4


def test_gate_p2_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = _make_trainer_model()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    torch.save({"model": model, "train_args": {}}, weights_dir / "best.pt")

    assert verify_saved_architecture(tmp_path, "gate_p2") is True
    loaded = YOLO(str(weights_dir / "best.pt")).model
    assert loaded.model[-1].nl == 4
    assert loaded.model[-1].stride.tolist() == [4.0, 8.0, 16.0, 32.0]
    loaded.eval()
    with torch.no_grad():
        output = loaded(torch.zeros(1, 3, 160, 160))
    assert output[0].shape == (1, 9, 2_125)
