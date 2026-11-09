"""Build LiteViT variants by wrapping pretrained YOLO detector layers."""

from __future__ import annotations

from pathlib import Path

from torch import nn
from ultralytics import YOLO

from litevit.models.blocks import FeatureWithBlock, LiteViTBlock
from litevit.models.gates import GATE_REDUCTION, GATE_SCALE, FeatureWithGate, WeatherGate

P5_LAYER_INDEX = 9
GATE_LAYER_INDICES = (15, 18, 21)


def _infer_out_channels(module: nn.Module) -> int:
    """Infer the output channel count from a detector layer."""
    cv2 = getattr(module, "cv2", None)
    conv = getattr(cv2, "conv", None)
    if isinstance(conv, nn.Conv2d):
        return conv.out_channels
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Conv2d):
            return child.out_channels
    raise ValueError(f"could not infer output channels for {type(module).__name__}")


def wrap_v1_p5(
    detection: nn.Module,
    heads: int = 4,
    mlp_ratio: float = 2.0,
) -> nn.Module:
    """Wrap the P5/SPPF layer of an existing detection model in place."""
    layers = getattr(detection, "model", None)
    if layers is None:
        raise RuntimeError("detection model has no layer sequence")
    if not 0 <= P5_LAYER_INDEX < len(layers):
        raise IndexError(
            f"layer index {P5_LAYER_INDEX} outside model with {len(layers)} layers"
        )
    base = layers[P5_LAYER_INDEX]
    dim = _infer_out_channels(base)
    block = LiteViTBlock(dim, heads=heads, mlp_ratio=mlp_ratio)
    layers[P5_LAYER_INDEX] = FeatureWithBlock(base, block)
    return detection


def build_v1_p5(
    weights: str | Path = "yolov8n.pt",
    heads: int = 4,
    mlp_ratio: float = 2.0,
) -> YOLO:
    """Build V1: one LiteViT block after the P5/SPPF backbone layer.

    The pretrained YOLOv8n weights are loaded first, so all original backbone,
    neck, and head weights are preserved. The new block is channel-preserving and
    identity-initialized.
    """
    model = YOLO(str(weights))
    if model.model is None:
        raise RuntimeError("YOLO model has no detector attached")
    wrap_v1_p5(model.model, heads=heads, mlp_ratio=mlp_ratio)
    return model


def wrap_gate_only(
    detection: nn.Module,
    reduction: int = GATE_REDUCTION,
    scale: float = GATE_SCALE,
) -> nn.Module:
    """Wrap the P3/P4/P5 neck outputs of an existing detection model with gates."""
    layers = getattr(detection, "model", None)
    if layers is None:
        raise RuntimeError("detection model has no layer sequence")
    for index in GATE_LAYER_INDICES:
        if not 0 <= index < len(layers):
            raise IndexError(
                f"layer index {index} outside model with {len(layers)} layers"
            )
        base = layers[index]
        if isinstance(base, FeatureWithGate):
            continue
        channels = _infer_out_channels(base)
        layers[index] = FeatureWithGate(
            base, WeatherGate(channels, reduction=reduction, scale=scale)
        )
    return detection


def build_gate_only(
    weights: str | Path = "yolov8n.pt",
    reduction: int = GATE_REDUCTION,
    scale: float = GATE_SCALE,
) -> YOLO:
    """Build the gate-only baseline: YOLOv8n plus P3/P4/P5 weather gates."""
    model = YOLO(str(weights))
    if model.model is None:
        raise RuntimeError("YOLO model has no detector attached")
    wrap_gate_only(model.model, reduction=reduction, scale=scale)
    return model
