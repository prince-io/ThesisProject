"""Build LiteViT variants by wrapping pretrained YOLO detector layers."""

from __future__ import annotations

from pathlib import Path

from torch import nn
from ultralytics import YOLO

from litevit.models.blocks import FeatureWithBlock, LiteViTBlock

P5_LAYER_INDEX = 9


def _infer_out_channels(module: nn.Module) -> int:
    """Infer the output channel count from the last Conv2d in a module."""
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Conv2d):
            return child.out_channels
    raise ValueError(f"could not infer output channels for {type(module).__name__}")


def _wrap_layer(model: YOLO, index: int, block: nn.Module) -> None:
    detection = model.model
    if detection is None:
        raise RuntimeError("YOLO model has no detector attached")
    layers = detection.model
    if not 0 <= index < len(layers):
        raise IndexError(f"layer index {index} outside model with {len(layers)} layers")
    base = layers[index]
    layers[index] = FeatureWithBlock(base, block)


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
    detection = model.model
    if detection is None:
        raise RuntimeError("YOLO model has no detector attached")
    base = detection.model[P5_LAYER_INDEX]
    dim = _infer_out_channels(base)
    block = LiteViTBlock(dim, heads=heads, mlp_ratio=mlp_ratio)
    _wrap_layer(model, P5_LAYER_INDEX, block)
    return model
