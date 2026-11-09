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
