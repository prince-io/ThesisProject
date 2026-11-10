"""Build LiteViT variants by wrapping pretrained YOLO detector layers."""

from __future__ import annotations

from pathlib import Path

from torch import nn
from ultralytics import YOLO

from litevit.models.blocks import FeatureWithBlock, LiteViTBlock
from litevit.models.edge import EdgeEnhance, FeatureWithEdge
from litevit.models.gates import (
    GATE_REDUCTION,
    GATE_SCALE,
    FeatureWithGate,
    GateWithSpatial,
    WeatherGate,
)

P5_LAYER_INDEX = 9
GATE_LAYER_INDICES = (15, 18, 21)
EDGE_LAYER_INDICES = (4, 6)
GATE_P5_HEADS = 4
GATE_P5_MLP_RATIO = 1.0
GATE_P5_DROP_PATH = 0.2
GATE_P5_DROPOUT = 0.1


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
    drop_path: float = 0.0,
    dropout: float = 0.0,
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
    block = LiteViTBlock(
        dim,
        heads=heads,
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
        dropout=dropout,
    )
    layers[P5_LAYER_INDEX] = FeatureWithBlock(base, block)
    return detection


def build_v1_p5(
    weights: str | Path = "yolov8n.pt",
    heads: int = 4,
    mlp_ratio: float = 2.0,
    drop_path: float = 0.0,
    dropout: float = 0.0,
) -> YOLO:
    """Build V1: one LiteViT block after the P5/SPPF backbone layer.

    The pretrained YOLOv8n weights are loaded first, so all original backbone,
    neck, and head weights are preserved. The new block is channel-preserving and
    identity-initialized.
    """
    model = YOLO(str(weights))
    if model.model is None:
        raise RuntimeError("YOLO model has no detector attached")
    wrap_v1_p5(
        model.model,
        heads=heads,
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
        dropout=dropout,
    )
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


def wrap_gate_p5(
    detection: nn.Module,
    heads: int = GATE_P5_HEADS,
    mlp_ratio: float = GATE_P5_MLP_RATIO,
    drop_path: float = GATE_P5_DROP_PATH,
    dropout: float = GATE_P5_DROPOUT,
) -> nn.Module:
    """Wrap P5 attention and P3/P4/P5 weather gates on an existing detector."""
    wrap_v1_p5(
        detection,
        heads=heads,
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
        dropout=dropout,
    )
    return wrap_gate_only(detection)


def build_gate_p5(
    weights: str | Path = "yolov8n.pt",
    heads: int = GATE_P5_HEADS,
    mlp_ratio: float = GATE_P5_MLP_RATIO,
    drop_path: float = GATE_P5_DROP_PATH,
    dropout: float = GATE_P5_DROPOUT,
) -> YOLO:
    """Build A+B: regularized P5 attention plus P3/P4/P5 weather gates."""
    model = YOLO(str(weights))
    if model.model is None:
        raise RuntimeError("YOLO model has no detector attached")
    wrap_gate_p5(
        model.model,
        heads=heads,
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
        dropout=dropout,
    )
    return model


def wrap_edge(detection: nn.Module) -> nn.Module:
    """Wrap backbone P3/P4 outputs with the Sobel edge residual."""
    layers = getattr(detection, "model", None)
    if layers is None:
        raise RuntimeError("detection model has no layer sequence")
    for index in EDGE_LAYER_INDICES:
        if not 0 <= index < len(layers):
            raise IndexError(
                f"layer index {index} outside model with {len(layers)} layers"
            )
        base = layers[index]
        if isinstance(base, FeatureWithEdge):
            continue
        layers[index] = FeatureWithEdge(base, EdgeEnhance(_infer_out_channels(base)))
    return detection


def wrap_gate_spatial(detection: nn.Module) -> nn.Module:
    """Wrap neck P3/P4/P5 outputs with channel gate plus spatial attention."""
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
        layers[index] = FeatureWithGate(
            base, GateWithSpatial(_infer_out_channels(base))
        )
    return detection


def wrap_gate_edge_cbam(detection: nn.Module) -> nn.Module:
    """Wrap edge enhancement plus channel/spatial weather attention."""
    return wrap_gate_spatial(wrap_edge(detection))


def build_gate_edge_cbam(weights: str | Path = "yolov8n.pt") -> YOLO:
    """Build the kitchen-sink model: edge + channel gate + spatial CBAM."""
    model = YOLO(str(weights))
    if model.model is None:
        raise RuntimeError("YOLO model has no detector attached")
    wrap_gate_edge_cbam(model.model)
    return model
