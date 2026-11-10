"""Add a stride-4 P2 detection branch to a YOLOv8 detector."""

from __future__ import annotations

import copy

import torch
from torch import nn
from ultralytics.nn.modules import C2f, Concat, Detect

P2_CHANNELS = 32
P2_SOURCE_INDEX = 2
P3_NECK_INDEX = 15
P4_NECK_INDEX = 18
P5_NECK_INDEX = 21
OLD_DETECT_INDEX = 22
P2_LAYER_INDEX = 24
DETECT_LAYER_INDEX = 25
EXPECTED_OLD_LAYERS = 23
P2_STRIDES = (4.0, 8.0, 16.0, 32.0)


def _out_channels(module: nn.Module) -> int:
    cv2 = getattr(module, "cv2", None)
    conv = getattr(cv2, "conv", None)
    if isinstance(conv, nn.Conv2d):
        return conv.out_channels
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Conv2d):
            return child.out_channels
    raise ValueError(f"could not infer output channels for {type(module).__name__}")


def _branch_in_channels(branch: nn.Module) -> int:
    first = branch[0]
    conv = getattr(first, "conv", None)
    if not isinstance(conv, nn.Conv2d):
        raise ValueError(f"could not infer input channels for {type(branch).__name__}")
    return conv.in_channels


def add_p2_head(detection: nn.Module) -> nn.Module:
    """Append a P2 detect branch while preserving the existing P3/P4/P5 branches.

    The new branch upsamples the P3 neck output, concatenates the backbone P2
    output, refines it with C2f, and feeds a fourth Detect level. Existing
    P3/P4/P5 box and class branches are copied into the new four-level Detect so
    pretrained capacity is retained. The new P2 branch is randomly initialized;
    this is not an identity-preserving operation.
    """
    layers = getattr(detection, "model", None)
    if layers is None:
        raise RuntimeError("detection model has no layer sequence")
    if len(layers) == EXPECTED_OLD_LAYERS + 3 and isinstance(layers[-1], Detect):
        if layers[-1].nl == 4:
            return detection
    if len(layers) != EXPECTED_OLD_LAYERS:
        raise RuntimeError(
            f"expected a {EXPECTED_OLD_LAYERS}-layer YOLOv8 detector, got {len(layers)}"
        )
    old_detect = layers[-1]
    if not isinstance(old_detect, Detect):
        raise RuntimeError("last detector layer is not Detect")
    if old_detect.nl != 3:
        raise RuntimeError(f"expected 3 detection levels, got {old_detect.nl}")
    if getattr(old_detect, "one2one_cv2", None) is not None:
        raise RuntimeError("end-to-end Detect heads are not supported")

    old_channels = [
        _branch_in_channels(old_detect.cv2[index]) for index in range(old_detect.nl)
    ]
    p2_backbone_channels = _out_channels(layers[P2_SOURCE_INDEX])
    p3_neck_channels = old_channels[0]

    new_detect = Detect(
        nc=old_detect.nc, reg_max=old_detect.reg_max, ch=[P2_CHANNELS, *old_channels]
    )
    new_detect.stride = torch.tensor(P2_STRIDES)
    new_detect.bias_init()
    new_detect.cv2 = nn.ModuleList(
        [new_detect.cv2[0], *(copy.deepcopy(branch) for branch in old_detect.cv2)]
    )
    new_detect.cv3 = nn.ModuleList(
        [new_detect.cv3[0], *(copy.deepcopy(branch) for branch in old_detect.cv3)]
    )
    new_detect.dfl.load_state_dict(old_detect.dfl.state_dict())
    new_detect.f = [
        P2_LAYER_INDEX,
        P3_NECK_INDEX,
        P4_NECK_INDEX,
        P5_NECK_INDEX,
    ]
    new_detect.i = DETECT_LAYER_INDEX
    new_detect.type = "Detect"
    new_detect.np = sum(parameter.numel() for parameter in new_detect.parameters())

    upsample = nn.Upsample(scale_factor=2, mode="nearest")
    upsample.f = P3_NECK_INDEX
    upsample.i = 22
    upsample.type = "nn.Upsample"
    upsample.np = 0

    concat = Concat(1)
    concat.f = [-1, P2_SOURCE_INDEX]
    concat.i = 23
    concat.type = "Concat"
    concat.np = 0

    c2f = C2f(p3_neck_channels + p2_backbone_channels, P2_CHANNELS, n=1, shortcut=True)
    c2f.f = -1
    c2f.i = P2_LAYER_INDEX
    c2f.type = "C2f"
    c2f.np = sum(parameter.numel() for parameter in c2f.parameters())

    branches = [layers[index] for index in range(OLD_DETECT_INDEX)]
    detection.model = nn.Sequential(*branches, upsample, concat, c2f, new_detect)
    detection.save = sorted({*getattr(detection, "save", []), P2_SOURCE_INDEX, P2_LAYER_INDEX})
    detection.stride = new_detect.stride
    return detection
