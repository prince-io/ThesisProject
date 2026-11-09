"""Lightweight weather channel gating modules."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

GATE_REDUCTION = 4
GATE_SCALE = 2.0


class WeatherGate(nn.Module):
    """Squeeze-excitation style channel gate with identity initialization.

    ``w = GATE_SCALE * sigmoid(fc2(SiLU(fc1(GAP(x)))))``. With the final layer
    zero-initialized, ``w = 1`` exactly, so the gate starts as an identity.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = GATE_REDUCTION,
        scale: float = GATE_SCALE,
    ) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.act = nn.SiLU()
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.scale = scale
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            pooled = F.adaptive_avg_pool2d(x, 1)
            weight = self.scale * torch.sigmoid(self.fc2(self.act(self.fc1(pooled))))
            x = x * weight
        return x.to(dtype=input_dtype)


class FeatureWithGate(nn.Module):
    """Wrap an existing detector layer and apply a channel gate after it."""

    def __init__(self, base: nn.Module, gate: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.gate = gate
        self.i = getattr(base, "i", None)
        self.f = getattr(base, "f", -1)
        self.type = getattr(base, "type", type(base).__name__)
        self.np = getattr(base, "np", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(self.base(x))
