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
            hidden = F.conv2d(
                pooled, self.fc1.weight.float(), self.fc1.bias.float()
            )
            hidden = F.silu(hidden)
            weight = F.conv2d(
                hidden, self.fc2.weight.float(), self.fc2.bias.float()
            )
            weight = self.scale * torch.sigmoid(weight)
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


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention with identity initialization."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=True
        )
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            avg = x.mean(dim=1, keepdim=True)
            max_value = x.max(dim=1, keepdim=True).values
            pooled = torch.cat([avg, max_value], dim=1)
            weight = F.conv2d(
                pooled,
                self.conv.weight.float(),
                self.conv.bias.float(),
                padding=self.conv.padding,
            )
            weight = 2.0 * torch.sigmoid(weight)
            x = x * weight
        return x.to(dtype=input_dtype)


class GateWithSpatial(nn.Module):
    """Channel weather gate followed by spatial attention."""

    def __init__(
        self,
        channels: int,
        reduction: int = GATE_REDUCTION,
        scale: float = GATE_SCALE,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.channel = WeatherGate(channels, reduction=reduction, scale=scale)
        self.spatial = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))
