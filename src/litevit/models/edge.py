"""Lightweight Sobel edge-enhancement branch."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class EdgeEnhance(nn.Module):
    """Zero-initialized Sobel edge residual.

    The fixed Sobel filters extract horizontal/vertical gradients. A 1x1
    projection and LayerScale-style gamma add the edge signal back to the
    feature map. Gamma is zero at initialization, so the module is an identity.
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.channels = channels
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x.repeat(channels, 1, 1, 1))
        self.register_buffer("sobel_y", sobel_y.repeat(channels, 1, 1, 1))
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            gx = F.conv2d(x, self.sobel_x, padding=1, groups=self.channels)
            gy = F.conv2d(x, self.sobel_y, padding=1, groups=self.channels)
            edge = torch.sqrt(gx.square() + gy.square() + self.eps)
            x = x + self.gamma * self.proj(edge)
        return x.to(dtype=input_dtype)


class FeatureWithEdge(nn.Module):
    """Wrap an existing detector layer and apply edge enhancement after it."""

    def __init__(self, base: nn.Module, edge: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.edge = edge
        self.i = getattr(base, "i", None)
        self.f = getattr(base, "f", -1)
        self.type = getattr(base, "type", type(base).__name__)
        self.np = getattr(base, "np", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.edge(self.base(x))
