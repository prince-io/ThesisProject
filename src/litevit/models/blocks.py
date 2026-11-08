"""Lightweight linear-attention blocks for the LiteViT neck."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension of a BCHW tensor."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class LinearAttention2d(nn.Module):
    """Linear attention over flattened spatial tokens.

    Expects a fused QKV tensor of shape (B, 3C, H, W). Attention is computed in
    float32 for numerical stability under AMP, then cast back to the input dtype.
    """

    def __init__(self, channels: int, heads: int = 4, eps: float = 1e-6) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.eps = eps

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        batch, fused_channels, height, width = qkv.shape
        if fused_channels != 3 * self.channels:
            raise ValueError(
                f"expected {3 * self.channels} fused channels, got {fused_channels}"
            )
        tokens = height * width
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(batch, self.heads, self.head_dim, tokens).float()
        k = k.reshape(batch, self.heads, self.head_dim, tokens).float()
        v = v.reshape(batch, self.heads, self.head_dim, tokens).float()

        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        kv = torch.einsum("bhdn,bhen->bhde", k, v)
        normalizer = torch.einsum("bhdn,bhd->bhn", q, k.sum(dim=-1))
        out = torch.einsum("bhdn,bhde->bhen", q, kv)
        out = out / (normalizer.unsqueeze(2) + self.eps)
        out = out.reshape(batch, self.channels, height, width)
        return out.to(dtype=qkv.dtype)


class ConvFFN(nn.Module):
    """Two-layer ConvFFN with a depthwise 3x3 local mixing stage."""

    def __init__(self, channels: int, ratio: float = 2.0) -> None:
        super().__init__()
        hidden = int(channels * ratio)
        if hidden < 1:
            raise ValueError("ConvFFN hidden width must be positive")
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.dw = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.dw(x))
        return self.fc2(x)


class LiteViTBlock(nn.Module):
    """Channel-preserving linear-attention block with identity initialization.

    The two LayerScale parameters are zero at initialization, so the block output
    is exactly its input until training updates the scales. This lets the wrapped
    detector start from the pretrained YOLOv8n behavior.
    """

    def __init__(
        self,
        channels: int,
        heads: int = 4,
        mlp_ratio: float = 2.0,
        layer_scale_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.norm1 = LayerNorm2d(channels)
        self.cpe = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.attention = LinearAttention2d(channels, heads=heads)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.full((1, channels, 1, 1), layer_scale_init))
        self.norm2 = LayerNorm2d(channels)
        self.ffn = ConvFFN(channels, ratio=mlp_ratio)
        self.gamma2 = nn.Parameter(torch.full((1, channels, 1, 1), layer_scale_init))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in (self.cpe, self.qkv, self.proj, self.ffn.fc1, self.ffn.dw, self.ffn.fc2):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = y + self.cpe(y)
        y = self.attention(self.qkv(y))
        x = x + self.gamma1 * self.proj(y)
        x = x + self.gamma2 * self.ffn(self.norm2(x))
        return x


class FeatureWithBlock(nn.Module):
    """Wrap an existing detector layer and apply a channel-preserving block after it."""

    def __init__(self, base: nn.Module, block: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.block = block
        self.i = getattr(base, "i", None)
        self.f = getattr(base, "f", -1)
        self.type = getattr(base, "type", type(base).__name__)
        self.np = getattr(base, "np", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.base(x))
