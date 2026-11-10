---
title: Gate + Sobel Edge + Spatial CBAM (Kitchen Sink)
status: draft
date: 2026-11-10
sources:
  - thesis/decisions/ADR-0011-gate-edge-cbam-design.md
  - thesis/methodology/weather-gate.md
  - thesis/findings/weather-gate-analysis.md
  - thesis/findings/v1-p5-analysis.md
  - src/litevit/models/edge.py
  - src/litevit/models/gates.py
  - configs/train/gate_edge_cbam.yaml
---

# Gate + Sobel Edge + Spatial CBAM (Kitchen Sink)

## Purpose

Test the cheapest combination that targets the two main error sources:

- **weather-degraded features** → channel gate;
- **lost object boundaries** → Sobel edge branch;
- **where to look spatially** → spatial CBAM.

This model deliberately excludes the P5 attention block. V1 and A+B showed that
extra global attention at P5 was neutral or harmful on this benchmark.

## Architecture

```text
YOLOv8n backbone
  P3/8  → Sobel edge residual → ┐
  P4/16 → Sobel edge residual → ┤
  P5/32 ---------------------→ ┤
                                ▼
                           PAN/FPN neck
                                │
                 P3 → gate + spatial attention ─┐
                 P4 → gate + spatial attention ─┼→ Detect head
                 P5 → gate + spatial attention ─┘
```

Layer indices:

- edge residuals: backbone P3 layer 4, P4 layer 6;
- channel gate + spatial attention: neck P3/P4/P5 layers 15/18/21.

All wrappers preserve the existing `i`, `f`, `type`, and output shape.

## Sobel edge residual

For each P3/P4 feature `x`:

```text
gx, gy = fixed Sobel x/y filters (depthwise)
edge   = sqrt(gx² + gy² + eps)
x'     = x + gamma · Conv1x1(edge)
```

- Sobel filters are fixed, non-trainable buffers.
- `gamma` is zero-initialized, so the edge branch starts as an exact identity.
- P3/P4 chosen because edges are most useful at higher resolution; P5 is too coarse.
- Parameters: P3 4,224; P4 16,640; total **20,864**.

## Channel gate

Unchanged from gate-only:

```text
w = 2 · sigmoid(Conv1x1(C/4 → C)(SiLU(Conv1x1(C → C/4)(GAP(F)))))
F' = F × w
```

- C/4 reduction; final layer zero-initialized; identity at start.
- Parameters: **43,568** across P3/P4/P5.

## Spatial CBAM

Applied after the channel gate on P3/P4/P5:

```text
s = concat(mean_c(F'), max_c(F'))
w = 2 · sigmoid(Conv7x7(2 → 1)(s))
F'' = F' × w
```

- Spatial-only: the channel gate already handles channel attention.
- 7×7 convolution, zero-initialized; identity at start.
- Parameters: **297** across three scales (99 each).

## Size and compute

| Component | Parameters |
|---|---:|
| YOLOv8n 5-class baseline | 3,138,975 |
| Sobel edge residual P3/P4 | 20,864 |
| Channel gates P3/P4/P5 | 43,568 |
| Spatial attention P3/P4/P5 | 297 |
| **Total** | **3,203,704** |

GFLOPs at 640: **8.895**. Batch remains 4; memory increase is small.

## Training protocol

Same as `EXP-0001`:

- 640×640, batch 4, AMP, `nbs=8`
- AdamW, `lr0=2e-4`, cosine, 3 warmup epochs, weight decay 5e-4
- 100 epochs, patience 20, seed 42, `cls_pw=0.5`
- Mosaic 0.5, MixUp 0.05, `close_mosaic=10`

## Evidence and caveat

- ECL-YOLOv11 reports a Sobel edge-enhancement branch helps adverse-weather detection.
- YOLOv8-SCS uses CBAM and reports gains on DAWN.
- Gate-only was stable and slightly positive on test, especially night/rain and
  bicycle/bus.
- This combined run confounds the three additions; separate edge-only and
  CBAM-only ablations are needed if it improves and attribution matters.

## Status

Implemented. Training and evaluation are pending (`EXP-0006`).
