---
title: ADR-0011 — Kitchen-Sink Weather Model (Gate + Edge + Spatial CBAM)
status: accepted
date: 2026-11-10
sources:
  - thesis/methodology/gate-edge-cbam.md
  - thesis/findings/weather-gate-analysis.md
  - thesis/findings/v1-p5-analysis.md
  - configs/train/gate_edge_cbam.yaml
---

# ADR-0011 — Kitchen-Sink Weather Model (Gate + Edge + Spatial CBAM)

## Context

V1 (P5 attention) overfit and hurt small/rare objects. A+B (gate + regularized P5
attention) was stable but neutral on test. Gate-only was stable and slightly
positive, especially at night/rain and for bicycle/bus.

The remaining errors are dominated by small objects and degraded boundaries. The
next attempt combines three cheap, complementary mechanisms without adding global
attention capacity:

- channel weather gate;
- Sobel edge residual at P3/P4;
- spatial CBAM after the gate.

TTA was tested on gate-only and was worse; it is not part of this model.

## Decision

Implement the kitchen-sink model:

- **Sobel edge residual** at backbone P3/P4 (layers 4/6):
  fixed Sobel filters, edge magnitude, 1x1 projection, zero-initialized gamma.
- **Channel gate** at neck P3/P4/P5 (layers 15/18/21): C/4, SiLU, 2·sigmoid,
  zero-initialized final layer.
- **Spatial CBAM** after the gate on P3/P4/P5: avg/max pooling, 7×7 conv,
  2·sigmoid, zero-initialized.
- No P5 or P4 attention block.
- Same training protocol as `EXP-0001`: 640, batch 4, AMP, `nbs=8`, AdamW 2e-4,
  100 epochs, patience 20, seed 42, `cls_pw=0.5`.
- Record as `EXP-0006`.

The model starts as an exact identity: edge gamma, gate final layer, and spatial
conv are zero-initialized.

## Consequences

- Total size is 3,203,704 parameters and 8.895 GFLOPs; batch 4 remains feasible.
- The combined run cannot attribute gains to edge, gate, or CBAM individually.
  Edge-only and CBAM-only ablations are required if attribution matters.
- Gate-only remains the clean fallback.
- TTA is excluded because it reduced gate-only mAP and doubled inference time.
- Single-seed and test-reuse caveats still apply.

## Status

Accepted.

## Supersedes

None.
