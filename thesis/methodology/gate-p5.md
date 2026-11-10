---
title: Gate + Regularized P5 Attention (A+B)
status: draft
date: 2026-11-10
sources:
  - thesis/decisions/ADR-0010-gate-p5-design.md
  - thesis/methodology/architecture.md
  - thesis/methodology/weather-gate.md
  - thesis/findings/v1-p5-analysis.md
  - thesis/findings/weather-gate-analysis.md
  - src/litevit/models/blocks.py
  - src/litevit/models/gates.py
  - configs/train/gate_p5.yaml
---

# Gate + Regularized P5 Attention (A+B)

## Purpose

Combine the two mechanisms that have independent evidence:

- **A:** P3/P4/P5 weather gate (small positive on test, stable);
- **B:** P5 linear-attention block, regularized and reduced in capacity.

V1 showed that the unregularized P5 block overfits after epoch 18. A+B keeps the
same placement but adds regularization and reduces block capacity.

## Architecture

```text
YOLOv8n backbone
  └─ P5/SPPF → regularized LiteViT-P5 → PAN/FPN neck
                                            ├─ P3 → Gate-P3 ─┐
                                            ├─ P4 → Gate-P4 ─┼→ Detect head
                                            └─ P5 → Gate-P5 ─┘
```

P3 remains CNN-only until the gate. The P5 block is after the SPPF backbone layer
(index 9); gates wrap neck outputs 15/18/21.

## Regularized P5 block

The block is the V1 `LiteViTBlock` with:

| Setting | V1 | A+B | Reason |
|---|---:|---:|---|
| Heads | 4 | 4 | Keep the same attention geometry. |
| MLP ratio | 2.0 | **1.0** | Reduce capacity from 535k to 401k block parameters. |
| DropPath | 0.0 | **0.2** | Stochastic depth is standard for ViT/Swin/ConvNeXt on limited data. |
| Dropout | 0.0 | **0.1** | Standard transformer regularization. |
| LayerScale | zero | zero | Keep identity initialization. |
| Compute dtype | FP32 | FP32 | Avoid fp16 LayerNorm/attention instability. |

Block parameters: **401,408**. The block still starts as an exact identity.

## Gate

Unchanged from gate-only:

```text
w = 2 · sigmoid(Conv1x1(C/4 → C)(SiLU(Conv1x1(C → C/4)(GAP(F)))))
F' = F × w
```

- C/4 hidden widths: 16/32/64.
- Final layer zero-initialized: `w = 1` at start.
- Gate parameters: **43,568**.
- Applied to P3/P4/P5 after the neck.

## Size and compute

Measured from the five-class model summary:

- Parameters: **3,583,951**
- GFLOPs at 640: **9.099**
- Added over baseline: 444,976 parameters (+14.2%)
- Block: 401,408 parameters
- Gates: 43,568 parameters

## Training protocol

Same as `EXP-0001`:

- 640×640, batch 4, AMP, `nbs=8`
- AdamW, `lr0=2e-4`, cosine, 3 warmup epochs, weight decay 5e-4
- 100 epochs, patience 20, seed 42, `cls_pw=0.5`
- Mosaic 0.5, MixUp 0.05, `close_mosaic=10`

No learning-rate change for the block is applied; DropPath, dropout, and the smaller
MLP ratio are the intended stabilizers.

## Evidence

- V1 without regularization overfit after epoch 18; A+B reduces capacity and adds
  DropPath/dropout.
- DeiT, Swin, and ConvNeXt use stochastic depth and dropout/weight decay for
  transformer training on limited data.
- Gate-only was stable and slightly positive on test, especially at night/rain and
  for bicycle/bus.
- The exact A+B combination remains a hypothesis until `EXP-0005` completes.

## Status

Implemented. Training and evaluation are pending (`EXP-0005`).
