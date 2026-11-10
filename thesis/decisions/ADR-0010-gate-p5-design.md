---
title: ADR-0010 — A+B Gate Plus Regularized P5 Attention
status: accepted
date: 2026-11-10
sources:
  - thesis/methodology/gate-p5.md
  - thesis/findings/v1-p5-analysis.md
  - thesis/findings/weather-gate-analysis.md
  - configs/train/gate_p5.yaml
---

# ADR-0010 — A+B Gate Plus Regularized P5 Attention

## Context

V1 (P5 linear attention, no gate) overfit after epoch 18 and underperformed the
baseline on test. Gate-only was stable and slightly positive on test, especially for
night/rain and bicycle/bus, but the overall gain was within single-seed noise.

The next question is whether a regularized, lower-capacity P5 block can add value on
top of the gate without repeating V1's overfitting.

## Decision

Implement A+B:

- P5 `LiteViTBlock` after SPPF with:
  - 4 heads;
  - MLP ratio 1.0;
  - DropPath 0.2;
  - dropout 0.1;
  - zero-initialized LayerScale;
  - FP32 block execution under AMP.
- Weather gate on neck P3/P4/P5 with the gate-only design:
  - C/4 reduction;
  - SiLU;
  - `2·sigmoid`;
  - zero-initialized final layer.
- Same training protocol as `EXP-0001`: 640, batch 4, AMP, `nbs=8`, AdamW 2e-4,
  100 epochs, patience 20, seed 42, `cls_pw=0.5`.
- Record as `EXP-0005`.

The model starts as an exact identity because the block LayerScale and gate final
layer are zero-initialized.

## Consequences

- A+B tests whether attention can add to the gate after regularization.
- It preserves the gate-only model as a fallback if attention still hurts.
- The block adds 401,408 parameters and the gates 43,568; total 3,583,951
  parameters (+14.2% over baseline) and 9.099 GFLOPs.
- If A+B does not improve validation over gate-only and baseline, keep gate-only as
  the final proposed variant and report P5 attention as a negative ablation.
- Single-seed and test-reuse caveats still apply.

## Status

Accepted.

## Supersedes

None.
