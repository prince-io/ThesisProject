---
title: ADR-0009 — Gate-Only Weather Adaptation Design
status: accepted
date: 2026-11-09
sources:
  - thesis/methodology/weather-gate.md
  - thesis/related-work/weather-adaptive-gating.md
  - thesis/related-work/architecture-grounding.md
  - thesis/results/litevit_v1_p5.md
---

# ADR-0009 — Gate-Only Weather Adaptation Design

## Context

V1 (P5 linear-attention block) peaked at epoch 18 and then overfit: validation
classification loss rose from 0.283 to 0.550 while training loss kept falling. Its
best checkpoint was below the baseline on test mAP and on tiny/rare-object recall.
The P5 block adds capacity but does not reliably improve adverse-weather detection.

The weather-adaptation hypothesis still needs a clean test without attention
capacity. A channel gate is inexpensive, identity-initialized, and directly targets
degraded feature channels.

## Decision

Implement a gate-only model: YOLOv8n with a Lightweight Weather Gate (L-WG) after
the neck on P3, P4, and P5.

- `g = global average pool(F)`
- `h = SiLU(Conv1x1(C → C/4)(g))`
- `w = 2 · sigmoid(Conv1x1(C/4 → C)(h))`
- `F' = F × w`
- Final 1×1 convolution initialized to zero weight and bias, so the gate starts as
  an exact identity.
- No spatial attention, no resolution change, no layer-index change.
- Same training protocol as `EXP-0001`: 640, batch 4, AMP, `nbs=8`, AdamW 2e-4,
  100 epochs, patience 20, seed 42, `cls_pw=0.5`.
- Run recorded as `EXP-0004`.

The reduction ratio C/4 is chosen for the small YOLOv8n neck channels: C/16 would
leave only 4 channels at P3, while C/2 would roughly double the parameter cost.
The gate adds 43,568 parameters (1.4% over the baseline).

## Consequences

- The gate-only run isolates the weather mechanism from attention.
- If gate-only improves results, a later combined model can add a regularized P4/P5
  attention block.
- If gate-only does not improve results, the weather-adaptation hypothesis is weak
  on this benchmark and should not be hidden behind a larger attention model.
- The design is inspired by SE/CBAM/WAAM but is not yet validated; no performance
  claim is made until `EXP-0004` completes.
- The test split remains reused from Stage 2A; results are provisional.

## Status

Accepted.

## Supersedes

None.
