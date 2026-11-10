---
title: ADR-0012 — P2 High-Resolution Detection Head
status: accepted
date: 2026-11-10
sources:
  - thesis/methodology/p2-head.md
  - thesis/findings/gate-p5-analysis.md
  - thesis/results/gate_edge_cbam.md
  - configs/train/gate_p2.yaml
  - src/litevit/models/p2.py
---

# ADR-0012 — P2 High-Resolution Detection Head

## Context

Gate-only remains the best proposed model on test, but the gain over the baseline
is small. A+B and the kitchen-sink model were neutral, and TTA was negative. The
remaining error is concentrated in tiny objects: tiny recall is about 0.34 and
night/rain are still weak.

The next experiment targets resolution rather than more attention or channel
modules. A stride-4 P2 detection level gives small objects a finer grid and more
positive assignments.

## Decision

Implement a gate-only model with a standard stride-4 P2 detection head:

- add a P2 branch: upsample P3 neck, concatenate backbone P2, C2f to 32 channels;
- use four Detect levels P2/P3/P4/P5 with strides 4/8/16/32;
- preserve the pretrained P3/P4/P5 Detect box and class branches;
- keep weather gates on P3/P4/P5 only; do not gate P2;
- no P5 attention, Sobel edge, spatial CBAM, TTA, or model soup;
- same training protocol as `EXP-0004`: 640, batch 4, AMP, 100 epochs,
  patience 20, seed 42, `cls_pw=0.5`;
- record as `EXP-0007`, architecture id `gate_p2`.

The P2 branch is not identity-initialized. The rest of the pretrained graph is
preserved, but initial detections differ from gate-only.

## Consequences

- Size: 3,270,388 parameters and 13.3 GFLOPs; about 51% more compute than gate-only.
- Expected training time: 7-8 h on the RTX 3050; batch 4 must be confirmed by smoke.
- The model may improve tiny recall but increase false positives and latency.
- This is a single-seed experiment with test reuse; sub-0.5 pp overall differences
  are not strong evidence.
- If P2 is neutral or negative, stop architecture expansion and finalize gate-only.

## Status

Accepted.

## Supersedes

None.
