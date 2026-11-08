---
title: ADR-0008 — Freeze Class Weighting at cls_pw=0.5
status: accepted
date: 2026-11-08
sources:
  - thesis/results/ablation-comparison.md
  - thesis/results/ablation_cls_pw0.md
  - thesis/results/ablation_cls_pw0-error-analysis.md
  - thesis/findings/ablation-analysis.md
  - thesis/decisions/ADR-0006-baseline-protocol.md
---

# ADR-0008 — Freeze Class Weighting at cls_pw=0.5

## Context

The YOLOv8n baseline (`EXP-0001`) uses inverse-frequency class weights with
`cls_pw=0.5`. `EXP-0002` tested disabling class weighting (`cls_pw=0.0`) to see
whether it improved rare-class performance or test mAP. The ablation was
effectively tied on validation and lower on test mAP; rare-class AP changes were
mixed, and the thresholded precision/recall gain at confidence 0.25 was not
matched by ranking-based mAP. The proposed LiteViT-WG model needs a fixed class
weighting setting so that architecture comparisons are not confounded.

## Decision

Retain `cls_pw=0.5` for LiteViT-WG and all subsequent comparisons with `EXP-0001`.
Do not tune class weighting further in Stage 2A; treat it as a fixed protocol
component. The ablation remains the evidence for this choice.

## Consequences

- The proposed model can be compared directly with `EXP-0001` without re-running
  the baseline.
- Class weighting is no longer an experimental variable in Stage 2B.
- The ablation does not establish class weighting as the cause of rare-class
  weakness; architecture, resolution, and weather adaptation remain the primary
  levers.
- Single-seed and test-reuse caveats apply; multi-seed validation is still needed
  before final claims.
- Any future class-weighting tuning must be a new experiment with its own record,
  not a silent change to the reference protocol.

## Status

Accepted.

## Supersedes

None. Complements `ADR-0006` and `ADR-0007`.
