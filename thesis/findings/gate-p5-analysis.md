---
title: A+B Analysis — Gate + Regularized P5 Attention
status: draft
date: 2026-11-10
sources:
  - thesis/results/gate_p5.md
  - thesis/results/gate_p5-error-analysis.md
  - thesis/results/architecture-comparison.md
---

# A+B Analysis — Gate + Regularized P5 Attention

## Observation

A+B was stable but did not improve test performance over baseline or gate-only
([A+B results](../results/gate_p5.md)):

- test mAP50 0.5387 vs baseline 0.5428 and gate-only 0.5444
- test mAP50-95 0.3290 vs baseline 0.3287 and gate-only 0.3301
- validation mAP50-95 0.2938, slightly above baseline 0.2904 and gate-only 0.2891

Training stopped at epoch 86; best mAP50-95 was epoch 66. It was stable, unlike V1.
A late epoch-86 checkpoint was also evaluated: test mAP50 0.5414 / mAP50-95 0.3307,
still within noise.

Error analysis shows the model is more conservative: higher precision, lower
recall, and lower tiny-object recall than gate-only
([A+B error analysis](../results/gate_p5-error-analysis.md)).

## Interpretation

1. **Regularization fixed V1's overfitting**, but did not make attention useful.
2. **The gate plus attention does not beat the gate alone.** P5 attention adds
   capacity but no consistent gain.
3. **Tiny-object performance is still worse** than gate-only/baseline, consistent
   with attention at P5 diluting high-resolution detail.
4. **A+B is a neutral/negative ablation**, not the final model.

## Limitations

- Single seed; differences are within run noise.
- Test split reuse.
- Only one attention regularization configuration.
- A+B confounds gate and attention; gate-only and V1 provide partial controls.

## Implication

Keep gate-only as the strongest candidate so far. The next attempt should target
the actual error buckets: small objects and degraded boundaries, using edge
enhancement and spatial attention rather than more global attention. That is the
kitchen-sink experiment (`EXP-0006`).

## Status

Draft. Based on completed `EXP-0005`.
