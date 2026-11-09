---
title: V1 P5 LiteViT Analysis — Negative Result
status: draft
date: 2026-11-09
sources:
  - thesis/results/litevit_v1_p5.md
  - thesis/results/litevit_v1_p5-error-analysis.md
  - runs/eval/litevit_v1_p5_seed42_v2/error_analysis/error_analysis.json
---

# V1 P5 LiteViT Analysis — Negative Result

## Observation

V1 early-stopped at epoch 38 with best epoch 18. Its best checkpoint was below the
`EXP-0001` baseline on test: mAP50 0.4888 vs 0.5428 and mAP50-95 0.2895 vs 0.3287
([V1 results](../results/litevit_v1_p5.md)).

At epoch 18, V1 and the baseline were effectively tied on validation (0.4231/0.2499
vs 0.4234/0.2485). The baseline continued improving through epoch 38; V1 did not.

Training and validation diverged after epoch 18:

- train classification loss fell from 0.303 to 0.272;
- validation classification loss rose from 0.283 to 0.550;
- best mAP was reached at epoch 18, then declined.

Error analysis at confidence 0.25 shows the block hurt the difficult cases
([V1 error analysis](../results/litevit_v1_p5-error-analysis.md)):

- tiny-object recall: 0.334 → 0.270
- bus recall: 0.387 → 0.279
- bicycle recall: 0.352 → 0.290
- night recall: 0.601 → 0.565

## Interpretation

1. **The P5 block is active but not beneficial.** It learns and changes features, but
   its best result is only baseline-level at the same epoch and then overfits.
2. **The failure mode is overfitting**, not numerical instability: training loss
   decreases while validation classification loss rises.
3. **Small and rare objects are most affected**, consistent with adding global
   attention at P5 without enough regularization or local detail.
4. **Do not claim a ViT gain from V1.** The correct conclusion is a negative result:
   one unregularized linear-attention block at P5 does not improve this benchmark.

## Limitations

- Single seed; run-to-run variation is not measured.
- Early stopping selected epoch 18; later epochs could theoretically recover, but
  the validation trend is clearly downward.
- The test split has been reused across Stage 2A runs.
- This experiment does not separate the effect of the block from the effect of
  reduced training time after early stopping.

## Implications

- Test the weather gate on its own before adding more attention capacity.
- If attention is revisited, regularize and reduce capacity:
  DropPath, dropout, smaller MLP ratio, or a standard/Swin block.
- Keep P3 unchanged and protect small-object detail.
- Report per-class and per-size results; overall mAP hides these regressions.

## Next checks

1. Run gate-only weather adaptation (`EXP-0004`).
2. If gate-only helps, add a regularized P4/P5 attention block later.
3. If gate-only does not help, treat P5 attention as a documented negative ablation.

## Status

Draft. Based on the completed V1 run and evaluation.
