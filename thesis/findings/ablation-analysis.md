---
title: Ablation Analysis — cls_pw=0.0
status: draft
date: 2026-11-08
sources:
  - thesis/results/ablation-comparison.md
  - thesis/results/ablation_cls_pw0.md
  - thesis/results/ablation_cls_pw0-error-analysis.md
  - thesis/results/baseline.md
  - thesis/results/baseline-error-analysis.md
---

# Ablation Analysis — cls_pw=0.0

## Observation

The ablation keeps the architecture, data, split, seed, and training schedule fixed
and only disables class weighting (`cls_pw=0.5` → `0.0`). On validation the two runs
are effectively tied: mAP50 changes by +0.0012 and mAP50-95 by +0.0003
([comparison](../results/ablation-comparison.md)). On test the ablation is lower by
0.0114 mAP50 and 0.0034 mAP50-95.

The result depends on the metric family. At the error-analysis operating point
(confidence 0.25, IoU 0.5), the ablation has slightly higher precision and recall
on both splits, test F1 rises from 0.672 to 0.681, and test false positives/false
negatives fall ([ablation error analysis](../results/ablation_cls_pw0-error-analysis.md)).
Its ranking-based test mAP is nevertheless lower. This is a calibration/ranking
trade-off, not a uniform improvement or degradation.

Class effects are mixed. Bus AP50 rises on validation but falls on test; bicycle
AP50 falls on both splits; motorcycle test AP50 falls by 0.0352; person and car
change only slightly ([comparison](../results/ablation-comparison.md)). There is
no consistent rare-class gain. Condition and source deltas are also mixed: night
test mAP50 rises by 0.0324 and snow validation mAP50 by 0.0284, while fog test
mAP50 falls by 0.0166 and RTTS test mAP50 by 0.0181. DAWN test mAP50 rises by
0.0371 but its mAP50-95 falls by 0.0427
([comparison](../results/ablation-comparison.md)).

At confidence 0.25, recall rises slightly in every size bin, including tiny
objects (+0.0056 test), and background car false positives fall by 178
([comparison](../results/ablation-comparison.md)). Tiny-object recall remains
around 0.34, far below small/medium/large recall.

## Interpretation

1. **Class weighting is not a major lever on this benchmark.** The validation
   difference is negligible for a single seed, and the test direction reverses
   between mAP and thresholded precision/recall. Disabling class weighting does
   not produce a clear gain.
2. **The rare-class hypothesis is not supported.** Inverse-frequency weighting
   was intended to help bus, bicycle, and motorcycle. The ablation improves bus
   AP50 on validation only and lowers bicycle AP50 on both splits, so the evidence
   does not show that class weighting is what limits rare-class performance.
3. **The thresholded-recall gain is mostly calibration.** Slightly higher recall
   at confidence 0.25 across all size bins, together with lower test mAP, is
   consistent with a confidence-distribution shift rather than better
   localization. Tiny-object recall stays near 0.34, so the main small-object
   problem remains.
4. **Condition/source deltas are not stable enough to interpret.** The night and
   DAWN mAP50 gains are not matched by mAP50-95, and several other subsets move
   in the opposite direction. With one seed and small per-source subsets, these
   are provisional.
5. **Retain `cls_pw=0.5` for the proposed model.** The baseline has the better
   test mAP ([comparison](../results/ablation-comparison.md)), the ablation does
   not establish a benefit, and keeping the setting fixed preserves comparability
   with `EXP-0001`. This is recorded in
   `decisions/ADR-0008-freeze-class-weighting.md`.

## Limitations

- Single seed (42); a ~0.01 mAP difference is not statistically established.
- The test split has now been used for both the baseline and the ablation, so it
  is no longer a fully untouched holdout.
- Only two class-weighting settings are compared; intermediate powers and focal
  loss are not tested.
- Validation and test differ in class/source composition.
- Per-source subsets are small (ACDC, DAWN, RTTS), so their deltas are noisy.
- The baseline VRAM reserved/GPU-used values come from a calibration run, while
  the ablation values come from the full run; the VRAM comparison is not
  apples-to-apples.
- Error analysis uses a fixed confidence threshold; mAP integrates over
  thresholds. Neither alone is sufficient.

## Implications for LiteViT-WG

- Freeze `cls_pw=0.5` for LiteViT-WG and all comparisons; do not spend Stage 2B
  time tuning class weighting.
- Treat architecture and weather adaptation as the main levers, especially for
  tiny objects, night, rain, snow, and ACDC/RTTS.
- Keep P3 high-resolution features for small objects; restrict attention to
  P4/P5 as planned.
- Report both mAP and thresholded precision/recall, plus per-class, per-size, and
  per-condition results; overall mAP hides the threshold/ranking trade-off.
- Use validation for model selection and reserve the test split for the final
  model to limit adaptive overfitting.

## Next checks

1. Implement the LiteViT/weather-gate technical spike and validate 640x640
   training memory on the RTX 3050.
2. Run multi-seed baselines and proposed models; report mean ± standard deviation.
3. Review the ablation's qualitative FN/FP overlays against the baseline to check
   whether the failure modes changed.
4. If time permits, test one intermediate `cls_pw` or focal-loss variant, but keep
   `cls_pw=0.5` as the reference protocol.
5. Treat the current test numbers as a second estimate, not a final unbiased
   holdout, because both Stage 2A runs used the test split.
