---
title: Baseline Analysis — YOLOv8n
status: draft
date: 2026-11-08
sources:
  - thesis/results/baseline.md
  - thesis/results/baseline-error-analysis.md
  - thesis/results/ablation-comparison.md
  - thesis/findings/ablation-analysis.md
  - runs/eval/baseline_yolov8n_seed42/error_analysis/error_analysis.json
---

# Baseline Analysis — YOLOv8n

## Observation

The YOLOv8n baseline reaches val mAP50 0.486 / mAP50-95 0.290 and test
mAP50 0.543 / mAP50-95 0.329 after 100 epochs. Training is stable and does not
show severe overfitting. Fog is the easiest condition; night, rain, and snow are
clearly weaker. ACDC is the weakest source; RTTS and DAWN are stronger.

At the 0.25 confidence / 0.5 IoU operating point, test precision is 0.717 and
recall 0.632 (val 0.722 / 0.622). The gap between operating recall and the
0.001-confidence ceiling is substantial for night (0.600 vs 0.838), rain
(0.609 vs 0.816), and snow (0.629 vs 0.829), indicating that a meaningful share
of misses is confidence/localization failure rather than completely absent
evidence.

## Interpretation

1. **Small objects dominate the error budget.** Tiny objects (<0.1% relative
   area) are ~33% of test GT and have recall 0.334; small objects have recall
   0.708, while medium/large objects are ~0.89/0.93. Improving small-object
   recall is the single largest opportunity.
2. **Rare classes remain hard.** Bicycle recall is 0.35 test / 0.25 val and bus
   is 0.39 / 0.43; even the 0.001-confidence ceiling is only ~0.55–0.62 for
   bicycle and ~0.57–0.59 for bus. This is not just a threshold problem.
3. **Night is the hardest weather condition.** Night has the lowest recall
   (0.56 val, 0.60 test) and the largest gap to its ceiling. This supports the
   weather-adaptive design; a channel gate should help suppress unreliable
   channels and amplify stable semantics.
4. **ACDC exposes a domain/resolution gap.** ACDC val recall is 0.427 with a
   0.651 ceiling; test recall is 0.570 with a 0.775 ceiling. ACDC images are
   1920x1080 and many objects are small after downscaling to 640, so this is
   consistent with the size analysis.
5. **Class confusion is mostly vehicle-related.** Bus→car (175) and car→bus
   (92) dominate off-diagonal confusions; car also produces the most background
   false positives (6,240). Better context and multi-scale fusion are plausible
   mitigations.
6. **Person recall is moderate.** Person recall is ~0.50–0.52 with a ceiling of
   ~0.75–0.80, so many person instances are still missed or poorly localized.

## Limitations

- Single seed (42); differences of a few points are not yet statistically
  established.
- The baseline uses class weighting (`cls_pw=0.5`); the completed `cls_pw=0.0`
  ablation does not establish a clear class-weighting benefit
  ([ablation analysis](ablation-analysis.md)).
- Error analysis uses a fixed operating point (conf 0.25, IoU 0.5); the ceiling
  pass is diagnostic, not a deployable operating point.
- ACDC/DAWN/RTTS subset sizes are small (hundreds of images), so per-source
  numbers are noisy. DAWN val is especially small.
- Test and val differ in class/source composition, so their overall values
  should be read as two estimates, not as a train/test improvement.

## Implications for LiteViT-WG

- Target rain/snow/night and small/rare objects explicitly in the proposed
  architecture and experiments.
- Preserve P3 (high-resolution) for small objects; restrict global attention to
  P4/P5 as planned.
- Use the weather gate to address low-contrast/rain-streak channels, but evaluate
  it per condition, not only overall.
- Track per-class and per-size recall in every ablation; overall mAP can hide
  rare-class and small-object regressions.
- The completed `cls_pw=0.0` ablation does not show a clear class-weighting
  benefit; keep `cls_pw=0.5` and focus on architecture/weather adaptation
  ([ablation analysis](ablation-analysis.md)).

## Next checks

1. Use the completed `cls_pw=0.0` ablation comparison to constrain class-weighting
   claims ([ablation analysis](ablation-analysis.md)).
2. Add multi-seed runs for the final baseline and proposed model.
3. Inspect the qualitative FN/FP overlays for night/rain/snow and ACDC.
4. Proceed to the LiteViT/weather-gate technical spike and memory validation.
