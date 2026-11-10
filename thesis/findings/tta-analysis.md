---
title: TTA Analysis — Gate-Only
status: draft
date: 2026-11-10
sources:
  - thesis/results/weather-gate-tta.md
  - thesis/results/weather_gate.md
---

# TTA Analysis — Gate-Only

## Observation

TTA reduced gate-only accuracy:

- test mAP50: 0.5444 → 0.5329
- test mAP50-95: 0.3301 → 0.3280
- val mAP50: 0.4824 → 0.4755
- val mAP50-95: 0.2891 → 0.2874

Inference cost roughly doubled: 4.06 ms → 9.20 ms per image, 246 → 109 FPS.
Fog, night, and rain all became worse; only snow mAP50-95 improved slightly
([TTA facts](../results/weather-gate-tta.md)).

## Interpretation

- TTA is not useful for this detector on this benchmark. Flip/multi-scale
  averaging changes the confidence ranking in an unfavourable direction.
- The lost recall (0.4965 → 0.4753) is larger than the precision change, so TTA is
  not a good trade even before considering the 2× latency.
- TTA should not be included in the final model or its headline numbers.

## Limitations

- Single seed and one TTA configuration; other transforms were not tested.
- TTA was applied to gate-only, not every variant.
- Test split reuse applies.

## Implication

Exclude TTA from the final model. Model soup may still be tested later as a
post-training trick, but it is separate from TTA and not required.
