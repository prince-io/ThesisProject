---
title: Weather Gate Analysis — Gate-Only Result
status: draft
date: 2026-11-10
sources:
  - thesis/results/weather_gate.md
  - thesis/results/weather_gate-error-analysis.md
  - thesis/results/architecture-comparison.md
---

# Weather Gate Analysis — Gate-Only Result

## Observation

Gate-only is a small positive result on test ([weather gate results](../results/weather_gate.md)):

- test mAP50 0.5444 vs baseline 0.5428 (+0.16 pp)
- test mAP50-95 0.3301 vs baseline 0.3287 (+0.14 pp)
- test precision 0.6857 vs 0.6801; recall 0.4965 vs 0.4810

Validation is slightly lower: 0.4824 / 0.2891 vs 0.4861 / 0.2904. The run completed
all 100 epochs without overfitting; best mAP50-95 was epoch 90 (0.29037) and best
mAP50 was epoch 100 (0.4844).

The gains are condition- and class-specific:

- night test mAP50 +4.13 pp, mAP50-95 +1.44 pp
- rain test mAP50 +2.19 pp, mAP50-95 +1.30 pp
- fog and snow are roughly neutral or slightly lower
- bicycle test AP50 +3.17 pp; bus +0.55 pp; motorcycle −3.00 pp
- DAWN test mAP50 +6.87 pp; ACDC +1.36 pp; BDD +1.02 pp; RTTS −0.22 pp

At confidence 0.25, the gate raises test recall slightly and lowers precision
slightly ([weather gate error analysis](../results/weather_gate-error-analysis.md)).
Tiny-object recall is 0.336 vs 0.334; bus recall 0.403 vs 0.387; bicycle recall
0.389 vs 0.352. Background car and person false positives increase.

## Interpretation

1. **The gate is safe but not a large win.** The overall test difference is within
   single-seed noise. It is not enough to claim a robust improvement.
2. **The direction is plausible for adverse weather.** Night and rain improve most,
   and rare-class recall rises. This is consistent with channel reweighting
   suppressing degraded features.
3. **There is a precision/recall trade-off.** The gate catches slightly more objects
   but also produces more background false positives; mAP changes only marginally.
4. **Snow and motorcycle do not improve.** The gate is not uniformly beneficial, so
   per-condition and per-class reporting remains necessary.
5. **Gate-only is a better fallback than V1 attention.** It is stable, tiny, and
   approximately baseline-level, whereas V1 overfit and hurt small/rare objects.

## Limitations

- Single seed (42); the overall gain is too small to separate from run variance.
- Test split has been reused across Stage 2A/2B experiments.
- Per-source subsets are small, so source-level deltas are noisy.
- The gate values were fixed a priori; no reduction-ratio or placement ablation.
- No gate-weight diagnostics per weather condition were collected.
- The test gain is smaller than per-condition noise in several subsets.

## Implications for the next experiment

- Use gate-only as the safe base for **A+B**: add a regularized, lower-capacity P5
  attention block on top of the gate.
- Do not add P4 attention or more gate capacity until A+B is measured.
- If A+B does not beat gate-only and baseline on validation, keep gate-only as the
  final proposed variant and report P5 attention as a negative ablation.
- Multi-seed validation is required before making a final weather-robustness claim.

## Next checks

1. Implement A+B with DropPath 0.2, dropout 0.1, and MLP ratio 1.0.
2. Evaluate A+B per condition, class, and size against both baseline and gate-only.
3. Log mean gate weights per condition if time permits.
4. Run at least one additional seed for the final chosen model.

## Status

Draft. Based on the completed `EXP-0004` run and evaluation.
