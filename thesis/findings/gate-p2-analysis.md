---
title: P2 Analysis — Gate-Only + High-Resolution Head
status: draft
date: 2026-11-11
sources:
  - thesis/results/gate_p2.md
  - thesis/results/gate_p2-error-analysis.md
  - thesis/results/gate-p2-comparison.md
---

# P2 Analysis — Gate-Only + High-Resolution Head

## Observation

P2 is the best measured model on test ([P2 results](../results/gate_p2.md)):

- test mAP50 **0.5571** vs baseline 0.5428 and gate-only 0.5444
- test mAP50-95 **0.3351** vs baseline 0.3287 and gate-only 0.3301
- precision **0.7063**; recall **0.4973**
- validation mAP50 / mAP50-95: 0.4850 / 0.2910, close to gate-only 0.4824 / 0.2891

The gain over gate-only is **+1.27 pp mAP50** and **+0.50 pp mAP50-95**. The
success criteria were met on test.

The gain is not uniform ([comparison](../results/gate-p2-comparison.md)):

- **Classes:** car, motorcycle, and person improve; bus and bicycle AP50 are lower.
  Motorcycle AP50 rises from 0.5633 to 0.6195. Car AP50 rises from 0.7171 to 0.7393.
- **Weather:** snow is the main gain, +7.08 pp AP50 over gate-only. Fog gains
  +1.33 pp AP50. Rain loses 1.26 pp AP50 and 0.76 pp AP50-95. Night loses 1.24 pp
  AP50 but gains 0.11 pp AP50-95.
- **Size:** tiny recall is 0.3410 vs gate-only 0.3358, a small gain; small-object
  recall falls from 0.7091 to 0.7000.
- **Sources:** ACDC and BDD100K improve; DAWN drops from 0.7533 to 0.7126 mAP50;
  RTTS AP50 improves slightly.

The cost is substantial: 13.281 GFLOPs, 6.381 ms, and 156.72 FPS vs gate-only
8.789 GFLOPs, 4.060 ms, and 246.32 FPS. Parameters rise only from 3.183 M to
3.270 M.

## Interpretation

The P2 head adds a stride-4 detection level, so the hypothesis was that tiny
objects would improve. The result only partially supports that: overall detection
improves, but tiny recall moves only slightly. The larger gains are in car,
motorcycle, snow, and fog, which suggests the extra high-resolution level helps
localization and coverage for some small/medium objects rather than solving the
tiny-object bottleneck generally.

P2 is better than gate-only overall, but it is not a uniform all-weather
improvement. Gate-only still wins rain, night AP50, bicycle, and bus. The overall
gain is therefore a trade between the snow/car/motorcycle gains and the
rain/night/bus losses.

## Limitations

The result is from a single seed; the overall difference from gate-only is small
and should not be overinterpreted.

## Conclusion

P2 is the best measured model and should be the primary result. It is a modest
accuracy gain at a clear compute cost, not a new solution to tiny objects or
night/rain. No further architecture modules are justified by this result.

See [P2 facts](../results/gate_p2.md),
[P2 error analysis](../results/gate_p2-error-analysis.md), and
[P2 comparison](../results/gate-p2-comparison.md).
