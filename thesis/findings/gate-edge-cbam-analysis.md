---
title: Kitchen-Sink Analysis — Gate + Edge + Spatial CBAM
status: draft
date: 2026-11-11
sources:
  - thesis/results/gate_edge_cbam.md
  - thesis/results/gate_edge_cbam-error-analysis.md
  - thesis/results/gate-p5-comparison.md
---

# Kitchen-Sink Analysis — Gate + Edge + Spatial CBAM

## Observation

The kitchen-sink model is neutral on test ([kitchen-sink results](../results/gate_edge_cbam.md)):

- test mAP50 **0.5420** vs baseline 0.5428 and gate-only 0.5444
- test mAP50-95 **0.3295** vs baseline 0.3287 and gate-only 0.3301
- validation mAP50 / mAP50-95: 0.4859 / 0.2910

It is effectively tied with the baseline and slightly below gate-only.

The model has a mixed class and weather profile:

- **Fog:** best of the proposed models, AP50 0.6817 / AP50-95 0.4489.
- **Night:** AP50 0.4090 / AP50-95 0.2212, below gate-only 0.4595 / 0.2420.
- **Rain:** AP50 0.4004 / AP50-95 0.2270, below gate-only 0.4169 / 0.2405.
- **Snow:** AP50 0.3846; AP50-95 0.2201, slightly above baseline and gate-only.
- **Bicycle:** AP50 0.3856 / AP50-95 0.2274, below gate-only 0.4126 / 0.2360.
- **Motorcycle:** AP50 0.5800 / AP50-95 0.3631, above gate-only.

The model is stable for 100 epochs: 3.204 M parameters, 8.895 GFLOPs, 4.17 ms,
239.6 FPS. Training took 5.57 h with a 1,467 MB GPU-used peak.

## Interpretation

Adding Sobel edge residuals and spatial CBAM to gate-only does not improve overall
detection. The fog gain does not compensate for the night/rain and bicycle losses.
The three mechanisms are not additive in this configuration.

The result supports the earlier pattern: extra neck modules do not beat the simple
channel gate. The kitchen-sink model is useful as a negative/neutral ablation, not
as a final model.

## Limitations

The combined run cannot separate the effects of edge enhancement and spatial CBAM;
edge-only and CBAM-only ablations were not run. The result is single-seed.

## Conclusion

Gate-only remains better than the kitchen sink. The kitchen-sink model should be
reported as a neutral ablation that did not improve the detector.

See [kitchen-sink facts](../results/gate_edge_cbam.md) and
[error analysis](../results/gate_edge_cbam-error-analysis.md).
