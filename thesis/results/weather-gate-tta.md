---
title: Weather Gate TTA Check
status: validated
date: 2026-11-10
sources:
  - runs/eval/weather_gate_seed42/metrics.json
  - runs/eval/weather_gate_seed42_tta/metrics.json
---

# Weather Gate TTA Check

- Weights: `runs/train/weather_gate_seed42/weights/best.pt`
- Normal evaluation: `runs/eval/weather_gate_seed42`
- TTA evaluation: `runs/eval/weather_gate_seed42_tta`
- TTA was enabled with Ultralytics validation augmentation.

## Overall

| Split | Metric | Normal | TTA |
|---|---|---:|---:|
| val | Precision | 0.6456 | 0.6729 |
| val | Recall | 0.4421 | 0.4193 |
| val | mAP50 | 0.4824 | 0.4755 |
| val | mAP50-95 | 0.2891 | 0.2874 |
| test | Precision | 0.6857 | 0.6836 |
| test | Recall | 0.4965 | 0.4753 |
| test | mAP50 | 0.5444 | 0.5329 |
| test | mAP50-95 | 0.3301 | 0.3280 |

## Per-condition test

| Condition | Normal mAP50 | TTA mAP50 | Normal mAP50-95 | TTA mAP50-95 |
|---|---:|---:|---:|---:|
| fog | 0.6773 | 0.6677 | 0.4454 | 0.4385 |
| night | 0.4595 | 0.4129 | 0.2420 | 0.2378 |
| rain | 0.4169 | 0.3990 | 0.2405 | 0.2359 |
| snow | 0.3898 | 0.3883 | 0.2130 | 0.2190 |

## Efficiency

| Item | Normal | TTA |
|---|---:|---:|
| Inference latency | 4.060 ms | 9.201 ms |
| FPS | 246.32 | 108.69 |
| Parameters | 3.183 M | 3.183 M |
| GFLOPs | 8.789 | 8.789 |

_Facts only; interpretation belongs in `thesis/findings/tta-analysis.md`._
