---
title: P2 Comparison — Baseline, Gate-Only, P2
status: validated
date: 2026-11-11
sources:
  - thesis/results/baseline.md
  - thesis/results/weather_gate.md
  - thesis/results/gate_p2.md
  - thesis/results/gate_p2-error-analysis.md
  - runs/eval/baseline_yolov8n_seed42/metrics.json
  - runs/eval/weather_gate_seed42/metrics.json
  - runs/eval/gate_p2_seed42/metrics.json
---

# P2 Comparison — Baseline, Gate-Only, P2

All variants use the same dataset, split, seed, and training protocol. Values are
test metrics unless marked val. Deltas are percentage points (pp).

## Overall

| Split | Metric | Baseline | Gate-only | P2 |
|---|---|---:|---:|---:|
| val | Precision | 0.6912 | 0.6456 | 0.6537 |
| val | Recall | 0.4180 | 0.4421 | 0.4275 |
| val | mAP50 | **0.4861** | 0.4824 | 0.4850 |
| val | mAP50-95 | 0.2904 | 0.2891 | **0.2910** |
| test | Precision | 0.6801 | 0.6857 | **0.7063** |
| test | Recall | 0.4810 | 0.4965 | **0.4973** |
| test | mAP50 | 0.5428 | 0.5444 | **0.5571** |
| test | mAP50-95 | 0.3287 | 0.3301 | **0.3351** |

P2 vs baseline: **+1.43 pp mAP50, +0.64 pp mAP50-95**.
P2 vs gate-only: **+1.27 pp mAP50, +0.50 pp mAP50-95**.

## Per class — test AP50

| Class | GT | Baseline | Gate-only | P2 | P2 − Base | P2 − Gate |
|---|---:|---:|---:|---:|---:|---:|
| person | 2,362 | 0.6102 | 0.6124 | **0.6161** | +0.59 pp | +0.37 pp |
| car | 13,173 | 0.7189 | 0.7171 | **0.7393** | +2.04 pp | +2.22 pp |
| bus | 377 | 0.4109 | **0.4164** | 0.4028 | −0.81 pp | −1.36 pp |
| bicycle | 162 | 0.3809 | **0.4126** | 0.4077 | +2.68 pp | −0.49 pp |
| motorcycle | 156 | 0.5933 | 0.5633 | **0.6195** | +2.62 pp | +5.62 pp |

## Per class — test AP50-95

| Class | GT | Baseline | Gate-only | P2 | P2 − Base | P2 − Gate |
|---|---:|---:|---:|---:|---:|---:|
| person | 2,362 | 0.3490 | 0.3480 | **0.3540** | +0.50 pp | +0.60 pp |
| car | 13,173 | 0.4357 | 0.4357 | **0.4527** | +1.70 pp | +1.70 pp |
| bus | 377 | **0.2779** | 0.2754 | 0.2728 | −0.51 pp | −0.26 pp |
| bicycle | 162 | 0.2227 | **0.2360** | 0.2341 | +1.14 pp | −0.19 pp |
| motorcycle | 156 | 0.3583 | 0.3556 | **0.3618** | +0.35 pp | +0.62 pp |

## Per condition — test AP50

| Condition | GT | Baseline | Gate-only | P2 | P2 − Base | P2 − Gate |
|---|---:|---:|---:|---:|---:|---:|
| fog | 3,586 | 0.6804 | 0.6773 | **0.6906** | +1.02 pp | +1.33 pp |
| rain | 4,113 | 0.3950 | **0.4169** | 0.4043 | +0.93 pp | −1.26 pp |
| snow | 4,476 | 0.3923 | 0.3898 | **0.4606** | +6.83 pp | +7.08 pp |
| night | 4,055 | 0.4182 | **0.4595** | 0.4471 | +2.89 pp | −1.24 pp |

## Per condition — test AP50-95

| Condition | GT | Baseline | Gate-only | P2 | P2 − Base | P2 − Gate |
|---|---:|---:|---:|---:|---:|---:|
| fog | 3,586 | 0.4441 | **0.4454** | 0.4441 | 0.00 pp | −0.13 pp |
| rain | 4,113 | 0.2275 | **0.2405** | 0.2329 | +0.54 pp | −0.76 pp |
| snow | 4,476 | 0.2188 | 0.2130 | **0.2345** | +1.57 pp | +2.15 pp |
| night | 4,055 | 0.2276 | 0.2420 | **0.2431** | +1.55 pp | +0.11 pp |

## Test recall by object size

| Size | Baseline | Gate-only | P2 |
|---|---:|---:|---:|
| tiny (<0.001) | 0.3339 | 0.3358 | **0.3410** |
| small (0.001–0.01) | **0.7084** | **0.7091** | 0.7000 |
| medium (0.01–0.05) | 0.8920 | **0.8983** | 0.8931 |
| large (>0.05) | 0.9302 | **0.9376** | 0.9302 |

## Per-source test mAP

| Source | Metric | Baseline | Gate-only | P2 |
|---|---|---:|---:|---:|
| ACDC | mAP50 | 0.3122 | 0.3258 | **0.3419** |
| ACDC | mAP50-95 | 0.1721 | 0.1753 | **0.1870** |
| BDD100K | mAP50 | 0.4225 | 0.4327 | **0.4357** |
| BDD100K | mAP50-95 | 0.2400 | 0.2448 | **0.2490** |
| DAWN | mAP50 | 0.6846 | **0.7533** | 0.7126 |
| DAWN | mAP50-95 | 0.4391 | **0.4578** | 0.4238 |
| RTTS | mAP50 | 0.6878 | 0.6856 | **0.6989** |
| RTTS | mAP50-95 | 0.4475 | **0.4487** | 0.4477 |

## Training and efficiency

| Item | Baseline | Gate-only | P2 |
|---|---:|---:|---:|
| Epochs completed | 100 | 100 | 82 (early stop) |
| Best epoch (mAP50-95) | 97 | 90 | 62 |
| Best epoch (mAP50) | 91 | 100 | 70 |
| Training time | 4.49 h | 4.68 h | 6.01 h |
| Parameters | 3.139 M | 3.183 M | 3.270 M |
| GFLOPs at 640 | 8.754 | 8.789 | 13.281 |
| Inference latency | 3.936 ms | 4.060 ms | 6.381 ms |
| FPS | 254.09 | 246.32 | 156.72 |
| Inference PyTorch peak | 161.3 MB | 162.0 MB | 270.9 MB |

## Training VRAM

| Measurement | Baseline | Gate-only | P2 |
|---|---:|---:|---:|
| PyTorch peak allocated | 683.8 MB | 697.3 MB | 1433.3 MB |
| PyTorch peak reserved | 778.0 MB | 1246.0 MB | 3106.0 MB |
| GPU used peak | 949.0 MB | 1417.0 MB | 3277.0 MB |

Baseline VRAM is from the 1-epoch calibration run; gate-only and P2 values are from
their full runs.

_Facts only; interpretation belongs in `thesis/findings/gate-p2-analysis.md`._
