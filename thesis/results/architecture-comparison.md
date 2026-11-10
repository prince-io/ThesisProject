---
title: Architecture Comparison — Baseline, V1, Gate-Only
status: validated
date: 2026-11-10
sources:
  - thesis/results/baseline.md
  - thesis/results/litevit_v1_p5.md
  - thesis/results/weather_gate.md
  - thesis/results/baseline-error-analysis.md
  - thesis/results/litevit_v1_p5-error-analysis.md
  - thesis/results/weather_gate-error-analysis.md
  - runs/eval/baseline_yolov8n_seed42/metrics.json
  - runs/eval/litevit_v1_p5_seed42_v2/metrics.json
  - runs/eval/weather_gate_seed42/metrics.json
---

# Architecture Comparison — Baseline, V1, Gate-Only

All variants use the same dataset, split, seed, and training protocol.
V1 is the early-stopped best checkpoint (epoch 18). Gate-only and baseline use
their best checkpoints.

## Overall run metrics

| Split | Metric | Baseline | V1 | Gate-only |
|---|---|---:|---:|---:|
| val | Precision | 0.6912 | 0.6108 | 0.6456 |
| val | Recall | 0.4180 | 0.3795 | 0.4421 |
| val | mAP50 | 0.4861 | 0.4231 | 0.4824 |
| val | mAP50-95 | 0.2904 | 0.2499 | 0.2891 |
| test | Precision | 0.6801 | 0.6359 | 0.6857 |
| test | Recall | 0.4810 | 0.4383 | 0.4965 |
| test | mAP50 | 0.5428 | 0.4888 | 0.5444 |
| test | mAP50-95 | 0.3287 | 0.2895 | 0.3301 |

## Per-condition test mAP

| Condition | Baseline mAP50 | V1 mAP50 | Gate mAP50 | Baseline mAP50-95 | V1 mAP50-95 | Gate mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| fog | 0.6804 | 0.6180 | 0.6773 | 0.4441 | 0.3904 | 0.4454 |
| night | 0.4182 | 0.3961 | 0.4595 | 0.2276 | 0.2061 | 0.2420 |
| rain | 0.3950 | 0.3419 | 0.4169 | 0.2275 | 0.1976 | 0.2405 |
| snow | 0.3923 | 0.4232 | 0.3898 | 0.2188 | 0.2016 | 0.2130 |

## Per-class test AP

| Class | Baseline AP50 | V1 AP50 | Gate AP50 | Baseline AP50-95 | V1 AP50-95 | Gate AP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| person | 0.6102 | 0.5689 | 0.6124 | 0.3490 | 0.3245 | 0.3480 |
| car | 0.7189 | 0.6828 | 0.7171 | 0.4357 | 0.4094 | 0.4357 |
| bus | 0.4109 | 0.3395 | 0.4164 | 0.2779 | 0.2313 | 0.2754 |
| bicycle | 0.3809 | 0.2972 | 0.4126 | 0.2227 | 0.1780 | 0.2360 |
| motorcycle | 0.5933 | 0.5554 | 0.5633 | 0.3583 | 0.3044 | 0.3556 |

## Per-source test mAP

| Source | Baseline mAP50 | V1 mAP50 | Gate mAP50 | Baseline mAP50-95 | V1 mAP50-95 | Gate mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| ACDC | 0.3122 | 0.2894 | 0.3258 | 0.1721 | 0.1456 | 0.1753 |
| BDD100K | 0.4225 | 0.3824 | 0.4327 | 0.2400 | 0.2174 | 0.2448 |
| DAWN | 0.6846 | 0.6177 | 0.7533 | 0.4391 | 0.3858 | 0.4578 |
| RTTS | 0.6878 | 0.6295 | 0.6856 | 0.4475 | 0.3951 | 0.4487 |

## Test recall by object size

| Size | Baseline | V1 | Gate-only |
|---|---:|---:|---:|
| tiny (<0.001) | 0.3339 | 0.2695 | 0.3358 |
| small (0.001–0.01) | 0.7084 | 0.6811 | 0.7091 |
| medium (0.01–0.05) | 0.8920 | 0.8775 | 0.8983 |
| large (>0.05) | 0.9302 | 0.9260 | 0.9376 |

## Training and efficiency

| Item | Baseline | V1 | Gate-only |
|---|---:|---:|---:|
| Epochs completed | 100 | 38 (early stop) | 100 |
| Best epoch (mAP50-95) | 97 | 18 | 90 |
| Best epoch (mAP50) | 91 | 16 | 100 |
| Training time | 4.49 h | 1.92 h | 4.68 h |
| Parameters | 3.139 M | 3.674 M | 3.183 M |
| GFLOPs at 640 | 8.754 | 9.206 | 8.789 |
| Inference latency | 3.936 ms | 3.928 ms | 4.060 ms |
| FPS | 254.09 | 254.61 | 246.32 |
| Inference PyTorch peak | 161.3 MB | 207.0 MB | 162.0 MB |

## Training VRAM

| Measurement | Baseline | V1 | Gate-only |
|---|---:|---:|---:|
| PyTorch peak allocated | 683.8 MB | 729.5 MB | 697.3 MB |
| PyTorch peak reserved | 778.0 MB | 1152.0 MB | 1246.0 MB |
| GPU used peak | 949.0 MB | 1355.0 MB | 1417.0 MB |

The baseline reserved/GPU-used values come from the 1-epoch calibration run; V1 and
gate-only values come from their full runs.

_Facts only; interpretation belongs in `thesis/findings/`._
