---
title: A+B Comparison — Baseline, Gate-Only, V1, A+B
status: validated
date: 2026-11-10
sources:
  - thesis/results/baseline.md
  - thesis/results/weather_gate.md
  - thesis/results/litevit_v1_p5.md
  - thesis/results/gate_p5.md
  - runs/eval/baseline_yolov8n_seed42/metrics.json
  - runs/eval/weather_gate_seed42/metrics.json
  - runs/eval/litevit_v1_p5_seed42_v2/metrics.json
  - runs/eval/gate_p5_seed42/metrics.json
---

# A+B Comparison — Baseline, Gate-Only, V1, A+B

A+B uses its best checkpoint (epoch 66). V1 uses its early-stopped best (epoch 18).

## Overall

| Split | Metric | Baseline | Gate-only | V1 | A+B |
|---|---|---:|---:|---:|---:|
| val | mAP50 | 0.4861 | 0.4824 | 0.4231 | **0.4865** |
| val | mAP50-95 | 0.2904 | 0.2891 | 0.2499 | **0.2938** |
| test | mAP50 | 0.5428 | **0.5444** | 0.4888 | 0.5387 |
| test | mAP50-95 | 0.3287 | **0.3301** | 0.2895 | 0.3290 |

## Per-condition test

| Condition | Baseline mAP50 | Gate mAP50 | V1 mAP50 | A+B mAP50 | Baseline mAP50-95 | Gate mAP50-95 | V1 mAP50-95 | A+B mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fog | **0.6804** | 0.6773 | 0.6180 | 0.6708 | 0.4441 | **0.4454** | 0.3904 | 0.4406 |
| night | 0.4182 | **0.4595** | 0.3961 | 0.4174 | 0.2276 | **0.2420** | 0.2061 | 0.2255 |
| rain | 0.3950 | **0.4169** | 0.3419 | 0.3994 | 0.2275 | **0.2405** | 0.1976 | 0.2319 |
| snow | 0.3923 | 0.3898 | **0.4232** | 0.3963 | 0.2188 | 0.2130 | 0.2016 | **0.2224** |

## Per-class test

| Class | Baseline AP50 | Gate AP50 | V1 AP50 | A+B AP50 | Baseline AP50-95 | Gate AP50-95 | V1 AP50-95 | A+B AP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| person | 0.6102 | **0.6124** | 0.5689 | 0.6083 | 0.3490 | 0.3480 | 0.3245 | 0.3482 |
| car | **0.7189** | 0.7171 | 0.6828 | 0.7156 | **0.4357** | **0.4357** | 0.4094 | 0.4352 |
| bus | 0.4109 | **0.4164** | 0.3395 | 0.4154 | 0.2779 | 0.2754 | 0.2313 | **0.2840** |
| bicycle | 0.3809 | **0.4126** | 0.2972 | 0.3832 | 0.2227 | **0.2360** | 0.1780 | 0.2277 |
| motorcycle | **0.5933** | 0.5633 | 0.5554 | 0.5709 | **0.3583** | 0.3556 | 0.3044 | 0.3501 |

## Test recall by size

| Size | Baseline | Gate-only | V1 | A+B |
|---|---:|---:|---:|---:|
| tiny (<0.001) | 0.3339 | **0.3358** | 0.2695 | 0.3245 |
| small (0.001–0.01) | 0.7084 | **0.7091** | 0.6811 | 0.7030 |
| medium (0.01–0.05) | 0.8920 | **0.8983** | 0.8775 | 0.8957 |
| large (>0.05) | 0.9302 | **0.9376** | 0.9260 | 0.9367 |

## Training and efficiency

| Item | Baseline | Gate-only | V1 | A+B |
|---|---:|---:|---:|---:|
| Epochs completed | 100 | 100 | 38 | 86 |
| Best epoch (mAP50-95) | 97 | 90 | 18 | 66 |
| Parameters | 3.139 M | 3.183 M | 3.674 M | 3.584 M |
| GFLOPs at 640 | 8.754 | 8.789 | 9.206 | 9.099 |
| Inference latency | 3.936 ms | 4.060 ms | 3.928 ms | 3.939 ms |
| FPS | 254.09 | 246.32 | 254.61 | 253.85 |

## Epoch-86 A+B check

The last A+B checkpoint (epoch 86) scored test mAP50 0.5414 / mAP50-95 0.3307,
still within noise of epoch 66 and gate-only. It did not justify a 100-epoch rerun.

_Facts only; interpretation belongs in `thesis/findings/gate-p5-analysis.md`._
