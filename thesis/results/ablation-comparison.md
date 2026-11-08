---
title: Baseline vs cls_pw=0.0 Comparison
status: validated
date: 2026-11-08
sources:
  - thesis/results/baseline.md
  - thesis/results/ablation_cls_pw0.md
  - thesis/results/baseline-error-analysis.md
  - thesis/results/ablation_cls_pw0-error-analysis.md
  - runs/eval/baseline_yolov8n_seed42/metrics.json
  - runs/eval/ablation_cls_pw0_seed42/metrics.json
  - runs/eval/baseline_yolov8n_seed42/error_analysis/error_analysis.json
  - runs/eval/ablation_cls_pw0_seed42/error_analysis/error_analysis.json
---

# Baseline vs cls_pw=0.0 Comparison

- Baseline: `runs/eval/baseline_yolov8n_seed42` (`cls_pw=0.5`)
- Ablation: `runs/eval/ablation_cls_pw0_seed42` (`cls_pw=0.0`)
- Deltas are ablation − baseline. Positive means the ablation is higher.
- Same dataset, split, seed, and architecture; only class weighting differs.

Two metric families are reported separately:

1. **Run metrics** from `metrics.json` (Ultralytics validation output), used in the run results docs.
2. **Error-analysis metrics** at confidence 0.25 / IoU 0.5, used in the error-analysis docs.

Do not mix values across the two families.

## Overall run metrics

| Split | Metric | Baseline | Ablation | Delta |
|---|---|---:|---:|---:|
| val | Precision | 0.6912 | 0.6531 | -0.0381 |
| val | Recall | 0.4180 | 0.4429 | +0.0249 |
| val | mAP50 | 0.4861 | 0.4874 | +0.0012 |
| val | mAP50-95 | 0.2904 | 0.2907 | +0.0003 |
| test | Precision | 0.6801 | 0.7152 | +0.0351 |
| test | Recall | 0.4810 | 0.4655 | -0.0155 |
| test | mAP50 | 0.5428 | 0.5314 | -0.0114 |
| test | mAP50-95 | 0.3287 | 0.3254 | -0.0034 |

## Overall error-analysis metrics

Operating point: confidence 0.25, IoU 0.5, same-class matching.

| Split | Metric | Baseline | Ablation | Delta |
|---|---|---:|---:|---:|
| val | TP | 9,537 | 9,691 | +154 |
| val | FP | 3,669 | 3,707 | +38 |
| val | FN | 5,797 | 5,643 | -154 |
| val | Precision | 0.722 | 0.723 | +0.001 |
| val | Recall | 0.622 | 0.632 | +0.010 |
| val | F1 | 0.668 | 0.675 | +0.006 |
| test | TP | 10,265 | 10,375 | +110 |
| test | FP | 4,056 | 3,885 | -171 |
| test | FN | 5,965 | 5,855 | -110 |
| test | Precision | 0.717 | 0.728 | +0.011 |
| test | Recall | 0.632 | 0.639 | +0.007 |
| test | F1 | 0.672 | 0.681 | +0.009 |

## Per-class AP deltas

| Class | val ΔAP50 | val ΔAP50-95 | test ΔAP50 | test ΔAP50-95 |
|---|---:|---:|---:|---:|
| person | +0.0037 | -0.0002 | +0.0037 | +0.0003 |
| car | +0.0080 | +0.0037 | +0.0081 | +0.0046 |
| bus | +0.0214 | +0.0055 | -0.0114 | +0.0003 |
| bicycle | -0.0211 | -0.0102 | -0.0221 | -0.0122 |
| motorcycle | -0.0058 | +0.0030 | -0.0352 | -0.0097 |

## Per-condition mAP deltas

| Condition | val ΔmAP50 | val ΔmAP50-95 | test ΔmAP50 | test ΔmAP50-95 |
|---|---:|---:|---:|---:|
| fog | -0.0017 | -0.0015 | -0.0166 | -0.0084 |
| night | -0.0083 | +0.0025 | +0.0324 | +0.0038 |
| rain | +0.0028 | -0.0060 | -0.0044 | -0.0005 |
| snow | +0.0284 | +0.0088 | -0.0147 | -0.0029 |

## Per-source mAP deltas

| Source | val ΔmAP50 | val ΔmAP50-95 | test ΔmAP50 | test ΔmAP50-95 |
|---|---:|---:|---:|---:|
| ACDC | -0.0007 | +0.0016 | -0.0118 | -0.0025 |
| BDD100K | +0.0062 | +0.0031 | +0.0017 | +0.0003 |
| DAWN | -0.0068 | +0.0357 | +0.0371 | -0.0427 |
| RTTS | -0.0018 | -0.0007 | -0.0181 | -0.0092 |

## Per-size recall deltas

Operating point: confidence 0.25, IoU 0.5.

| Size | val ΔRecall | test ΔRecall |
|---|---:|---:|
| tiny (<0.001) | +0.0139 | +0.0056 |
| small (0.001–0.01) | +0.0101 | +0.0077 |
| medium (0.01–0.05) | +0.0038 | +0.0060 |
| large (>0.05) | +0.0054 | +0.0082 |

## Confusion and background false positives

Background false positives:

| Predicted class | Baseline | Ablation | Delta |
|---|---:|---:|---:|
| car | 6,240 | 6,062 | -178 |
| person | 920 | 972 | +52 |
| bus | 191 | 176 | -15 |
| bicycle | 67 | 68 | +1 |
| motorcycle | 48 | 59 | +11 |

Top off-diagonal matched confusions:

| GT class | Predicted class | Baseline | Ablation | Delta |
|---|---|---:|---:|---:|
| bus | car | 175 | 162 | -13 |
| car | bus | 92 | 94 | +2 |
| person | car | 21 | 15 | -6 |
| motorcycle | bicycle | 12 | 8 | -4 |
| bicycle | motorcycle | 11 | 12 | +1 |
| motorcycle | person | 10 | 10 | 0 |
| motorcycle | car | 8 | 9 | +1 |
| bicycle | person | 9 | 8 | -1 |

## Training and efficiency

| Item | Baseline | Ablation |
|---|---:|---:|
| Epochs completed | 100 | 100 |
| Best epoch (mAP50-95) | 97 | 86 |
| Training time | 4.49 h | 4.50 h |
| Parameters | 3.139 M | 3.139 M |
| GFLOPs at 640 | 8.754 | 8.754 |
| Inference latency | 3.936 ms/image | 3.650 ms/image |
| FPS | 254.09 | 273.99 |

## VRAM

| Measurement | Baseline | Ablation |
|---|---:|---:|
| Training PyTorch peak allocated | 683.8 MB | 683.8 MB |
| Training PyTorch peak reserved | 778.0 MB | 1218.0 MB |
| Training GPU used peak (`nvidia-smi`) | 949.0 MB | 1389.0 MB |
| Training GPU used baseline | 15.0 MB | 15.0 MB |
| Training GPU used peak delta | 934.0 MB | 1374.0 MB |
| Inference PyTorch peak allocated | 161.3 MB | 161.3 MB |

The baseline reserved/GPU-used values come from the 1-epoch calibration run
(`runs/train/vram_calibration/vram.json`); the ablation values come from the full
run (`runs/train/ablation_cls_pw0_seed42/vram.json`). The two are not directly
comparable. Inference latency/FPS and VRAM timing are single-run measurements of
the same architecture.

_Facts only; interpretation belongs in `thesis/findings/ablation-analysis.md`._
