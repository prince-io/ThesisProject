---
title: Ablation Results — cls_pw=0.0
status: validated
date: 2026-11-08
sources:
  - runs/train/ablation_cls_pw0_seed42
  - runs/eval/ablation_cls_pw0_seed42
---

# Ablation Results — cls_pw=0.0

- Train run: `runs/train/ablation_cls_pw0_seed42`
- Evaluation run: `runs/eval/ablation_cls_pw0_seed42`
- Weights: `runs/train/ablation_cls_pw0_seed42/weights/best.pt`
- Seed: 42
- Class weighting power (`cls_pw`): 0.0
- Dataset checksum: `7de437da75a1698d73195565cfff0c2b3e83da48`

## Overall

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| val | 0.6531 | 0.4429 | 0.4874 | 0.2907 |
| test | 0.7152 | 0.4655 | 0.5314 | 0.3254 |

## Per class

| Class | val AP50 | val AP50-95 | test AP50 | test AP50-95 |
|---|---:|---:|---:|---:|
| bicycle | 0.2916 | 0.1672 | 0.3588 | 0.2105 |
| bus | 0.4693 | 0.3191 | 0.3995 | 0.2783 |
| car | 0.7264 | 0.4446 | 0.7270 | 0.4403 |
| motorcycle | 0.3754 | 0.2091 | 0.5581 | 0.3486 |
| person | 0.5742 | 0.3137 | 0.6139 | 0.3493 |

## Per condition

| Name | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 |
|---|---:|---:|---:|---:|
| fog | 0.6871 | 0.4488 | 0.6638 | 0.4357 |
| night | 0.3333 | 0.1730 | 0.4507 | 0.2315 |
| rain | 0.4700 | 0.2773 | 0.3906 | 0.2270 |
| snow | 0.3931 | 0.2200 | 0.3776 | 0.2159 |

## Per source

| Name | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 |
|---|---:|---:|---:|---:|
| acdc | 0.2897 | 0.1590 | 0.3004 | 0.1695 |
| bdd | 0.4208 | 0.2345 | 0.4242 | 0.2403 |
| dawn | 0.8524 | 0.5134 | 0.7217 | 0.3964 |
| rtts | 0.6759 | 0.4480 | 0.6697 | 0.4382 |

## Training

- Epochs completed: 100
- Best epoch (mAP50-95): 86
- Best val mAP50 / mAP50-95: 0.4874 / 0.2914
- Total training time: 4.50 h

## Efficiency

- Parameters: 3.139 M
- GFLOPs at 640: 8.754
- Inference latency: 3.650 ms/image
- FPS: 273.99

### VRAM

| Measurement | Value |
|---|---:|
| Training PyTorch peak allocated | 683.8 MB |
| Training PyTorch peak reserved | 1218.0 MB |
| Training GPU used peak (nvidia-smi) | 1389.0 MB |
| Training GPU used baseline | 15.0 MB |
| Training GPU used peak delta | 1374.0 MB |
| Inference PyTorch peak allocated | 161.3 MB |

_GPU-used and reserved values were recorded during the full training run._

## Figures

- [test_BoxF1_curve.png](figures/ablation_cls_pw0/test_BoxF1_curve.png)
- [test_BoxPR_curve.png](figures/ablation_cls_pw0/test_BoxPR_curve.png)
- [test_BoxP_curve.png](figures/ablation_cls_pw0/test_BoxP_curve.png)
- [test_BoxR_curve.png](figures/ablation_cls_pw0/test_BoxR_curve.png)
- [test_confusion_matrix.png](figures/ablation_cls_pw0/test_confusion_matrix.png)
- [test_confusion_matrix_normalized.png](figures/ablation_cls_pw0/test_confusion_matrix_normalized.png)
- [val_BoxF1_curve.png](figures/ablation_cls_pw0/val_BoxF1_curve.png)
- [val_BoxPR_curve.png](figures/ablation_cls_pw0/val_BoxPR_curve.png)
- [val_BoxP_curve.png](figures/ablation_cls_pw0/val_BoxP_curve.png)
- [val_BoxR_curve.png](figures/ablation_cls_pw0/val_BoxR_curve.png)
- [val_confusion_matrix.png](figures/ablation_cls_pw0/val_confusion_matrix.png)
- [val_confusion_matrix_normalized.png](figures/ablation_cls_pw0/val_confusion_matrix_normalized.png)

## Artifacts

- Training directory: `runs/train/ablation_cls_pw0_seed42`
- Evaluation directory: `runs/eval/ablation_cls_pw0_seed42`
- Run config: `runs/train/ablation_cls_pw0_seed42/resolved_config.yaml`
- Error analysis: `runs/eval/ablation_cls_pw0_seed42/error_analysis/`
- Error-analysis facts: `thesis/results/ablation_cls_pw0-error-analysis.md`
- Baseline comparison: `thesis/results/ablation-comparison.md`
- Interpretation: `thesis/findings/ablation-analysis.md`
- Git commit: `4d9003375a36fed0376c937b639e969bd9bc1470`

_Facts only; interpretation belongs in `thesis/findings/`._
