---
title: Baseline Results — YOLOv8n
status: validated
date: 2026-11-07
sources:
  - runs/train/baseline_yolov8n_seed42
  - runs/eval/baseline_yolov8n_seed42
---

# Baseline Results — YOLOv8n

- Train run: `runs/train/baseline_yolov8n_seed42`
- Evaluation run: `runs/eval/baseline_yolov8n_seed42`
- Weights: `runs/train/baseline_yolov8n_seed42/weights/best.pt`
- Seed: 42
- Class weighting power (`cls_pw`): 0.5
- Dataset checksum: `7de437da75a1698d73195565cfff0c2b3e83da48`

## Overall

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| val | 0.6912 | 0.4180 | 0.4861 | 0.2904 |
| test | 0.6801 | 0.4810 | 0.5428 | 0.3287 |

## Per class

| Class | val AP50 | val AP50-95 | test AP50 | test AP50-95 |
|---|---:|---:|---:|---:|
| bicycle | 0.3127 | 0.1774 | 0.3809 | 0.2227 |
| bus | 0.4479 | 0.3136 | 0.4109 | 0.2779 |
| car | 0.7184 | 0.4409 | 0.7189 | 0.4357 |
| motorcycle | 0.3812 | 0.2062 | 0.5933 | 0.3583 |
| person | 0.5705 | 0.3140 | 0.6102 | 0.3490 |

## Per condition

| Name | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 |
|---|---:|---:|---:|---:|
| fog | 0.6887 | 0.4503 | 0.6804 | 0.4441 |
| night | 0.3416 | 0.1705 | 0.4182 | 0.2276 |
| rain | 0.4672 | 0.2832 | 0.3950 | 0.2275 |
| snow | 0.3647 | 0.2112 | 0.3923 | 0.2188 |

## Per source

| Name | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 |
|---|---:|---:|---:|---:|
| acdc | 0.2904 | 0.1575 | 0.3122 | 0.1721 |
| bdd | 0.4146 | 0.2314 | 0.4225 | 0.2400 |
| dawn | 0.8591 | 0.4777 | 0.6846 | 0.4391 |
| rtts | 0.6778 | 0.4487 | 0.6878 | 0.4475 |

## Training

- Epochs completed: 100
- Best epoch (mAP50-95): 97
- Best val mAP50 / mAP50-95: 0.4860 / 0.2902
- Total training time: 4.49 h

## Efficiency

- Parameters: 3.139 M
- GFLOPs at 640: 8.754
- Inference latency: 3.936 ms/image
- FPS: 254.09

### VRAM

| Measurement | Value |
|---|---:|
| Training PyTorch peak allocated | 683.8 MB |
| Training PyTorch peak reserved | 778.0 MB |
| Training GPU used peak (nvidia-smi) | 949.0 MB |
| Training GPU used baseline | 15.0 MB |
| Training GPU used peak delta | 934.0 MB |
| Inference PyTorch peak allocated | 161.3 MB |

_GPU-used and reserved values come from an identical 1-epoch calibration run._

## Figures

- [test_BoxF1_curve.png](figures/baseline/test_BoxF1_curve.png)
- [test_BoxPR_curve.png](figures/baseline/test_BoxPR_curve.png)
- [test_BoxP_curve.png](figures/baseline/test_BoxP_curve.png)
- [test_BoxR_curve.png](figures/baseline/test_BoxR_curve.png)
- [test_confusion_matrix.png](figures/baseline/test_confusion_matrix.png)
- [test_confusion_matrix_normalized.png](figures/baseline/test_confusion_matrix_normalized.png)
- [val_BoxF1_curve.png](figures/baseline/val_BoxF1_curve.png)
- [val_BoxPR_curve.png](figures/baseline/val_BoxPR_curve.png)
- [val_BoxP_curve.png](figures/baseline/val_BoxP_curve.png)
- [val_BoxR_curve.png](figures/baseline/val_BoxR_curve.png)
- [val_confusion_matrix.png](figures/baseline/val_confusion_matrix.png)
- [val_confusion_matrix_normalized.png](figures/baseline/val_confusion_matrix_normalized.png)

## Artifacts

- Training directory: `runs/train/baseline_yolov8n_seed42`
- Evaluation directory: `runs/eval/baseline_yolov8n_seed42`
- Run config: `runs/train/baseline_yolov8n_seed42/resolved_config.yaml`
- Error analysis: `runs/eval/baseline_yolov8n_seed42/error_analysis/`
- Error-analysis facts: `thesis/results/baseline-error-analysis.md`
- Git commit: `d61cad83bd7763b3aca367828d595ae71a0540d0`

_Facts only; interpretation belongs in `thesis/findings/`._
