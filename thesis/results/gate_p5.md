---
title: A+B Results — Gate + Regularized P5 Attention
status: validated
date: 2026-11-10
sources:
  - runs/train/gate_p5_seed42
  - runs/eval/gate_p5_seed42
---

# A+B Results — Gate + Regularized P5 Attention

- Train run: `runs/train/gate_p5_seed42`
- Evaluation run: `runs/eval/gate_p5_seed42`
- Weights: `runs/train/gate_p5_seed42/weights/best.pt`
- Seed: 42
- Class weighting power (`cls_pw`): 0.5
- Dataset checksum: `7de437da75a1698d73195565cfff0c2b3e83da48`

## Overall

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| val | 0.6620 | 0.4365 | 0.4865 | 0.2938 |
| test | 0.7179 | 0.4663 | 0.5387 | 0.3290 |

## Per class

| Class | val AP50 | val AP50-95 | test AP50 | test AP50-95 |
|---|---:|---:|---:|---:|
| bicycle | 0.3141 | 0.1791 | 0.3832 | 0.2277 |
| bus | 0.4585 | 0.3196 | 0.4154 | 0.2840 |
| car | 0.7167 | 0.4393 | 0.7156 | 0.4352 |
| motorcycle | 0.3727 | 0.2157 | 0.5709 | 0.3501 |
| person | 0.5702 | 0.3154 | 0.6083 | 0.3482 |

## Per condition

| Name | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 |
|---|---:|---:|---:|---:|
| fog | 0.6923 | 0.4571 | 0.6708 | 0.4406 |
| night | 0.3275 | 0.1715 | 0.4174 | 0.2255 |
| rain | 0.4529 | 0.2716 | 0.3994 | 0.2319 |
| snow | 0.3951 | 0.2214 | 0.3963 | 0.2224 |

## Per source

| Name | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 |
|---|---:|---:|---:|---:|
| acdc | 0.2889 | 0.1634 | 0.3075 | 0.1656 |
| bdd | 0.4228 | 0.2329 | 0.4255 | 0.2438 |
| dawn | 0.8556 | 0.4864 | 0.7606 | 0.4574 |
| rtts | 0.6817 | 0.4580 | 0.6807 | 0.4458 |

## Training

- Epochs completed: 86
- Best epoch (mAP50-95): 66
- Best val mAP50 / mAP50-95: 0.4862 / 0.2943
- Total training time: 4.48 h

## Efficiency

- Parameters: 3.584 M
- GFLOPs at 640: 9.099
- Inference latency: 3.939 ms/image
- FPS: 253.85

### VRAM

| Measurement | Value |
|---|---:|
| Training PyTorch peak allocated | 776.0 MB |
| Training PyTorch peak reserved | 1328.0 MB |
| Training GPU used peak (nvidia-smi) | 1535.0 MB |
| Training GPU used baseline | 15.0 MB |
| Training GPU used peak delta | 1520.0 MB |
| Inference PyTorch peak allocated | 205.4 MB |

_GPU-used and reserved values were recorded during the full training run._

## Figures

- [test_BoxF1_curve.png](figures/gate_p5/test_BoxF1_curve.png)
- [test_BoxPR_curve.png](figures/gate_p5/test_BoxPR_curve.png)
- [test_BoxP_curve.png](figures/gate_p5/test_BoxP_curve.png)
- [test_BoxR_curve.png](figures/gate_p5/test_BoxR_curve.png)
- [test_confusion_matrix.png](figures/gate_p5/test_confusion_matrix.png)
- [test_confusion_matrix_normalized.png](figures/gate_p5/test_confusion_matrix_normalized.png)
- [val_BoxF1_curve.png](figures/gate_p5/val_BoxF1_curve.png)
- [val_BoxPR_curve.png](figures/gate_p5/val_BoxPR_curve.png)
- [val_BoxP_curve.png](figures/gate_p5/val_BoxP_curve.png)
- [val_BoxR_curve.png](figures/gate_p5/val_BoxR_curve.png)
- [val_confusion_matrix.png](figures/gate_p5/val_confusion_matrix.png)
- [val_confusion_matrix_normalized.png](figures/gate_p5/val_confusion_matrix_normalized.png)

## Artifacts

- Training directory: `runs/train/gate_p5_seed42`
- Evaluation directory: `runs/eval/gate_p5_seed42`
- Run config: `runs/train/gate_p5_seed42/resolved_config.yaml`
- Error analysis: `runs/eval/gate_p5_seed42/error_analysis/`
- Error-analysis facts: `thesis/results/gate_p5-error-analysis.md`
- Git commit: `1d5ab73cbca15fb42f973ec872f039ce5fbdbc9b`

_Facts only; interpretation belongs in `thesis/findings/`._
