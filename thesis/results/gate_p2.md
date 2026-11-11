---
title: Gate + P2 Head Results
status: validated
date: 2026-11-11
sources:
  - runs/train/gate_p2_seed42
  - runs/eval/gate_p2_seed42
---

# Gate + P2 Head Results

- Train run: `runs/train/gate_p2_seed42`
- Evaluation run: `runs/eval/gate_p2_seed42`
- Weights: `runs/train/gate_p2_seed42/weights/best.pt`
- Seed: 42
- Class weighting power (`cls_pw`): 0.5
- Dataset checksum: `7de437da75a1698d73195565cfff0c2b3e83da48`

## Overall

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| val | 0.6537 | 0.4275 | 0.4850 | 0.2910 |
| test | 0.7063 | 0.4973 | 0.5571 | 0.3351 |

## Per class

| Class | val AP50 | val AP50-95 | test AP50 | test AP50-95 |
|---|---:|---:|---:|---:|
| bicycle | 0.3188 | 0.1814 | 0.4077 | 0.2341 |
| bus | 0.4297 | 0.2976 | 0.4028 | 0.2728 |
| car | 0.7359 | 0.4544 | 0.7393 | 0.4527 |
| motorcycle | 0.3559 | 0.2021 | 0.6195 | 0.3618 |
| person | 0.5847 | 0.3193 | 0.6161 | 0.3540 |

## Per condition

| Name | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 |
|---|---:|---:|---:|---:|
| fog | 0.6838 | 0.4492 | 0.6906 | 0.4441 |
| night | 0.3421 | 0.1800 | 0.4471 | 0.2431 |
| rain | 0.4793 | 0.2789 | 0.4043 | 0.2329 |
| snow | 0.3736 | 0.2126 | 0.4606 | 0.2345 |

## Per source

| Name | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 |
|---|---:|---:|---:|---:|
| acdc | 0.3045 | 0.1690 | 0.3419 | 0.1870 |
| bdd | 0.4176 | 0.2340 | 0.4357 | 0.2490 |
| dawn | 0.8398 | 0.4889 | 0.7126 | 0.4238 |
| rtts | 0.6699 | 0.4458 | 0.6989 | 0.4477 |

## Training

- Epochs completed: 82
- Best epoch (mAP50-95): 62
- Best val mAP50: 0.4850 (epoch 70)
- Best val mAP50-95: 0.2910 (epoch 62)
- Total training time: 6.01 h

## Efficiency

- Parameters: 3.270 M
- GFLOPs at 640: 13.281
- Inference latency: 6.381 ms/image
- FPS: 156.72

### VRAM

| Measurement | Value |
|---|---:|
| Training PyTorch peak allocated | 1433.3 MB |
| Training PyTorch peak reserved | 3106.0 MB |
| Training GPU used peak (nvidia-smi) | 3277.0 MB |
| Training GPU used baseline | 15.0 MB |
| Training GPU used peak delta | 3262.0 MB |
| Inference PyTorch peak allocated | 270.9 MB |

_GPU-used and reserved values were recorded during the full training run._

## Figures

- [test_BoxF1_curve.png](figures/gate_p2/test_BoxF1_curve.png)
- [test_BoxPR_curve.png](figures/gate_p2/test_BoxPR_curve.png)
- [test_BoxP_curve.png](figures/gate_p2/test_BoxP_curve.png)
- [test_BoxR_curve.png](figures/gate_p2/test_BoxR_curve.png)
- [test_confusion_matrix.png](figures/gate_p2/test_confusion_matrix.png)
- [test_confusion_matrix_normalized.png](figures/gate_p2/test_confusion_matrix_normalized.png)
- [val_BoxF1_curve.png](figures/gate_p2/val_BoxF1_curve.png)
- [val_BoxPR_curve.png](figures/gate_p2/val_BoxPR_curve.png)
- [val_BoxP_curve.png](figures/gate_p2/val_BoxP_curve.png)
- [val_BoxR_curve.png](figures/gate_p2/val_BoxR_curve.png)
- [val_confusion_matrix.png](figures/gate_p2/val_confusion_matrix.png)
- [val_confusion_matrix_normalized.png](figures/gate_p2/val_confusion_matrix_normalized.png)

## Artifacts

- Training directory: `runs/train/gate_p2_seed42`
- Evaluation directory: `runs/eval/gate_p2_seed42`
- Run config: `runs/train/gate_p2_seed42/resolved_config.yaml`
- Error analysis: `runs/eval/gate_p2_seed42/error_analysis/`
- Error-analysis facts: `thesis/results/gate_p2-error-analysis.md`
- Comparison: `thesis/results/gate-p2-comparison.md`
- Interpretation: `thesis/findings/gate-p2-analysis.md`
- Git commit: `9909a884f580998a595fa0e58b3577c0e7d7e5c5`

_Facts only; interpretation belongs in `thesis/findings/`._
