---
title: V1 Results — P5 LiteViT
status: validated
date: 2026-11-09
sources:
  - runs/train/litevit_v1_p5_seed42_v2
  - runs/eval/litevit_v1_p5_seed42_v2
---

# V1 Results — P5 LiteViT

- Train run: `runs/train/litevit_v1_p5_seed42_v2`
- Evaluation run: `runs/eval/litevit_v1_p5_seed42_v2`
- Weights: `runs/train/litevit_v1_p5_seed42_v2/weights/best.pt`
- Seed: 42
- Class weighting power (`cls_pw`): 0.5
- Dataset checksum: `7de437da75a1698d73195565cfff0c2b3e83da48`

## Overall

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| val | 0.6108 | 0.3795 | 0.4231 | 0.2499 |
| test | 0.6359 | 0.4383 | 0.4888 | 0.2895 |

## Per class

| Class | val AP50 | val AP50-95 | test AP50 | test AP50-95 |
|---|---:|---:|---:|---:|
| bicycle | 0.2615 | 0.1500 | 0.2972 | 0.1780 |
| bus | 0.3668 | 0.2459 | 0.3395 | 0.2313 |
| car | 0.6813 | 0.4133 | 0.6828 | 0.4094 |
| motorcycle | 0.2763 | 0.1513 | 0.5554 | 0.3044 |
| person | 0.5297 | 0.2888 | 0.5689 | 0.3245 |

## Per condition

| Name | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 |
|---|---:|---:|---:|---:|
| fog | 0.6140 | 0.3918 | 0.6180 | 0.3904 |
| night | 0.2968 | 0.1533 | 0.3961 | 0.2061 |
| rain | 0.4260 | 0.2416 | 0.3419 | 0.1976 |
| snow | 0.3218 | 0.1847 | 0.4232 | 0.2016 |

## Per source

| Name | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 |
|---|---:|---:|---:|---:|
| acdc | 0.2717 | 0.1421 | 0.2894 | 0.1456 |
| bdd | 0.3579 | 0.1999 | 0.3824 | 0.2174 |
| dawn | 0.7801 | 0.4655 | 0.6177 | 0.3858 |
| rtts | 0.6016 | 0.3898 | 0.6295 | 0.3951 |

## Training

- Epochs completed: 38
- Best epoch (mAP50-95): 18
- Best val mAP50 / mAP50-95: 0.4232 / 0.2497
- Total training time: 1.92 h

## Efficiency

- Parameters: 3.674 M
- GFLOPs at 640: 9.206
- Inference latency: 3.928 ms/image
- FPS: 254.61

### VRAM

| Measurement | Value |
|---|---:|
| Training PyTorch peak allocated | 729.5 MB |
| Training PyTorch peak reserved | 1152.0 MB |
| Training GPU used peak (nvidia-smi) | 1355.0 MB |
| Training GPU used baseline | 15.0 MB |
| Training GPU used peak delta | 1340.0 MB |
| Inference PyTorch peak allocated | 207.0 MB |

_GPU-used and reserved values were recorded during the full training run._

## Figures

- [test_BoxF1_curve.png](figures/litevit_v1_p5/test_BoxF1_curve.png)
- [test_BoxPR_curve.png](figures/litevit_v1_p5/test_BoxPR_curve.png)
- [test_BoxP_curve.png](figures/litevit_v1_p5/test_BoxP_curve.png)
- [test_BoxR_curve.png](figures/litevit_v1_p5/test_BoxR_curve.png)
- [test_confusion_matrix.png](figures/litevit_v1_p5/test_confusion_matrix.png)
- [test_confusion_matrix_normalized.png](figures/litevit_v1_p5/test_confusion_matrix_normalized.png)
- [val_BoxF1_curve.png](figures/litevit_v1_p5/val_BoxF1_curve.png)
- [val_BoxPR_curve.png](figures/litevit_v1_p5/val_BoxPR_curve.png)
- [val_BoxP_curve.png](figures/litevit_v1_p5/val_BoxP_curve.png)
- [val_BoxR_curve.png](figures/litevit_v1_p5/val_BoxR_curve.png)
- [val_confusion_matrix.png](figures/litevit_v1_p5/val_confusion_matrix.png)
- [val_confusion_matrix_normalized.png](figures/litevit_v1_p5/val_confusion_matrix_normalized.png)

## Artifacts

- Training directory: `runs/train/litevit_v1_p5_seed42_v2`
- Evaluation directory: `runs/eval/litevit_v1_p5_seed42_v2`
- Run config: `runs/train/litevit_v1_p5_seed42_v2/resolved_config.yaml`
- Error analysis: `runs/eval/litevit_v1_p5_seed42_v2/error_analysis/`
- Error-analysis facts: `thesis/results/litevit_v1_p5-error-analysis.md`
- Git commit: `f77a949809737b6215c03d61da6ade77782986da`

_Facts only; interpretation belongs in `thesis/findings/`._
