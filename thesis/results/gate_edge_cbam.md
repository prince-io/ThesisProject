---
title: Gate + Edge + CBAM Results
status: validated
date: 2026-11-10
sources:
  - runs/train/gate_edge_cbam_seed42
  - runs/eval/gate_edge_cbam_seed42
---

# Gate + Edge + CBAM Results

- Train run: `runs/train/gate_edge_cbam_seed42`
- Evaluation run: `runs/eval/gate_edge_cbam_seed42`
- Weights: `runs/train/gate_edge_cbam_seed42/weights/best.pt`
- Seed: 42
- Class weighting power (`cls_pw`): 0.5
- Dataset checksum: `7de437da75a1698d73195565cfff0c2b3e83da48`

## Overall

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| val | 0.6511 | 0.4331 | 0.4859 | 0.2910 |
| test | 0.7097 | 0.4803 | 0.5420 | 0.3295 |

## Per class

| Class | val AP50 | val AP50-95 | test AP50 | test AP50-95 |
|---|---:|---:|---:|---:|
| bicycle | 0.2998 | 0.1757 | 0.3856 | 0.2274 |
| bus | 0.4582 | 0.3195 | 0.4138 | 0.2741 |
| car | 0.7200 | 0.4395 | 0.7186 | 0.4358 |
| motorcycle | 0.3757 | 0.2062 | 0.5800 | 0.3631 |
| person | 0.5761 | 0.3139 | 0.6121 | 0.3473 |

## Per condition

| Name | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 |
|---|---:|---:|---:|---:|
| fog | 0.7036 | 0.4646 | 0.6817 | 0.4489 |
| night | 0.3320 | 0.1679 | 0.4090 | 0.2212 |
| rain | 0.4499 | 0.2636 | 0.4004 | 0.2270 |
| snow | 0.3656 | 0.2034 | 0.3846 | 0.2201 |

## Per source

| Name | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 |
|---|---:|---:|---:|---:|
| acdc | 0.2867 | 0.1513 | 0.2943 | 0.1628 |
| bdd | 0.4169 | 0.2298 | 0.4236 | 0.2383 |
| dawn | 0.8768 | 0.5128 | 0.7108 | 0.4279 |
| rtts | 0.6921 | 0.4631 | 0.6881 | 0.4507 |

## Training

- Epochs completed: 100
- Best epoch (mAP50-95): 85
- Best val mAP50 / mAP50-95: 0.4895 / 0.2911
- Total training time: 5.57 h

## Efficiency

- Parameters: 3.204 M
- GFLOPs at 640: 8.895
- Inference latency: 4.174 ms/image
- FPS: 239.61

### VRAM

| Measurement | Value |
|---|---:|
| Training PyTorch peak allocated | 755.8 MB |
| Training PyTorch peak reserved | 1294.0 MB |
| Training GPU used peak (nvidia-smi) | 1467.0 MB |
| Training GPU used baseline | 15.0 MB |
| Training GPU used peak delta | 1452.0 MB |
| Inference PyTorch peak allocated | 173.2 MB |

_GPU-used and reserved values were recorded during the full training run._

## Figures

- [test_BoxF1_curve.png](figures/gate_edge_cbam/test_BoxF1_curve.png)
- [test_BoxPR_curve.png](figures/gate_edge_cbam/test_BoxPR_curve.png)
- [test_BoxP_curve.png](figures/gate_edge_cbam/test_BoxP_curve.png)
- [test_BoxR_curve.png](figures/gate_edge_cbam/test_BoxR_curve.png)
- [test_confusion_matrix.png](figures/gate_edge_cbam/test_confusion_matrix.png)
- [test_confusion_matrix_normalized.png](figures/gate_edge_cbam/test_confusion_matrix_normalized.png)
- [val_BoxF1_curve.png](figures/gate_edge_cbam/val_BoxF1_curve.png)
- [val_BoxPR_curve.png](figures/gate_edge_cbam/val_BoxPR_curve.png)
- [val_BoxP_curve.png](figures/gate_edge_cbam/val_BoxP_curve.png)
- [val_BoxR_curve.png](figures/gate_edge_cbam/val_BoxR_curve.png)
- [val_confusion_matrix.png](figures/gate_edge_cbam/val_confusion_matrix.png)
- [val_confusion_matrix_normalized.png](figures/gate_edge_cbam/val_confusion_matrix_normalized.png)

## Artifacts

- Training directory: `runs/train/gate_edge_cbam_seed42`
- Evaluation directory: `runs/eval/gate_edge_cbam_seed42`
- Run config: `runs/train/gate_edge_cbam_seed42/resolved_config.yaml`
- Error analysis: `runs/eval/gate_edge_cbam_seed42/error_analysis/`
- Error-analysis facts: `thesis/results/gate_edge_cbam-error-analysis.md`
- Interpretation: `thesis/findings/gate-edge-cbam-analysis.md`
- Git commit: `f093b300ee6694fae63260d680ae60ed21a6e782`

_Facts only; interpretation belongs in `thesis/findings/`._
