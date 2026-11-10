---
title: Weather Gate Results
status: validated
date: 2026-11-10
sources:
  - runs/train/weather_gate_seed42
  - runs/eval/weather_gate_seed42
---

# Weather Gate Results

- Train run: `runs/train/weather_gate_seed42`
- Evaluation run: `runs/eval/weather_gate_seed42`
- Weights: `runs/train/weather_gate_seed42/weights/best.pt`
- Seed: 42
- Class weighting power (`cls_pw`): 0.5
- Dataset checksum: `7de437da75a1698d73195565cfff0c2b3e83da48`

## Overall

| Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| val | 0.6456 | 0.4421 | 0.4824 | 0.2891 |
| test | 0.6857 | 0.4965 | 0.5444 | 0.3301 |

## Per class

| Class | val AP50 | val AP50-95 | test AP50 | test AP50-95 |
|---|---:|---:|---:|---:|
| bicycle | 0.3121 | 0.1825 | 0.4126 | 0.2360 |
| bus | 0.4574 | 0.3165 | 0.4164 | 0.2754 |
| car | 0.7178 | 0.4391 | 0.7171 | 0.4357 |
| motorcycle | 0.3592 | 0.1987 | 0.5633 | 0.3556 |
| person | 0.5654 | 0.3086 | 0.6124 | 0.3480 |

## Per condition

| Name | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 |
|---|---:|---:|---:|---:|
| fog | 0.6832 | 0.4476 | 0.6773 | 0.4454 |
| night | 0.3323 | 0.1682 | 0.4595 | 0.2420 |
| rain | 0.4569 | 0.2720 | 0.4169 | 0.2405 |
| snow | 0.3791 | 0.2170 | 0.3898 | 0.2130 |

## Per source

| Name | val mAP50 | val mAP50-95 | test mAP50 | test mAP50-95 |
|---|---:|---:|---:|---:|
| acdc | 0.3008 | 0.1641 | 0.3258 | 0.1753 |
| bdd | 0.4190 | 0.2332 | 0.4327 | 0.2448 |
| dawn | 0.8308 | 0.4501 | 0.7533 | 0.4578 |
| rtts | 0.6735 | 0.4491 | 0.6856 | 0.4487 |

## Training

- Epochs completed: 100
- Best epoch (mAP50-95): 90
- Best val mAP50 / mAP50-95: 0.4823 / 0.2904
- Total training time: 4.68 h

## Efficiency

- Parameters: 3.183 M
- GFLOPs at 640: 8.789
- Inference latency: 4.060 ms/image
- FPS: 246.32

### VRAM

| Measurement | Value |
|---|---:|
| Training PyTorch peak allocated | 697.3 MB |
| Training PyTorch peak reserved | 1246.0 MB |
| Training GPU used peak (nvidia-smi) | 1417.0 MB |
| Training GPU used baseline | 15.0 MB |
| Training GPU used peak delta | 1402.0 MB |
| Inference PyTorch peak allocated | 162.0 MB |

_GPU-used and reserved values were recorded during the full training run._

## Figures

- [test_BoxF1_curve.png](figures/weather_gate/test_BoxF1_curve.png)
- [test_BoxPR_curve.png](figures/weather_gate/test_BoxPR_curve.png)
- [test_BoxP_curve.png](figures/weather_gate/test_BoxP_curve.png)
- [test_BoxR_curve.png](figures/weather_gate/test_BoxR_curve.png)
- [test_confusion_matrix.png](figures/weather_gate/test_confusion_matrix.png)
- [test_confusion_matrix_normalized.png](figures/weather_gate/test_confusion_matrix_normalized.png)
- [val_BoxF1_curve.png](figures/weather_gate/val_BoxF1_curve.png)
- [val_BoxPR_curve.png](figures/weather_gate/val_BoxPR_curve.png)
- [val_BoxP_curve.png](figures/weather_gate/val_BoxP_curve.png)
- [val_BoxR_curve.png](figures/weather_gate/val_BoxR_curve.png)
- [val_confusion_matrix.png](figures/weather_gate/val_confusion_matrix.png)
- [val_confusion_matrix_normalized.png](figures/weather_gate/val_confusion_matrix_normalized.png)

## Artifacts

- Training directory: `runs/train/weather_gate_seed42`
- Evaluation directory: `runs/eval/weather_gate_seed42`
- Run config: `runs/train/weather_gate_seed42/resolved_config.yaml`
- Error analysis: `runs/eval/weather_gate_seed42/error_analysis/`
- Error-analysis facts: `thesis/results/weather_gate-error-analysis.md`
- Git commit: `c554d78eb6b470210bb8a869b83a17b29445a13a`

_Facts only; interpretation belongs in `thesis/findings/`._
