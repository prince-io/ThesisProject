---
title: EXP-0002 — cls_pw=0.0 Ablation
status: validated
date: 2026-11-08
sources:
  - runs/train/ablation_cls_pw0_seed42
  - runs/eval/ablation_cls_pw0_seed42
---

# EXP-0002 — cls_pw=0.0 Ablation

## Objective

Record the detector run and its measured outcomes on `adverse16k`.

Isolate the effect of class weighting: this run matches `EXP-0001` except for
`cls_pw=0.0` (baseline uses `cls_pw=0.5`).

## Environment

| Item | Value |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.14.1+cu130 |
| Ultralytics | 8.4.204 |
| CUDA | 13.0 |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| GPU memory | 5806.3 MB |
| Git commit | `4d9003375a36fed0376c937b639e969bd9bc1470` |

## Configuration

| Key | Value |
|---|---|
| `amp` | `True` |
| `batch` | `4` |
| `cache` | `False` |
| `close_mosaic` | `10` |
| `cls_pw` | `0.0` |
| `cls_remap` | `True` |
| `cos_lr` | `True` |
| `data` | `/home/ps/Desktop/thesis_project/configs/datasets/adverse16k.yaml` |
| `deterministic` | `True` |
| `device` | `0` |
| `epochs` | `100` |
| `exist_ok` | `False` |
| `imgsz` | `640` |
| `lr0` | `0.0002` |
| `lrf` | `0.01` |
| `mixup` | `0.05` |
| `model` | `/home/ps/Desktop/thesis_project/yolov8n.pt` |
| `mosaic` | `0.5` |
| `name` | `ablation_cls_pw0_seed42` |
| `nbs` | `8` |
| `optimizer` | `AdamW` |
| `patience` | `20` |
| `plots` | `True` |
| `pretrained` | `True` |
| `project` | `/home/ps/Desktop/thesis_project/runs/train` |
| `seed` | `42` |
| `val` | `True` |
| `warmup_epochs` | `3.0` |
| `weight_decay` | `0.0005` |
| `workers` | `4` |

## Commands

```bash
.venv/bin/python scripts/train.py --config configs/train/ablation_cls_pw0.yaml --name ablation_cls_pw0_seed42
.venv/bin/python scripts/evaluate.py --weights runs/train/ablation_cls_pw0_seed42/weights/best.pt --data configs/datasets/adverse16k.yaml --splits val test --out runs/eval/ablation_cls_pw0_seed42 --subsets data/processed/adverse16k/subsets --figures-dir thesis/results/figures/ablation_cls_pw0
.venv/bin/python scripts/error_analysis.py --weights runs/train/ablation_cls_pw0_seed42/weights/best.pt --out runs/eval/ablation_cls_pw0_seed42/error_analysis --splits val test
.venv/bin/python scripts/document_results.py --train-run runs/train/ablation_cls_pw0_seed42 --eval-run runs/eval/ablation_cls_pw0_seed42 --results thesis/results/ablation_cls_pw0.md --experiment thesis/experiments/EXP-0002-cls-pw0.md --figures-dir thesis/results/figures/ablation_cls_pw0 --vram-report runs/train/ablation_cls_pw0_seed42/vram.json --title "Ablation Results — cls_pw=0.0" --experiment-title "EXP-0002 — cls_pw=0.0 Ablation"
```

## Dataset

- Dataset config: `/home/ps/Desktop/thesis_project/configs/datasets/adverse16k.yaml`
- Checksums: `/home/ps/Desktop/thesis_project/data/processed/adverse16k/checksums.sha1`
- Checksums SHA-1: `7de437da75a1698d73195565cfff0c2b3e83da48`

## VRAM

| Measurement | Value |
|---|---:|
| Training PyTorch peak allocated | 683.8 MB |
| Training PyTorch peak reserved | 1218.0 MB |
| Training GPU used peak (nvidia-smi) | 1389.0 MB |
| Training GPU used baseline | 15.0 MB |
| Training GPU used peak delta | 1374.0 MB |
| Inference PyTorch peak allocated | 161.3 MB |

_GPU-used and reserved values were recorded during the full training run._

## Results

See the sibling results document for the full tables.

- Val mAP50 / mAP50-95: 0.4874 / 0.2907
- Test mAP50 / mAP50-95: 0.5314 / 0.3254

## Error Analysis

- Raw artifacts: `runs/eval/ablation_cls_pw0_seed42/error_analysis/`
- Curated facts: `thesis/results/ablation_cls_pw0-error-analysis.md`

## Artifacts

- Training: `runs/train/ablation_cls_pw0_seed42`
- Evaluation: `runs/eval/ablation_cls_pw0_seed42`
- Figures: `thesis/results/figures/ablation_cls_pw0/`
- Baseline comparison: `thesis/results/ablation-comparison.md`
- Interpretation: `thesis/findings/ablation-analysis.md`
