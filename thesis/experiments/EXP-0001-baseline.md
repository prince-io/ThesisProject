---
title: EXP-0001 — YOLOv8n Baseline
status: validated
date: 2026-11-07
sources:
  - runs/train/baseline_yolov8n_seed42
  - runs/eval/baseline_yolov8n_seed42
---

# EXP-0001 — YOLOv8n Baseline

## Objective

Record the detector run and its measured outcomes on `adverse16k`.

## Environment

| Item | Value |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.14.1+cu130 |
| Ultralytics | 8.4.204 |
| CUDA | 13.0 |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| GPU memory | 5806.3 MB |
| Git commit | `d61cad83bd7763b3aca367828d595ae71a0540d0` |

## Configuration

| Key | Value |
|---|---|
| `amp` | `True` |
| `batch` | `4` |
| `cache` | `False` |
| `close_mosaic` | `10` |
| `cls_pw` | `0.5` |
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
| `name` | `baseline_yolov8n_seed42` |
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
.venv/bin/python scripts/train.py --config configs/train/baseline.yaml --name baseline_yolov8n_seed42
.venv/bin/python scripts/evaluate.py --weights runs/train/baseline_yolov8n_seed42/weights/best.pt --data configs/datasets/adverse16k.yaml --splits val test --out runs/eval/baseline_yolov8n_seed42 --subsets data/processed/adverse16k/subsets --figures-dir thesis/results/figures/baseline
.venv/bin/python scripts/error_analysis.py --weights runs/train/baseline_yolov8n_seed42/weights/best.pt --out runs/eval/baseline_yolov8n_seed42/error_analysis --splits val test
.venv/bin/python scripts/document_results.py --train-run runs/train/baseline_yolov8n_seed42 --eval-run runs/eval/baseline_yolov8n_seed42 --results thesis/results/baseline.md --experiment thesis/experiments/EXP-0001-baseline.md --figures-dir thesis/results/figures/baseline --vram-report runs/train/vram_calibration/vram.json --title "Baseline Results — YOLOv8n" --experiment-title "EXP-0001 — YOLOv8n Baseline"
```

## Dataset

- Dataset config: `/home/ps/Desktop/thesis_project/configs/datasets/adverse16k.yaml`
- Checksums: `/home/ps/Desktop/thesis_project/data/processed/adverse16k/checksums.sha1`
- Checksums SHA-1: `7de437da75a1698d73195565cfff0c2b3e83da48`

## VRAM

| Measurement | Value |
|---|---:|
| Training PyTorch peak allocated | 683.8 MB |
| Training PyTorch peak reserved | 778.0 MB |
| Training GPU used peak (nvidia-smi) | 949.0 MB |
| Training GPU used baseline | 15.0 MB |
| Training GPU used peak delta | 934.0 MB |
| Inference PyTorch peak allocated | 161.3 MB |

_GPU-used and reserved values come from an identical 1-epoch calibration run._

## Results

See the sibling results document for the full tables.

- Val mAP50 / mAP50-95: 0.4861 / 0.2904
- Test mAP50 / mAP50-95: 0.5428 / 0.3287

## Error Analysis

- Raw artifacts: `runs/eval/baseline_yolov8n_seed42/error_analysis/`
- Curated facts: `thesis/results/baseline-error-analysis.md`

## Artifacts

- Training: `runs/train/baseline_yolov8n_seed42`
- Evaluation: `runs/eval/baseline_yolov8n_seed42`
- Figures: `thesis/results/figures/baseline_yolov8n_seed42/`
