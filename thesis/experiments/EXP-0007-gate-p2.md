---
title: EXP-0007 — Gate-Only + P2 High-Resolution Head
status: validated
date: 2026-11-11
sources:
  - runs/train/gate_p2_seed42
  - runs/eval/gate_p2_seed42
  - thesis/results/gate_p2.md
  - thesis/results/gate_p2-error-analysis.md
  - thesis/results/gate-p2-comparison.md
---

# EXP-0007 — Gate-Only + P2 High-Resolution Head

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
| Git commit | `9909a884f580998a595fa0e58b3577c0e7d7e5c5` |

## Configuration

| Key | Value |
|---|---|
| `amp` | `True` |
| `arch` | `gate_p2` |
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
| `name` | `gate_p2_seed42` |
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
.venv/bin/python scripts/train.py --config configs/train/gate_p2.yaml --name gate_p2_seed42
.venv/bin/python scripts/evaluate.py --weights runs/train/gate_p2_seed42/weights/best.pt --data configs/datasets/adverse16k.yaml --splits val test --out runs/eval/gate_p2_seed42 --subsets data/processed/adverse16k/subsets --figures-dir thesis/results/figures/gate_p2
.venv/bin/python scripts/error_analysis.py --weights runs/train/gate_p2_seed42/weights/best.pt --out runs/eval/gate_p2_seed42/error_analysis --splits val test
.venv/bin/python scripts/document_results.py --train-run runs/train/gate_p2_seed42 --eval-run runs/eval/gate_p2_seed42 --results thesis/results/gate_p2.md --experiment thesis/experiments/EXP-0007-gate-p2.md --figures-dir thesis/results/figures/gate_p2 --vram-report runs/train/gate_p2_seed42/vram.json --title "Gate + P2 Head Results" --experiment-title "EXP-0007 — Gate-Only + P2 High-Resolution Head"
```

## Dataset

- Dataset config: `/home/ps/Desktop/thesis_project/configs/datasets/adverse16k.yaml`
- Checksums: `/home/ps/Desktop/thesis_project/data/processed/adverse16k/checksums.sha1`
- Checksums SHA-1: `7de437da75a1698d73195565cfff0c2b3e83da48`

## VRAM

| Measurement | Value |
|---|---:|
| Training PyTorch peak allocated | 1433.3 MB |
| Training PyTorch peak reserved | 3106.0 MB |
| Training GPU used peak (nvidia-smi) | 3277.0 MB |
| Training GPU used baseline | 15.0 MB |
| Training GPU used peak delta | 3262.0 MB |
| Inference PyTorch peak allocated | 270.9 MB |

_GPU-used and reserved values were recorded during the full training run._

## Training

- Epochs completed: 82 (early stop; configured 100, patience 20)
- Best epoch (mAP50-95): 62
- Best epoch (mAP50): 70
- Training time: 6.01 h

## Results

See the sibling results document for the full tables.

- Val mAP50 / mAP50-95: 0.4850 / 0.2910
- Test mAP50 / mAP50-95: 0.5571 / 0.3351

## Error Analysis

- Raw artifacts: `runs/eval/gate_p2_seed42/error_analysis/`
- Curated facts: `thesis/results/gate_p2-error-analysis.md`
- Comparison: `thesis/results/gate-p2-comparison.md`
- Interpretation: `thesis/findings/gate-p2-analysis.md`

## Artifacts

- Training: `runs/train/gate_p2_seed42`
- Evaluation: `runs/eval/gate_p2_seed42`
- Figures: `thesis/results/figures/gate_p2/`
