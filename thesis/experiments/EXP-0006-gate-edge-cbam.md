---
title: EXP-0006 — Gate + Sobel Edge + Spatial CBAM
status: validated
date: 2026-11-10
sources:
  - runs/train/gate_edge_cbam_seed42
  - runs/eval/gate_edge_cbam_seed42
---

# EXP-0006 — Gate + Sobel Edge + Spatial CBAM

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
| Git commit | `f093b300ee6694fae63260d680ae60ed21a6e782` |

## Configuration

| Key | Value |
|---|---|
| `amp` | `True` |
| `arch` | `gate_edge_cbam` |
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
| `name` | `gate_edge_cbam_seed42` |
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
.venv/bin/python scripts/train.py --config configs/train/gate_edge_cbam.yaml --name gate_edge_cbam_seed42
.venv/bin/python scripts/evaluate.py --weights runs/train/gate_edge_cbam_seed42/weights/best.pt --data configs/datasets/adverse16k.yaml --splits val test --out runs/eval/gate_edge_cbam_seed42 --subsets data/processed/adverse16k/subsets --figures-dir thesis/results/figures/gate_edge_cbam
.venv/bin/python scripts/error_analysis.py --weights runs/train/gate_edge_cbam_seed42/weights/best.pt --out runs/eval/gate_edge_cbam_seed42/error_analysis --splits val test
.venv/bin/python scripts/document_results.py --train-run runs/train/gate_edge_cbam_seed42 --eval-run runs/eval/gate_edge_cbam_seed42 --results thesis/results/gate_edge_cbam.md --experiment thesis/experiments/EXP-0006-gate-edge-cbam.md --figures-dir thesis/results/figures/gate_edge_cbam --vram-report runs/train/gate_edge_cbam_seed42/vram.json --title "Gate + Edge + CBAM Results" --experiment-title "EXP-0006 — Gate + Sobel Edge + Spatial CBAM"
```

## Dataset

- Dataset config: `/home/ps/Desktop/thesis_project/configs/datasets/adverse16k.yaml`
- Checksums: `/home/ps/Desktop/thesis_project/data/processed/adverse16k/checksums.sha1`
- Checksums SHA-1: `7de437da75a1698d73195565cfff0c2b3e83da48`

## VRAM

| Measurement | Value |
|---|---:|
| Training PyTorch peak allocated | 755.8 MB |
| Training PyTorch peak reserved | 1294.0 MB |
| Training GPU used peak (nvidia-smi) | 1467.0 MB |
| Training GPU used baseline | 15.0 MB |
| Training GPU used peak delta | 1452.0 MB |
| Inference PyTorch peak allocated | 173.2 MB |

_GPU-used and reserved values were recorded during the full training run._

## Results

See the sibling results document for the full tables.

- Val mAP50 / mAP50-95: 0.4859 / 0.2910
- Test mAP50 / mAP50-95: 0.5420 / 0.3295

## Error Analysis

- Raw artifacts: `runs/eval/gate_edge_cbam_seed42/error_analysis/`
- Curated facts: `thesis/results/gate_edge_cbam-error-analysis.md`

## Artifacts

- Training: `runs/train/gate_edge_cbam_seed42`
- Evaluation: `runs/eval/gate_edge_cbam_seed42`
- Figures: `thesis/results/figures/gate_edge_cbam/`
