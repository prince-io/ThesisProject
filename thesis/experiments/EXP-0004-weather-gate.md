---
title: EXP-0004 — Gate-Only Weather Adaptation
status: validated
date: 2026-11-10
sources:
  - runs/train/weather_gate_seed42
  - runs/eval/weather_gate_seed42
---

# EXP-0004 — Gate-Only Weather Adaptation

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
| Git commit | `c554d78eb6b470210bb8a869b83a17b29445a13a` |

## Configuration

| Key | Value |
|---|---|
| `amp` | `True` |
| `arch` | `gate_only` |
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
| `name` | `weather_gate_seed42` |
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
.venv/bin/python scripts/train.py --config configs/train/weather_gate.yaml --name weather_gate_seed42
.venv/bin/python scripts/evaluate.py --weights runs/train/weather_gate_seed42/weights/best.pt --data configs/datasets/adverse16k.yaml --splits val test --out runs/eval/weather_gate_seed42 --subsets data/processed/adverse16k/subsets --figures-dir thesis/results/figures/weather_gate
.venv/bin/python scripts/error_analysis.py --weights runs/train/weather_gate_seed42/weights/best.pt --out runs/eval/weather_gate_seed42/error_analysis --splits val test
.venv/bin/python scripts/document_results.py --train-run runs/train/weather_gate_seed42 --eval-run runs/eval/weather_gate_seed42 --results thesis/results/weather_gate.md --experiment thesis/experiments/EXP-0004-weather-gate.md --figures-dir thesis/results/figures/weather_gate --vram-report runs/train/weather_gate_seed42/vram.json --title "Weather Gate Results" --experiment-title "EXP-0004 — Gate-Only Weather Adaptation"
```

## Dataset

- Dataset config: `/home/ps/Desktop/thesis_project/configs/datasets/adverse16k.yaml`
- Checksums: `/home/ps/Desktop/thesis_project/data/processed/adverse16k/checksums.sha1`
- Checksums SHA-1: `7de437da75a1698d73195565cfff0c2b3e83da48`

## VRAM

| Measurement | Value |
|---|---:|
| Training PyTorch peak allocated | 697.3 MB |
| Training PyTorch peak reserved | 1246.0 MB |
| Training GPU used peak (nvidia-smi) | 1417.0 MB |
| Training GPU used baseline | 15.0 MB |
| Training GPU used peak delta | 1402.0 MB |
| Inference PyTorch peak allocated | 162.0 MB |

_GPU-used and reserved values were recorded during the full training run._

## Results

See the sibling results document for the full tables.

- Val mAP50 / mAP50-95: 0.4824 / 0.2891
- Test mAP50 / mAP50-95: 0.5444 / 0.3301

## Error Analysis

- Raw artifacts: `runs/eval/weather_gate_seed42/error_analysis/`
- Curated facts: `thesis/results/weather_gate-error-analysis.md`

## Artifacts

- Training: `runs/train/weather_gate_seed42`
- Evaluation: `runs/eval/weather_gate_seed42`
- Figures: `thesis/results/figures/weather_gate/`
