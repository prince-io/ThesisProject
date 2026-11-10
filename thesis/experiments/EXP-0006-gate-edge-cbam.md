---
title: EXP-0006 — Gate + Sobel Edge + Spatial CBAM
status: draft
date: 2026-11-10
sources:
  - configs/train/gate_edge_cbam.yaml
  - thesis/methodology/gate-edge-cbam.md
  - thesis/decisions/ADR-0011-gate-edge-cbam-design.md
---

# EXP-0006 — Gate + Sobel Edge + Spatial CBAM

## Objective

Test a cheap combined weather model: channel gate, Sobel edge residual, and
spatial CBAM, without a P5 attention block.

## Scope

- YOLOv8n backbone/neck/head unchanged.
- Sobel edge residual on backbone P3/P4 (layers 4/6), zero-initialized.
- Channel gate on neck P3/P4/P5 (layers 15/18/21), zero-initialized.
- Spatial CBAM after the gate on P3/P4/P5, zero-initialized.
- No P5/P4 attention.
- Same protocol as `EXP-0001`.

## Size

| Component | Parameters |
|---|---:|
| YOLOv8n 5-class baseline | 3,138,975 |
| Sobel edge P3/P4 | 20,864 |
| Channel gates P3/P4/P5 | 43,568 |
| Spatial attention P3/P4/P5 | 297 |
| **Total** | **3,203,704** |

GFLOPs at 640: **8.895**.

## Setup

| Item | Value |
|---|---|
| Config | `configs/train/gate_edge_cbam.yaml` |
| Architecture | `arch: gate_edge_cbam` |
| Pretrained weights | `yolov8n.pt` |
| Dataset | `adverse16k` |
| Seed | 42 |
| Hardware | RTX 3050 6 GB |
| Train run | `runs/train/gate_edge_cbam_seed42` |
| Eval run | `runs/eval/gate_edge_cbam_seed42` |

## Commands

```bash
# Optional 5-epoch smoke run
.venv/bin/python scripts/train.py \
  --config configs/train/gate_edge_cbam.yaml \
  --name gate_edge_cbam_smoke --epochs 5

# Full kitchen-sink run
.venv/bin/python scripts/train.py \
  --config configs/train/gate_edge_cbam.yaml \
  --name gate_edge_cbam_seed42

.venv/bin/python scripts/evaluate.py \
  --weights runs/train/gate_edge_cbam_seed42/weights/best.pt \
  --data configs/datasets/adverse16k.yaml \
  --splits val test \
  --out runs/eval/gate_edge_cbam_seed42 \
  --subsets data/processed/adverse16k/subsets \
  --figures-dir thesis/results/figures/gate_edge_cbam

.venv/bin/python scripts/error_analysis.py \
  --weights runs/train/gate_edge_cbam_seed42/weights/best.pt \
  --out runs/eval/gate_edge_cbam_seed42/error_analysis \
  --splits val test

.venv/bin/python scripts/document_results.py \
  --train-run runs/train/gate_edge_cbam_seed42 \
  --eval-run runs/eval/gate_edge_cbam_seed42 \
  --results thesis/results/gate_edge_cbam.md \
  --experiment thesis/experiments/EXP-0006-gate-edge-cbam.md \
  --figures-dir thesis/results/figures/gate_edge_cbam \
  --vram-report runs/train/gate_edge_cbam_seed42/vram.json \
  --title "Gate + Edge + CBAM Results" \
  --experiment-title "EXP-0006 — Gate + Sobel Edge + Spatial CBAM"
```

## Hypothesis and checks

- Initial output should match the baseline exactly.
- Saved `best.pt` must contain `FeatureWithEdge` at 4/6 and `FeatureWithGate` at
  15/18/21, with about **3.204 M** parameters.
- Compare against baseline, V1, gate-only, and A+B.
- If it improves, run edge-only and CBAM-only ablations for attribution.
- If not, keep gate-only as the final model.

## Status

Planned. Training and evaluation are pending.
