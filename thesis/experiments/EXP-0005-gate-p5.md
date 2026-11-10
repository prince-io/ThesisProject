---
title: EXP-0005 — Gate + Regularized P5 Attention (A+B)
status: draft
date: 2026-11-10
sources:
  - configs/train/gate_p5.yaml
  - thesis/methodology/gate-p5.md
  - thesis/decisions/ADR-0010-gate-p5-design.md
---

# EXP-0005 — Gate + Regularized P5 Attention (A+B)

## Objective

Test whether regularized, lower-capacity P5 attention adds value on top of the
P3/P4/P5 weather gate. V1 attention alone overfit; gate-only was stable but only
slightly positive.

## Scope

- YOLOv8n backbone, neck, and head unchanged.
- Regularized LiteViT block after SPPF/P5 (layer 9):
  - 4 heads;
  - MLP ratio 1.0;
  - DropPath 0.2;
  - dropout 0.1;
  - zero-initialized LayerScale;
  - FP32 block execution under AMP.
- Weather gate on P3/P4/P5 after the neck (layers 15/18/21):
  - C/4 reduction, SiLU, `2·sigmoid`, zero-initialized final layer.
- Same protocol as `EXP-0001`.

## Size

| Component | Parameters |
|---|---:|
| YOLOv8n 5-class baseline | 3,138,975 |
| Regularized P5 block | 401,408 |
| P3/P4/P5 gates | 43,568 |
| **Total** | **3,583,951** |

GFLOPs at 640: **9.099**.

## Setup

| Item | Value |
|---|---|
| Config | `configs/train/gate_p5.yaml` |
| Architecture | `arch: gate_p5` |
| Pretrained weights | `yolov8n.pt` |
| Dataset | `adverse16k` |
| Seed | 42 |
| Hardware | RTX 3050 6 GB |
| Train run | `runs/train/gate_p5_seed42` |
| Eval run | `runs/eval/gate_p5_seed42` |

## Commands

```bash
# Optional 5-epoch smoke run
.venv/bin/python scripts/train.py \
  --config configs/train/gate_p5.yaml \
  --name gate_p5_smoke --epochs 5

# Full A+B run
.venv/bin/python scripts/train.py \
  --config configs/train/gate_p5.yaml \
  --name gate_p5_seed42

.venv/bin/python scripts/evaluate.py \
  --weights runs/train/gate_p5_seed42/weights/best.pt \
  --data configs/datasets/adverse16k.yaml \
  --splits val test \
  --out runs/eval/gate_p5_seed42 \
  --subsets data/processed/adverse16k/subsets \
  --figures-dir thesis/results/figures/gate_p5

.venv/bin/python scripts/error_analysis.py \
  --weights runs/train/gate_p5_seed42/weights/best.pt \
  --out runs/eval/gate_p5_seed42/error_analysis \
  --splits val test

.venv/bin/python scripts/document_results.py \
  --train-run runs/train/gate_p5_seed42 \
  --eval-run runs/eval/gate_p5_seed42 \
  --results thesis/results/gate_p5.md \
  --experiment thesis/experiments/EXP-0005-gate-p5.md \
  --figures-dir thesis/results/figures/gate_p5 \
  --vram-report runs/train/gate_p5_seed42/vram.json \
  --title "A+B Results — Gate + Regularized P5 Attention" \
  --experiment-title "EXP-0005 — Gate + Regularized P5 Attention"
```

## Hypothesis and checks

- Initial output should match the baseline exactly.
- Saved `best.pt` must contain `FeatureWithBlock` at layer 9 and
  `FeatureWithGate` at 15/18/21, with about **3.584 M** parameters.
- Compare against baseline, V1, and gate-only per condition/class/size.
- If A+B does not beat gate-only and baseline on validation, keep gate-only and
  report P5 attention as a negative ablation.

## Status

Planned. Training and evaluation are pending.
