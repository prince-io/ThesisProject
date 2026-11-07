---
title: EXP-0002 — cls_pw=0.0 Ablation (pending)
status: draft
date: 2026-11-07
sources:
  - configs/train/ablation_cls_pw0.yaml
  - thesis/decisions/ADR-0006-baseline-protocol.md
---

# EXP-0002 — cls_pw=0.0 Ablation (pending)

## Objective

Measure the effect of disabling class weighting relative to the baseline (`cls_pw=0.5`).

## Setup

- Config: `configs/train/ablation_cls_pw0.yaml`
- All settings match `configs/train/baseline.yaml` except `cls_pw: 0.0`.
- Train run: `runs/train/ablation_cls_pw0_seed42`
- Eval run: `runs/eval/ablation_cls_pw0_seed42`
- Seed: 42; 100 epochs; batch 4; 640; AMP; AdamW 2e-4 + cosine.
- Dataset: `adverse16k` (checksum in the run metadata).

## Commands

```bash
.venv/bin/python scripts/train.py \
  --config configs/train/ablation_cls_pw0.yaml \
  --name ablation_cls_pw0_seed42

.venv/bin/python scripts/evaluate.py \
  --weights runs/train/ablation_cls_pw0_seed42/weights/best.pt \
  --data configs/datasets/adverse16k.yaml \
  --splits val test \
  --out runs/eval/ablation_cls_pw0_seed42 \
  --subsets data/processed/adverse16k/subsets \
  --figures-dir thesis/results/figures/ablation_cls_pw0

.venv/bin/python scripts/error_analysis.py \
  --weights runs/train/ablation_cls_pw0_seed42/weights/best.pt \
  --out runs/eval/ablation_cls_pw0_seed42/error_analysis \
  --splits val test

.venv/bin/python scripts/document_results.py \
  --train-run runs/train/ablation_cls_pw0_seed42 \
  --eval-run runs/eval/ablation_cls_pw0_seed42 \
  --results thesis/results/ablation_cls_pw0.md \
  --experiment thesis/experiments/EXP-0002-cls-pw0.md \
  --figures-dir thesis/results/figures/ablation_cls_pw0 \
  --vram-report runs/train/ablation_cls_pw0_seed42/vram.json \
  --title "Ablation Results — cls_pw=0.0" \
  --experiment-title "EXP-0002 — cls_pw=0.0 Ablation"
```

## Results

Pending. After the run, `scripts/document_results.py` will replace this section with
the generated record and write `thesis/results/ablation_cls_pw0.md`.
