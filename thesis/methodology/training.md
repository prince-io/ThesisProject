---
title: Training and Evaluation Protocol
status: draft
date: 2026-11-07
sources:
  - thesis/decisions/ADR-0006-baseline-protocol.md
  - thesis/experiments/dataset-adverse16k.md
---

# Training and Evaluation Protocol

## Data

`adverse16k`: 16,000 images, 4,000 per condition (fog, rain, snow, night), 5 classes,
split 12,800/1,600/1,600. The dataset is an independent copy with sequence-aware splits
and checksums; see `thesis/experiments/dataset-card-adverse16k.md`.

## Reference model

YOLOv8n, COCO-pretrained, with class-head remapping to the 5 target classes.

## Training

- Input 640x640; batch 4; AMP on an RTX 3050 6 GB.
- Nominal batch 8 (gradient accumulation 2).
- AdamW, initial LR 2e-4, cosine schedule, 3 warmup epochs, weight decay 5e-4.
- 100 epochs, early stopping patience 20.
- Mosaic 0.5, MixUp 0.05, mosaic disabled for the final 10 epochs.
- Deterministic training, seed 42.
- Class weighting: `cls_pw=0.5`, i.e. inverse-frequency weights raised to the power 0.5
  and normalized to mean 1. This dampens the severe car dominance.

## Evaluation

- Overall val/test precision, recall, mAP50, mAP50-95.
- Per-class AP50 and AP50-95, with target counts to expose low-support classes.
- Per-condition and per-source subsets.
- Efficiency: parameters, GFLOPs, inference latency, FPS, peak VRAM.
- Full artifacts in `runs/`; curated facts in `thesis/results/`; interpretation in
  `thesis/findings/`.

## Reproducibility

- Config: `configs/train/baseline.yaml`.
- Dataset checksums: `data/processed/adverse16k/checksums.sha1`.
- Run metadata records git commit, package versions, GPU, and config.
- Commands are recorded in the experiment record.
