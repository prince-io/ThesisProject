---
title: ADR-0006 — Baseline Training and Evaluation Protocol
status: accepted
date: 2026-11-07
sources:
  - thesis/experiments/dataset-adverse16k.md
  - thesis/experiments/dataset-card-adverse16k.md
---

# ADR-0006 — Baseline Training and Evaluation Protocol

## Context

The proposed YOLO-LiteViT-WG architecture needs a reference baseline trained and evaluated under the same hardware and data conditions. The thesis must compare variants fairly and document every result.

## Decision

Use YOLOv8n (`yolov8n.pt`, COCO-pretrained, `cls_remap=True`) as the reference detector.

Training protocol:

- 640x640, batch 4, AMP; `nbs=8` (gradient accumulation 2).
- AdamW, `lr0=2e-4`, cosine schedule, 3 warmup epochs, weight decay 0.0005.
- 100 epochs, patience 20.
- Mosaic 0.5, MixUp 0.05, `close_mosaic=10`.
- Seed 42, deterministic, 4 dataloader workers.
- **Class weighting:** `cls_pw=0.5` (inverse-frequency weights, square-root dampened).
- Dataset: `configs/datasets/adverse16k.yaml`.

Evaluation protocol:

- Overall val/test: precision, recall, mAP50, mAP50-95.
- Per-class AP50 and AP50-95.
- Per-condition (fog/rain/snow/night) and per-source (ACDC/DAWN/BDD100K/RTTS).
- Efficiency: parameters, GFLOPs, latency, FPS, peak VRAM.
- Save full artifacts under `runs/`; curated facts under `thesis/results/`.

## Consequences

- The baseline is class-weighted; a `cls_pw=0.0` ablation is required to isolate the effect.
- Memory labeling and error-analysis details are refined by `ADR-0007-measurement-and-analysis-protocol.md`.
- All later variants must use the same protocol for a fair comparison.
- Results are not directly comparable to the official ACDC/RTTS splits because the benchmark uses sequence-aware re-splits.
- One seed is used for Stage 2A; the final model should use three seeds if time permits.

## Status

Accepted.

## Supersedes

None.
