---
title: Project Overview
status: draft
date: 2026-11-07
sources:
  - ref/idea.md
  - experiments/dataset-adverse16k.md
---

# Project Overview

## Objective

B.Tech final-year thesis on lightweight object detection robust to adverse weather (fog, rain, snow, night).

## Proposed architecture

YOLO-LiteViT-WG: YOLOv8n backbone + linear-attention ViT blocks at neck P4/P5 + lightweight weather channel gate. Proposed in `ref/idea.md`; adoption recorded in `decisions/ADR-0002-yolo-litevit-wg-architecture.md`.

## Hard constraints

- Train stably at 640x640 on an RTX 3050 6GB.
- Batch size 4, mixed precision (AMP).
- Target ~3.5–4.0 M parameters.

## Tooling

Python 3.12 + PyTorch 2.14.1 (CUDA 13.0) + torchvision + Ultralytics + timm/einops, aligned to the base paper's PyTorch ecosystem. See `decisions/ADR-0003-python-pytorch-stack.md`.

## Dataset

`adverse16k`: 16,000 images, 4,000 per condition (fog, rain, snow, night), 5 classes (`person, car, bus, bicycle, motorcycle`), split 12,800/1,600/1,600. Built from ACDC + DAWN + BDD100K + RTTS as an independent copy. A separate 2,000-image unlabeled ACDC test set is kept for demos and final qualitative testing. See `experiments/dataset-adverse16k.md` and `decisions/ADR-0004-adverse16k-dataset.md`.

## Non-goals

- Methodology content is not written until implementation starts.

## Open questions

- Mitigation for class imbalance (car dominates) during training — deferred.
- Whether to seed methodology from `ref/idea.md` — deferred.
