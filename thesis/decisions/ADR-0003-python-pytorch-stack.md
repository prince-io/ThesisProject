---
title: ADR-0003 — Python and PyTorch Stack Aligned to the Base Paper
status: accepted
date: 2026-11-07
sources:
  - ref/Real_Time_Object_Detection_in_Adverse_Weather_Conditions_Using_Transformer.pdf
---

# ADR-0003 — Python and PyTorch Stack Aligned to the Base Paper

## Context

The base paper (Nie & Wang, 2025) uses PyTorch 2.1, CUDA 12.1, and cuDNN 8.8 with ViT/ResNet backbones, COCO-format datasets, COCO mAP, and TensorRT/ONNX deployment. This project must stay in the same ecosystem but train on an RTX 3050 6 GB.

## Decision

- Python 3.12 virtualenv at `.venv/`.
- PyTorch 2.14.1 + torchvision 0.29.1 from the official CUDA 13.0 wheel index (CUDA 13.0 runtime; driver is CUDA 13.2).
- Ultralytics 8.4.x for the YOLOv8 backbone/head; timm + einops for ViT components.
- COCO evaluation via pycocotools; metrics via torchmetrics.
- TensorBoard logging; ONNX + ONNX Runtime installed now; TensorRT deferred to deployment.
- Custom OpenCV/numpy weather augmentations, matching the base paper's bespoke pipeline; albumentations deferred to avoid an OpenCV duplicate-package conflict.

Modern versions are chosen over the paper's exact pins because PyTorch 2.1 has no Python 3.12 wheels and current Ultralytics/timm require newer PyTorch.

## Consequences

- Reproducible via `pyproject.toml` and `.venv/`.
- Exact paper versions are not reproduced; API-level parity is preserved.
- CUDA 13.0 wheels run on the installed CUDA 13.2 driver.
- If albumentations is needed later, resolve the OpenCV packaging conflict first.

## Status

Accepted.

## Supersedes

None.
