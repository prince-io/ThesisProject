---
title: Project Overview
status: draft
date: 2026-11-10
sources:
  - ref/idea.md
  - experiments/dataset-adverse16k.md
  - experiments/EXP-0001-baseline.md
  - experiments/EXP-0002-cls-pw0.md
  - experiments/EXP-0003-v1-p5.md
  - experiments/EXP-0004-weather-gate.md
  - experiments/EXP-0005-gate-p5.md
  - results/baseline.md
  - results/ablation-comparison.md
  - findings/baseline-analysis.md
  - findings/ablation-analysis.md
  - findings/v1-invalid-run.md
  - findings/v1-p5-analysis.md
  - results/litevit_v1_p5.md
  - results/litevit_v1_p5-error-analysis.md
  - results/weather_gate.md
  - results/weather_gate-error-analysis.md
  - results/architecture-comparison.md
  - findings/weather-gate-analysis.md
  - decisions/ADR-0008-freeze-class-weighting.md
  - related-work/architecture-grounding.md
  - bibliography/references.bib
  - methodology/architecture.md
  - methodology/weather-gate.md
  - methodology/gate-p5.md
  - decisions/ADR-0009-weather-gate-design.md
  - decisions/ADR-0010-gate-p5-design.md
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

`adverse16k`: 16,000 images, 4,000 per condition (fog, rain, snow, night), 5 classes (`person, car, bus, bicycle, motorcycle`), split 12,800/1,600/1,600. Built from ACDC + DAWN + BDD100K + RTTS as an independent copy with sequence-aware splits (zero sequence leakage). A separate 2,000-image unlabeled ACDC test set is kept for demos and final qualitative testing. See `experiments/dataset-adverse16k.md`, `experiments/dataset-card-adverse16k.md`, `decisions/ADR-0004-adverse16k-dataset.md`, and `decisions/ADR-0005-sequence-aware-splits.md`.

## Baseline status

YOLOv8n baseline (`EXP-0001`, seed 42, `cls_pw=0.5`): val mAP50 0.486 / mAP50-95 0.290;
test mAP50 0.543 / mAP50-95 0.329. Fog is strongest; night/rain/snow and ACDC are
weakest. Error analysis is in `results/baseline-error-analysis.md`; interpretation is in
`findings/baseline-analysis.md`.

## Ablation status

`EXP-0002` (`cls_pw=0.0`, seed 42) is complete: val mAP50 0.487 / mAP50-95 0.291;
test mAP50 0.531 / mAP50-95 0.325. Validation is effectively tied, test mAP is
lower than the baseline, and rare-class effects are mixed. At confidence 0.25 the
ablation has slightly higher precision/recall, so the result is a
calibration/ranking trade-off rather than a clear gain. Facts are in
`results/ablation-comparison.md`; interpretation is in `findings/ablation-analysis.md`.
`cls_pw=0.5` is retained for LiteViT-WG (`ADR-0008`).

## Literature status

The architecture literature-grounding pass is complete. `bibliography/references.bib`
holds verified entries; `related-work/architecture-grounding.md` separates supported
design choices from assumptions. The key conclusion is that P4/P5 global context,
linear attention, CPE, stable residual init, and channel gating are each
literature-backed, but the exact YOLO-LiteViT-WG combination and expected gains are
hypotheses. `ref/idea.md` claims about “C2f-ViT” remain unverified; YOLO-CasViT is
the verified closest work.

## Architecture status

V1 (one identity-initialized linear-attention block after P5/SPPF, before PAN/FPN)
is implemented in `src/litevit/models/` and configured in
`configs/train/litevit_v1_p5.yaml`. The first run was invalid because Ultralytics
discarded the wrapper; `V1DetectionTrainer` now builds V1 inside trainer setup and
verifies the saved checkpoint. The FP32 block passed a 10-epoch stability test, and the full V1 run early-stopped at
epoch 38 with best epoch 18. V1 improved slightly over the baseline at the same epoch
but overfit afterwards and finished below the baseline on test; see
`results/litevit_v1_p5.md`.

Gate-only (L-WG on P3/P4/P5) is complete (`EXP-0004`): test mAP50 0.5444 vs baseline
0.5428 and mAP50-95 0.3301 vs 0.3287; validation is slightly lower, but training is
stable for 100 epochs and night/rain plus bicycle/bus show small gains. See
`results/weather_gate.md`, `results/architecture-comparison.md`, and
`findings/weather-gate-analysis.md`.

A+B (gate + regularized P5 attention: DropPath 0.2, dropout 0.1, MLP ratio 1.0) is
implemented and pending training (`EXP-0005`). It has 3.584 M parameters and
9.099 GFLOPs. V2 (P4/P5 attention) is not implemented.

## Non-goals

- Final LiteViT/weather-gate design is not fixed until the Stage 2B technical spike.

## Open questions

- A+B (gate + regularized P5 attention) training/evaluation — pending (`EXP-0005`).
- Multi-seed validation of gate-only/A+B — pending.
- Multi-seed validation for final claims — pending.
- Whether the test split should be reserved for the final model after Stage 2A reuse — pending.
