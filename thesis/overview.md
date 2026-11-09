---
title: Project Overview
status: draft
date: 2026-11-08
sources:
  - ref/idea.md
  - experiments/dataset-adverse16k.md
  - experiments/EXP-0001-baseline.md
  - experiments/EXP-0002-cls-pw0.md
  - experiments/EXP-0003-v1-p5.md
  - results/baseline.md
  - results/ablation-comparison.md
  - findings/baseline-analysis.md
  - findings/ablation-analysis.md
  - findings/v1-invalid-run.md
  - decisions/ADR-0008-freeze-class-weighting.md
  - related-work/architecture-grounding.md
  - bibliography/references.bib
  - methodology/architecture.md
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
verifies the saved checkpoint. The FP32 block passed a 10-epoch stability test with finite loss and a correct
3.674 M block; the wrapper failed after training before metadata, and fallback
logging was added. A repeat test and full run are pending (`EXP-0003`). V2 (P4 block) and V3 (weather gate) are not implemented.

## Non-goals

- Final LiteViT/weather-gate design is not fixed until the Stage 2B technical spike.

## Open questions

- V1 P5-only LiteViT training and evaluation — pending.
- V2/V3 architecture and memory feasibility — pending V1 results.
- Multi-seed validation for final claims — pending.
- Whether the test split should be reserved for the final model after Stage 2A reuse — pending.
