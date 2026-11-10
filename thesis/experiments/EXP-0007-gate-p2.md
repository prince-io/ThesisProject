---
title: EXP-0007 — Gate-Only + P2 High-Resolution Head
status: draft
date: 2026-11-10
sources:
  - configs/train/gate_p2.yaml
  - thesis/decisions/ADR-0012-p2-head-design.md
  - thesis/methodology/p2-head.md
  - src/litevit/models/p2.py
---

# EXP-0007 — Gate-Only + P2 High-Resolution Head

## Objective

Measure whether a stride-4 P2 detection level improves tiny-object detection on
`adverse16k` when added to the gate-only model.

## Setup

| Item | Value |
|---|---|
| Architecture | `gate_p2` |
| Base | gate-only, P3/P4/P5 weather gates |
| Change | P2/P3/P4/P5 Detect head, strides 4/8/16/32 |
| Params | 3,270,388 |
| GFLOPs at 640 | 13.3 |
| Dataset | `adverse16k`, checksum `7de437da75a1698d73195565cfff0c2b3e83da48` |
| Config | `configs/train/gate_p2.yaml` |
| Seed | 42 |
| Epochs | 100, patience 20 |
| Batch | 4 |
| AMP | True |
| `cls_pw` | 0.5 |
| Hardware | RTX 3050 6 GB Laptop GPU |

P2 is not identity-initialized. Pretrained P3/P4/P5 branches are preserved; the
P2 branch is new. See `../methodology/p2-head.md`.

## Commands

```bash
.venv/bin/python scripts/train.py --config configs/train/gate_p2.yaml --name gate_p2_smoke --epochs 5
.venv/bin/python scripts/train.py --config configs/train/gate_p2.yaml --name gate_p2_seed42
```

Evaluation and documentation commands are in the project `README.md` and root
`AGENTS.md`.

## Preflight

A 640x640 batch-4 smoke run (1 epoch, 0.2% of training images) passed architecture
verification, checkpoint reload, and half-precision final validation. Peak GPU
memory was **1,595 MB**; no OOM or NaN occurred. The smoke metrics are not results.

## Status

Pending. Results will be recorded in `../results/gate_p2.md` only after the run.
