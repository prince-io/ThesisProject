---
title: Experiment Registry
status: validated
date: 2026-11-08
sources:
  - thesis/experiments/EXP-0001-baseline.md
  - thesis/experiments/EXP-0002-cls-pw0.md
  - thesis/experiments/EXP-0003-v1-p5.md
---

# Experiment Registry

| ID | Config | Train run | Eval run | Status |
|---|---|---|---|---|
| EXP-0001 | `configs/train/baseline.yaml` | `runs/train/baseline_yolov8n_seed42` | `runs/eval/baseline_yolov8n_seed42` | validated |
| EXP-0002 | `configs/train/ablation_cls_pw0.yaml` | `runs/train/ablation_cls_pw0_seed42` | `runs/eval/ablation_cls_pw0_seed42` | validated |
| EXP-0003 | `configs/train/litevit_v1_p5.yaml` | `runs/train/litevit_v1_p5_seed42` | `runs/eval/litevit_v1_p5_seed42` | planned |
