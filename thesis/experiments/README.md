---
title: Experiment Registry
status: validated
date: 2026-11-10
sources:
  - thesis/experiments/EXP-0001-baseline.md
  - thesis/experiments/EXP-0002-cls-pw0.md
  - thesis/experiments/EXP-0003-v1-p5.md
  - thesis/experiments/EXP-0004-weather-gate.md
  - thesis/experiments/EXP-0005-gate-p5.md
  - thesis/experiments/EXP-0006-gate-edge-cbam.md
  - thesis/experiments/EXP-0007-gate-p2.md
---

# Experiment Registry

| ID | Config | Train run | Eval run | Status |
|---|---|---|---|---|
| EXP-0001 | `configs/train/baseline.yaml` | `runs/train/baseline_yolov8n_seed42` | `runs/eval/baseline_yolov8n_seed42` | validated |
| EXP-0002 | `configs/train/ablation_cls_pw0.yaml` | `runs/train/ablation_cls_pw0_seed42` | `runs/eval/ablation_cls_pw0_seed42` | validated |
| EXP-0003 | `configs/train/litevit_v1_p5.yaml` | `runs/train/litevit_v1_p5_seed42_v2` | `runs/eval/litevit_v1_p5_seed42_v2` | validated |
| EXP-0004 | `configs/train/weather_gate.yaml` | `runs/train/weather_gate_seed42` | `runs/eval/weather_gate_seed42` | validated |
| EXP-0005 | `configs/train/gate_p5.yaml` | `runs/train/gate_p5_seed42` | `runs/eval/gate_p5_seed42` | validated |
| EXP-0006 | `configs/train/gate_edge_cbam.yaml` | `runs/train/gate_edge_cbam_seed42` | `runs/eval/gate_edge_cbam_seed42` | validated |
| EXP-0007 | `configs/train/gate_p2.yaml` | `runs/train/gate_p2_seed42` | `runs/eval/gate_p2_seed42` | planned |

EXP-0003 predecessor `runs/train/litevit_v1_p5_seed42` is invalid: the LiteViT
wrapper was discarded before training, so it trained stock YOLOv8n. It is retained
for audit only.
