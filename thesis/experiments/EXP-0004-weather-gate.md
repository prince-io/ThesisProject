---
title: EXP-0004 — Gate-Only Weather Adaptation
status: draft
date: 2026-11-09
sources:
  - configs/train/weather_gate.yaml
  - thesis/methodology/weather-gate.md
  - thesis/decisions/ADR-0009-weather-gate-design.md
---

# EXP-0004 — Gate-Only Weather Adaptation

## Objective

Test whether a lightweight channel gate after the YOLOv8n neck improves
adverse-weather detection without adding attention capacity. This isolates the
weather-adaptation mechanism from the V1 P5 attention block.

## Scope

- YOLOv8n backbone, neck, and head unchanged.
- Lightweight Weather Gate after the neck on P3, P4, and P5 (layers 15/18/21).
- Gate: GAP → C/4 → C → `2·sigmoid`, final layer zero-initialized.
- No spatial attention, no resolution change, no layer-index change.
- Same training/evaluation protocol as `EXP-0001`.

## Why these values

| Choice | Value | Reason |
|---|---|---|
| Reduction | C/4 | P3 has 64 channels; C/16 would leave hidden width 4. C/4 gives 16/32/64. |
| Activation | SiLU | YOLO-native; no extra parameters. |
| Gate range | 2·sigmoid | Exactly 1 at zero init; bounded 0–2; can suppress or amplify. |
| Init | zero final weight/bias | Identity at start. |
| Placement | P3/P4/P5 after neck | Weather affects all scales; P3 protects small objects. |

Parameters: **43,568** gate parameters; **3,182,543** total five-class parameters
(+1.4% over baseline). GFLOPs and VRAM impact are negligible.

## Setup

| Item | Value |
|---|---|
| Config | `configs/train/weather_gate.yaml` |
| Architecture | `arch: gate_only` |
| Pretrained weights | `yolov8n.pt` |
| Dataset | `adverse16k` |
| Seed | 42 |
| Hardware | RTX 3050 6 GB |
| Train run | `runs/train/weather_gate_seed42` |
| Eval run | `runs/eval/weather_gate_seed42` |

Training protocol is unchanged: 640, batch 4, AMP, `nbs=8`, AdamW 2e-4, cosine,
100 epochs, patience 20, `cls_pw=0.5`.

## Commands

```bash
# Optional 5-epoch smoke run
.venv/bin/python scripts/train.py \
  --config configs/train/weather_gate.yaml \
  --name weather_gate_smoke --epochs 5

# Full gate-only run
.venv/bin/python scripts/train.py \
  --config configs/train/weather_gate.yaml \
  --name weather_gate_seed42

.venv/bin/python scripts/evaluate.py \
  --weights runs/train/weather_gate_seed42/weights/best.pt \
  --data configs/datasets/adverse16k.yaml \
  --splits val test \
  --out runs/eval/weather_gate_seed42 \
  --subsets data/processed/adverse16k/subsets \
  --figures-dir thesis/results/figures/weather_gate

.venv/bin/python scripts/error_analysis.py \
  --weights runs/train/weather_gate_seed42/weights/best.pt \
  --out runs/eval/weather_gate_seed42/error_analysis \
  --splits val test

.venv/bin/python scripts/document_results.py \
  --train-run runs/train/weather_gate_seed42 \
  --eval-run runs/eval/weather_gate_seed42 \
  --results thesis/results/weather_gate.md \
  --experiment thesis/experiments/EXP-0004-weather-gate.md \
  --figures-dir thesis/results/figures/weather_gate \
  --vram-report runs/train/weather_gate_seed42/vram.json \
  --title "Weather Gate Results" \
  --experiment-title "EXP-0004 — Gate-Only Weather Adaptation"
```

## Hypothesis and checks

- Initial output should match the baseline exactly because the gate is an identity.
- Training should be stable; saved `best.pt` must contain `FeatureWithGate` at
  layers 15/18/21 and about 3.183 M parameters.
- Compare per-condition, per-class, and per-size metrics with `EXP-0001`.
- If the gate helps, later combine it with a regularized P4/P5 attention block.

## Status

Planned. Training and evaluation are pending.
