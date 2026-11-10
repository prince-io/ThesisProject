# configs/ — Training and Experiment Configs

## Purpose

Hold YAML configs for datasets, models, training, and evaluation.

## Ownership

- Authoritative for run settings.

## Local Contracts

- One config per experiment; immutable once used for a recorded run.
- YAML; reference paths explicitly, never inline secrets.
- Each config must be traceable from a `thesis/experiments/` record.
- Keep hardware limits explicit: 640x640, batch 4, AMP.

## Work Guidance

- Dataset configs exist for `adverse16k`; `train/baseline.yaml` defines the YOLOv8n reference protocol.
- `train/litevit_v1_p5.yaml` defines the V1 P5-only LiteViT run (`arch: v1_p5`).
- `train/weather_gate.yaml` defines the gate-only run (`arch: gate_only`).
- `train/gate_p2.yaml` defines gate-only plus a stride-4 P2 detection head (`arch: gate_p2`).
- `train/gate_p5.yaml` defines A+B: gate plus regularized P5 attention (`arch: gate_p5`).
- `train/gate_edge_cbam.yaml` defines the gate + Sobel edge + spatial CBAM model.

## Verification

- No automated checks.

## Child DOX Index

- `datasets/AGENTS.md` — dataset path and class-name contracts.
- `train/AGENTS.md` — training hyperparameter contracts.
