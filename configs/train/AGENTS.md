# configs/train/ — Training Configs

## Purpose

Hold training configurations for baseline and proposed models.

## Ownership

- Authoritative for training hyperparameters.

## Local Contracts

- One config per experiment; immutable once used for a recorded run.
- Every config must be traceable from a `thesis/experiments/` record.
- Keep the hardware contract explicit: 640x640, batch 4, AMP.
- Dataset paths stay relative to the repository root.

## Work Guidance

- `baseline.yaml` defines the YOLOv8n reference protocol, including `cls_pw: 0.5`.
- `ablation_cls_pw0.yaml` is the class-weighting ablation (`cls_pw: 0.0`); all other settings match the baseline.
- `litevit_v1_p5.yaml` is the V1 P5-only LiteViT run (`arch: v1_p5`); all other settings match the baseline.
- `weather_gate.yaml` is the gate-only run (`arch: gate_only`); all other settings match the baseline.
- `gate_p2.yaml` is gate-only plus a standard stride-4 P2 detection head (`arch: gate_p2`); all other settings match the baseline.
- `gate_p5.yaml` is A+B: gate plus regularized P5 attention (`arch: gate_p5`); all other settings match the baseline.
- `gate_edge_cbam.yaml` is the kitchen-sink model (`arch: gate_edge_cbam`); all other settings match the baseline.
- Start new variants from the baseline config and change one factor at a time.

## Verification

- `.venv/bin/python -c "from litevit.training.config import load_config; load_config('configs/train/baseline.yaml')"`

## Child DOX Index

- (none)
