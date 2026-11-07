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
- Start new variants from the baseline config and change one factor at a time.

## Verification

- `.venv/bin/python -c "from litevit.training.config import load_config; load_config('configs/train/baseline.yaml')"`

## Child DOX Index

- (none)
