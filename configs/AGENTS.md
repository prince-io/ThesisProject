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

- (none yet)

## Verification

- No automated checks.

## Child DOX Index

- (none)
