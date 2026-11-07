# configs/datasets/ — Dataset Configs

## Purpose

Hold Ultralytics dataset configs for training and evaluation.

## Ownership

- Authoritative for dataset paths, splits, and class names.

## Local Contracts

- `adverse16k.yaml` points to `data/processed/adverse16k` with splits `images/{train,val,test}`.
- Class order is fixed: `0 person, 1 car, 2 bus, 3 bicycle, 4 motorcycle`.
- Do not change class ids without an ADR and a full dataset rebuild.

## Work Guidance

- Keep `path` relative to the repository root.

## Verification

- `.venv/bin/python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('configs/datasets/adverse16k.yaml')"`

## Child DOX Index

- (none)
