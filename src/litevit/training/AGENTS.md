# litevit/training/ — Training and Evaluation

## Purpose

Run training jobs, build evaluation subsets, and evaluate detectors overall and per condition/source.

## Ownership

- Authoritative for training/evaluation behavior and metric aggregation.

## Local Contracts

- Configs come from `configs/train/`; dataset configs from `configs/datasets/`.
- Every run writes `run_meta.json` (git commit, versions, GPU, config, checksums).
- Evaluation writes `metrics.json`, `per_class.csv`, `per_condition.csv`, and `per_source.csv`.
- Results are facts only; interpretation belongs in `thesis/findings/`.

## Work Guidance

- `train.py` wraps Ultralytics training, writes run metadata, and records labeled VRAM metrics.
- `subsets.py` derives per-condition/per-source lists from `manifest.csv`.
- `evaluate.py` uses Ultralytics validation and saves subset metrics.
- `error_analysis.py` matches predictions to GT and reports per-condition/source/class/size errors.

## Verification

- `.venv/bin/ruff check .`
- `.venv/bin/pytest`
- `.venv/bin/python scripts/make_subsets.py --manifest data/processed/adverse16k/manifest.csv --out data/processed/adverse16k/subsets`

## Child DOX Index

- (none)
