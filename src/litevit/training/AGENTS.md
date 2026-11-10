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

- `train.py` wraps Ultralytics training, selects `baseline`, `v1_p5`, `gate_only`, `gate_p2`, `gate_p5`, or `gate_edge_cbam` from the config `arch`, verifies the saved architecture, writes run metadata, records labeled VRAM metrics, and logs training exceptions to `train_error.txt`.
- `trainer.py` implements `LiteDetectionTrainer`, `V1DetectionTrainer`, `GateDetectionTrainer`, `GateP2DetectionTrainer`, `GateP5DetectionTrainer`, and `GateEdgeCBAMTrainer`, which build and wrap variants inside Ultralytics `get_model` so wrappers survive trainer setup; it logs post-training validation/plot failures and falls back to live validation.
- `subsets.py` derives per-condition/per-source lists from `manifest.csv`.
- `evaluate.py` uses Ultralytics validation and saves subset metrics.
- `error_analysis.py` matches predictions to GT and reports per-condition/source/class/size errors.

## Verification

- `.venv/bin/ruff check .`
- `.venv/bin/pytest`
- `.venv/bin/python scripts/make_subsets.py --manifest data/processed/adverse16k/manifest.csv --out data/processed/adverse16k/subsets`

## Child DOX Index

- (none)
