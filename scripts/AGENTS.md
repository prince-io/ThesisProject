# scripts/ — Entrypoints and Utilities

## Purpose

Hold thin CLI entrypoints for training, evaluation, export, and data preparation.

## Ownership

- Authoritative for command-line usage.

## Local Contracts

- Scripts are thin wrappers over `litevit`; business logic lives in `src/litevit/`.
- No hardcoded paths; accept config/CLI arguments.
- Each script documents usage in its docstring.

## Work Guidance

- `build_adverse16k.py`, `verify_adverse16k.py`, `audit_adverse16k.py`, `prepare_acdc_test.py` — dataset.
- `train.py`, `make_subsets.py`, `evaluate.py`, `error_analysis.py`, `document_results.py`, `export.py` — baseline training/evaluation.

## Verification

- `.venv/bin/ruff check .`

## Child DOX Index

- (none)
