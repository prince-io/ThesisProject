# tests/ — Verification Tests

## Purpose

Hold automated verification for the `litevit` package.

## Ownership

- Authoritative for test coverage.

## Local Contracts

- pytest; tests live under `tests/` and import `litevit`.
- Tests must not require the dataset or GPU unless explicitly marked.
- Mark slow/GPU tests so CPU-only runs stay fast.

## Work Guidance

- Add a test for each new public behavior.
- Dataset class mapping (including BDD legacy aliases), box conversion, condition assignment, sequence assignment, dHash grouping, and split arithmetic are covered in `test_dataset_pipeline.py` and `test_acdc_test.py`.
- Training config validation, eval-subset generation, and metric aggregation are covered in `test_training.py`.

## Verification

- `.venv/bin/pytest`

## Child DOX Index

- (none)
