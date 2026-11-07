# litevit/datasets/ — Dataset Preparation

## Purpose

Load raw ACDC, DAWN, BDD100K, and RTTS annotations, build the balanced `adverse16k` benchmark, and prepare the unlabeled ACDC test set.

## Ownership

- Authoritative for dataset construction, class mapping, and split policy.

## Local Contracts

- Five classes: `person`, `car` (truck merged), `bus`, `bicycle`, `motorcycle`; `rider -> person`.
- Four conditions: fog, rain, snow, night; assignment is mutually exclusive.
- Deterministic seed 42; **sequence-aware splits** for ACDC/RTTS and near-duplicate grouping for DAWN; fill shortfalls from RTTS (fog) or BDD100K (rain/snow/night).
- BDD labels must accept both modern and legacy aliases (`person`, `bike`, `motor`).
- Processed images and the ACDC test copy are independent files (link count 1); the raw `ref/` datasets are retained for now.
- ACDC test has no ground truth and is for demos/qualitative testing only.
- No import-time file I/O; raw paths are passed by the CLI/build entrypoint.

## Work Guidance

- Keep one loader per source; shared types, maps, and serializers live in `common.py`.
- `build.py` owns allocation; `verify.py` owns verification gates.

## Verification

- `.venv/bin/python scripts/build_adverse16k.py --root . --out data/processed/adverse16k --force --copy-images`
- `.venv/bin/python scripts/verify_adverse16k.py --root . --out data/processed/adverse16k --report runs/dataset_checks/adverse16k --require-independent --checksums`
- `.venv/bin/python scripts/audit_adverse16k.py --root . --out data/processed/adverse16k --report runs/dataset_checks/adverse16k`
- `.venv/bin/python scripts/prepare_acdc_test.py --root . --out data/unlabeled/acdc_test --report runs/dataset_checks/acdc_test --force`
- `.venv/bin/pytest`

## Child DOX Index

- (none)
