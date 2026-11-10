# experiments/ — Curated Dataset and Run Records

## Purpose

Hold curated records of dataset preparation and experiments: what was built/run and under which conditions.

## Ownership

- Authoritative for experimental setup; raw artifacts stay in `runs/`.

## Local Contracts

- One record per dataset build or experiment, YAML front matter (`title`, `status`, `date`, `sources`).
- Dataset records include sources, counts, class map, split policy, and the verification report path.
- Run records include objective, config path, seed, dataset, hardware, and a run ID pointing to `runs/<id>/`.
- Do not paste raw logs or full metric dumps; link them.
- Do not include interpretation; that belongs in `../findings/`.

## Work Guidance

- `dataset-adverse16k.md` records the current benchmark; `dataset-card-adverse16k.md` is its datasheet.
- `EXP-0001-baseline.md` records the YOLOv8n reference run; `EXP-0002-cls-pw0.md` records the completed class-weighting ablation.
- `EXP-0003-v1-p5.md` records the corrected V1 P5-only LiteViT experiment; the invalid predecessor is documented in `../findings/v1-invalid-run.md`.
- `EXP-0004-weather-gate.md` records the gate-only weather-adaptation experiment.
- `EXP-0005-gate-p5.md` records A+B: gate plus regularized P5 attention.
- `EXP-0006-gate-edge-cbam.md` records the gate + Sobel edge + spatial CBAM model.
- `EXP-0007-gate-p2.md` records gate-only plus a stride-4 P2 detection head.
- `README.md` is the experiment registry; update it when a run is added or completes.

## Verification

- No automated checks.

## Child DOX Index

- (none)
