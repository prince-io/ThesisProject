# experiments/ — Curated Run Records

## Purpose

Hold curated records of experiments: what was run and under which conditions.

## Ownership

- Authoritative for experimental setup; raw artifacts stay in `runs/`.

## Local Contracts

- One record per experiment, YAML front matter (`title`, `status`, `date`, `sources`).
- Include objective, config path, seed, dataset, hardware, and a run ID pointing to `runs/<id>/`.
- Do not paste raw logs or full metric dumps; link them.
- Do not include interpretation; that belongs in `../findings/`.

## Work Guidance

- (empty until implementation)

## Verification

- No automated checks.

## Child DOX Index

- (none)
