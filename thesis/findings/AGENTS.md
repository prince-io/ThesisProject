# findings/ — Interpretation and Conclusions

## Purpose

Interpret results and record lessons, conclusions, and implications for the thesis.

## Ownership

- Authoritative for interpretation.

## Local Contracts

- YAML front matter on every doc.
- Every claim must cite the `../results/` doc it interprets.
- Never restate raw metrics as if measured here; reference them.
- Separate observation, interpretation, and limitation.

## Work Guidance

- `baseline-analysis.md` interprets the baseline results and error analysis.
- `ablation-analysis.md` interprets the `cls_pw=0.0` ablation and its implications.
- `v1-invalid-run.md` diagnoses the discarded-wrapper V1 run and records the lesson.
- `v1-p5-analysis.md` interprets the completed V1 P5 experiment as a negative result.
- Keep interpretation separate from the facts in `../results/` and cite them.

## Verification

- No automated checks.

## Child DOX Index

- (none)
