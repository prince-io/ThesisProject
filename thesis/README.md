---
title: Thesis Knowledge Base Index
status: draft
date: 2026-11-07
sources: []
---

# Thesis Knowledge Base

Authoritative curated knowledge for the thesis. Start with `overview.md`.

## Reading order

1. `overview.md` — scope, goals, constraints, open questions.
2. `decisions/` — accepted and proposed decisions (ADRs).
3. `methodology/` — approach (empty until implementation).
4. `experiments/` — dataset record (`dataset-adverse16k.md`), datasheet (`dataset-card-adverse16k.md`), and run records.
5. `results/` → `findings/` — facts, interpretation.
6. `related-work/` + `bibliography/` — literature.

## Status

- Environment is set up (Python 3.12, PyTorch 2.14.1 CUDA).
- `adverse16k` dataset is built, verified, and audited (16,000 images, 4,000/condition) as an independent copy.
- Sequence leakage and cross-split near-duplicates are zero; the datasheet records composition, splits, biases, and limitations.
- Unlabeled ACDC test set prepared (2,000 images) for demos and final qualitative testing.
- Baseline training/evaluation harness is implemented; `EXP-0001-baseline.md` is filled after the run.
- Proposed LiteViT/weather-gate implementation has not started.

## Field map

| Folder | Holds | Does not hold |
|---|---|---|
| `methodology/` | approach, architecture, rationale | measured results |
| `decisions/` | dated ADRs | narrative history |
| `experiments/` | curated run records | raw logs |
| `results/` | measured facts | interpretation |
| `findings/` | interpretation and conclusions | raw metrics |
| `related-work/` | literature synthesis | uncited claims |
| `bibliography/` | BibTeX | PDFs |
