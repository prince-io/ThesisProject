---
title: Thesis Knowledge Base Index
status: draft
date: 2026-11-08
sources:
  - thesis/overview.md
  - thesis/experiments/README.md
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
- Baseline training/evaluation harness is implemented; `EXP-0001-baseline.md` records the completed run.
- Baseline results and error analysis are documented in `results/` and interpreted in `findings/`.
- `EXP-0002` (`cls_pw=0.0`) is complete; comparison and interpretation are in `results/ablation-comparison.md` and `findings/ablation-analysis.md`.
- `cls_pw=0.5` is frozen for later comparisons (`decisions/ADR-0008-freeze-class-weighting.md`).
- Architecture literature grounding is complete: verified BibTeX entries are in `bibliography/references.bib`; synthesis is in `related-work/`.
- The design-choice → evidence → assumption map is in `related-work/architecture-grounding.md`.
- V1 (P5-only LiteViT) is next; the proposed LiteViT/weather-gate implementation has not started.

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
