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
4. `experiments/` → `results/` → `findings/` — run records, facts, interpretation.
5. `related-work/` + `bibliography/` — literature.

## Status

- Environment is set up (Python 3.12, PyTorch 2.14.1 CUDA).
- Implementation has not started.
- Methodology content is deferred until implementation begins.

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
