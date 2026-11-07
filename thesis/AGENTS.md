# thesis/ — Curated Source of Truth

## Purpose

Hold the curated, authoritative project knowledge used to draft the final thesis: methodology, decisions, experiments, results, findings, related work, and bibliography.

## Ownership

- Authoritative for all thesis content.
- `manuscript/` derives from this folder only.

## Local Contracts

- Field separation:
  - `methodology/` — approach, architecture, design rationale (empty until implementation).
  - `decisions/` — append-only, dated ADRs (`ADR-NNNN-slug.md`); never delete, mark superseded.
  - `experiments/` — curated run records: objective, setup, config, seed, run ID.
  - `results/` — facts only: metrics and tables tied to a run ID.
  - `findings/` — interpretation of results, lessons, conclusions.
  - `related-work/` — curated literature synthesis with citations.
  - `bibliography/` — BibTeX sources.
- Results and findings stay separate: never mix measured numbers with interpretation in one doc.
- Full raw artifacts live in `runs/` (gitignored); only curated summaries belong here.
- Every curated doc carries YAML front matter: `title`, `status` (`draft` | `validated` | `superseded`), `date`, `sources`.
- `ref/` is never cited as truth; cite `thesis/` docs or `bibliography/` keys.

## Work Guidance

- One topic per file; link related docs instead of duplicating.
- Cite sources for every external claim.
- Do not add methodology content until implementation starts.

## Verification

- No automated checks yet. Manual: confirm YAML front matter, working source links, and that results do not contain interpretation.

## Child DOX Index

- `methodology/AGENTS.md`
- `decisions/AGENTS.md`
- `experiments/AGENTS.md`
- `results/AGENTS.md`
- `findings/AGENTS.md`
- `related-work/AGENTS.md`
- `bibliography/AGENTS.md`
