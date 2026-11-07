# B.Tech Thesis — Lightweight Adverse-Weather Object Detection

Final-year thesis project. Target: YOLO-LiteViT-WG (YOLOv8n backbone + linear-attention ViT at neck P4/P5 + lightweight weather channel gate), trainable at 640x640 on an RTX 3050 6GB.

## Repository layout

- `ref/` — non-authoritative input inbox (ideas, suggestions, papers). Never cited as truth.
- `thesis/` — curated source of truth: methodology, decisions, experiments, results, findings, related work, bibliography.
- `manuscript/` — derived LaTeX thesis, written from `thesis/`.
- `runs/` — full raw artifacts (untracked). Curated summaries live in `thesis/experiments` and `thesis/results`.

## Status

Implementation has not started. Methodology content is deferred until implementation begins.

## Planned structure (not yet created)

- `src/` — implementation, src-layout Python package.
- `configs/` — training and experiment YAML configs.
- `scripts/` — entrypoints and utilities.
- `tests/` — verification.
- `data/` — datasets (gitignored).
- `runs/` — full run artifacts (gitignored).
- `pyproject.toml` — dependencies and tooling (uv, PyTorch CUDA, Ultralytics).

## Conventions

- Every durable decision is an append-only ADR in `thesis/decisions/`.
- Curated docs use YAML front matter: `title`, `status` (`draft` | `validated` | `superseded`), `date`, `sources`.
- `ref/*.pdf` is untracked.
