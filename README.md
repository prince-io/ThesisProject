# B.Tech Thesis — Lightweight Adverse-Weather Object Detection

Final-year thesis project. Target: YOLO-LiteViT-WG (YOLOv8n backbone + linear-attention ViT at neck P4/P5 + lightweight weather channel gate), trainable at 640x640 on an RTX 3050 6GB.

## Repository layout

- `ref/` — non-authoritative input inbox (ideas, suggestions, papers). Never cited as truth.
- `thesis/` — curated source of truth: methodology, decisions, experiments, results, findings, related work, bibliography.
- `manuscript/` — derived LaTeX thesis, written from `thesis/`.
- `runs/` — full raw artifacts (untracked). Curated summaries live in `thesis/experiments` and `thesis/results`.

## Status

Environment is set up. Implementation has not started. Methodology content is deferred until implementation begins.

## Code layout

- `src/litevit/` — implementation package (editable install).
- `configs/` — training and experiment YAML configs.
- `scripts/` — entrypoints and utilities.
- `tests/` — pytest suite.
- `data/` — datasets (gitignored).
- `runs/` — full run artifacts (gitignored).
- `pyproject.toml` — dependencies and tooling.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install -e ".[dev]"
```

Verify CUDA: `.venv/bin/python -c "import torch; print(torch.cuda.is_available())"`.
Run checks: `.venv/bin/ruff check .` and `.venv/bin/pytest`.

## Conventions

- Every durable decision is an append-only ADR in `thesis/decisions/`.
- Curated docs use YAML front matter: `title`, `status` (`draft` | `validated` | `superseded`), `date`, `sources`.
- `ref/*.pdf` is untracked.
