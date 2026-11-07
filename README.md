# B.Tech Thesis — Lightweight Adverse-Weather Object Detection

Final-year thesis project. Target: YOLO-LiteViT-WG (YOLOv8n backbone + linear-attention ViT at neck P4/P5 + lightweight weather channel gate), trainable at 640x640 on an RTX 3050 6GB.

## Repository layout

- `ref/` — non-authoritative input inbox (ideas, suggestions, papers). Never cited as truth.
- `thesis/` — curated source of truth: methodology, decisions, experiments, results, findings, related work, bibliography.
- `manuscript/` — derived LaTeX thesis, written from `thesis/`.
- `runs/` — full raw artifacts (untracked). Curated summaries live in `thesis/experiments` and `thesis/results`.

## Status

Environment is set up and the `adverse16k` dataset is built and verified. Implementation has not started; methodology content is deferred until implementation begins.

## Code layout

- `src/litevit/` — implementation package (editable install).
- `configs/` — training and experiment YAML configs (`configs/datasets/adverse16k.yaml`).
- `scripts/` — entrypoints and utilities.
- `tests/` — pytest suite.
- `data/` — datasets (gitignored). Processed benchmark: `data/processed/adverse16k`.
- `runs/` — full run artifacts (gitignored).
- `pyproject.toml` — dependencies and tooling.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install -e ".[dev,datasets]"
```

Verify CUDA: `.venv/bin/python -c "import torch; print(torch.cuda.is_available())"`.
Run checks: `.venv/bin/ruff check .` and `.venv/bin/pytest`.

## Dataset (adverse16k)

16,000 images, 4,000 per condition (fog, rain, snow, night), 5 classes
(`person, car, bus, bicycle, motorcycle`), split 12,800/1,600/1,600.
Built from ACDC + DAWN + BDD100K + RTTS as an **independent copy** with
`checksums.sha1`. See `thesis/experiments/dataset-adverse16k.md`.

```bash
.venv/bin/python scripts/build_adverse16k.py --root . --out data/processed/adverse16k \
  --force --copy-images
.venv/bin/python scripts/verify_adverse16k.py --root . --out data/processed/adverse16k \
  --report runs/dataset_checks/adverse16k --require-independent --checksums
.venv/bin/python scripts/prepare_acdc_test.py --root . --out data/unlabeled/acdc_test \
  --report runs/dataset_checks/acdc_test --force
```

The unlabeled **ACDC test** set (`data/unlabeled/acdc_test`, 2,000 images) is for
demos and final qualitative testing; its ground truth is withheld by ACDC.

## Conventions

- Every durable decision is an append-only ADR in `thesis/decisions/`.
- Curated docs use YAML front matter: `title`, `status` (`draft` | `validated` | `superseded`), `date`, `sources`.
- Raw datasets live under `ref/` (untracked) and are retained for now; processed and test copies live under `data/` and are independent.
- `ref/*.pdf` is untracked.
