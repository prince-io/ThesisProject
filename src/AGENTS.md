# src/ — Implementation Package

## Purpose

Hold the `litevit` Python package: backbone/neck/head modules, weather gating, data pipeline, training, and evaluation code.

## Ownership

- Authoritative for implementation behavior.
- Lives in `src/litevit/`; installed editable as `litevit`.

## Local Contracts

- Python 3.12; src layout; imports use `litevit.*`.
- Model code stays framework-native (PyTorch + torchvision + timm); no TensorFlow.
- Keep modules small and composable; no I/O or dataset paths at import time.
- Respect the 6 GB budget: document peak VRAM for any new block.
- Do not hardcode dataset paths; read from `configs/`.

## Work Guidance

- Dataset preparation and the training/evaluation harness are implemented.
- V1 (P5 attention), gate-only, P2 (stride-4 head), A+B (gate + regularized P5
  attention), and the kitchen-sink model (gate + Sobel edge + spatial CBAM) are
  implemented in `litevit/models/` and built inside `litevit/training/trainer.py`;
  V2 (P4 attention) is not implemented yet.
- Add modules one concern at a time.
- Add a test in `tests/` for each new public behavior.

## Verification

- `.venv/bin/ruff check .`
- `.venv/bin/pytest`
- `.venv/bin/python -c "import litevit"`

## Child DOX Index

- `litevit/datasets/AGENTS.md` — dataset loaders, class mapping, split policy, and build/verify entrypoints.
- `litevit/models/AGENTS.md` — LiteViT blocks, identity initialization, and detector builders.
- `litevit/training/AGENTS.md` — training, subset generation, evaluation, and run metadata.
