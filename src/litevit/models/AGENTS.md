# litevit/models/ — LiteViT Blocks and Builders

## Purpose

Implement the LiteViT blocks and build the proposed detector variants on top of
pretrained Ultralytics YOLO models.

## Ownership

- Authoritative for block math, initialization, and model wrapping.

## Local Contracts

- Blocks are channel- and resolution-preserving unless documented otherwise.
- New blocks initialize near identity so pretrained detector behavior is preserved.
- Builders load pretrained weights first, then wrap existing layers; do not shift
  Ultralytics layer indices.
- Keep model code framework-native (PyTorch/Ultralytics); no dataset or file I/O at
  import time.
- Document parameter and VRAM impact for every new block.

## Work Guidance

- `blocks.py` — LayerNorm2d, linear attention, ConvFFN, LiteViTBlock,
  FeatureWithBlock.
- `gates.py` — WeatherGate and FeatureWithGate (P3/P4/P5 channel gating).
- `build.py` — `wrap_v1_p5`/`build_v1_p5` (P5 layer 9),
  `wrap_gate_only`/`build_gate_only` (P3/P4/P5 layers 15/18/21), and
  `wrap_gate_p5`/`build_gate_p5` for A+B.
- V2 (P4 attention) is not implemented yet; V1, gate-only, and A+B are implemented.
- Add tests in `tests/test_litevit_models.py` for shape, identity, parameter budget,
  and integration.

## Verification

- `.venv/bin/ruff check .`
- `.venv/bin/pytest`
- Build V1 and gate-only and compare their initial outputs with the baseline model.

## Child DOX Index

- (none)
