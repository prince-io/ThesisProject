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
- `build.py` — `wrap_v1_p5` and `build_v1_p5`, which wrap the P5/SPPF layer (index 9).
- V2 (P4) and V3 (weather gate) are not implemented yet.
- Add tests in `tests/test_litevit_models.py` for shape, identity, parameter budget,
  and integration.

## Verification

- `.venv/bin/ruff check .`
- `.venv/bin/pytest`
- Build V1 and compare its initial output with the baseline model.

## Child DOX Index

- (none)
