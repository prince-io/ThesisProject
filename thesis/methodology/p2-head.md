---
title: P2 High-Resolution Detection Head
status: draft
date: 2026-11-10
sources:
  - src/litevit/models/p2.py
  - src/litevit/models/build.py
  - configs/train/gate_p2.yaml
  - thesis/decisions/ADR-0012-p2-head-design.md
  - thesis/experiments/EXP-0007-gate-p2.md
---

# P2 High-Resolution Detection Head

## Scope

The P2 model is the gate-only detector plus a stride-4 detection level:

```text
YOLOv8n backbone
  -> P3/P4/P5 PAN/FPN neck with weather gates
  -> Detect(P2, P3, P4, P5)
```

It does **not** add P5/P4 attention, Sobel edge, spatial CBAM, or TTA. The aim is
to isolate the effect of a high-resolution detection level on tiny objects.

## P2 branch

The existing P3 neck output (layer 15, 64 channels, stride 8) is upsampled by 2
to stride 4. It is concatenated with the backbone P2 output (layer 2, 32 channels,
stride 4), producing 96 channels at 160x160 for a 640x640 input. A C2f block
reduces this to 32 channels, which feeds the new P2 Detect branch.

```text
P3 neck (64, 80x80) --upsample--> 64, 160x160
backbone P2 (32, 160x160)
concat -> 96, 160x160
C2f(96 -> 32) -> P2 feature
Detect P2 at stride 4
```

The four detection levels use channels **32 / 64 / 128 / 256** for P2/P3/P4/P5.

## Pretrained-weight handling

The pretrained YOLOv8n layers 0-21 are kept. The original P3/P4/P5 Detect box and
class branches are copied into the new four-level Detect head. The new P2 Detect
branch is randomly initialized. This is **not** an identity-initialized model:
initial detections differ from the baseline, unlike V1 and the gate models.

The graph is extended before the old Detect layer, so all original layer indices
used by the gates (15/18/21) and the P2 source (2) remain valid. New layers are:

| Index | Layer | Function |
|---:|---|---|
| 22 | `nn.Upsample` | P3 -> stride 4 |
| 23 | `Concat` | upsampled P3 + backbone P2 |
| 24 | `C2f` | 96 -> 32 channels |
| 25 | `Detect` | P2/P3/P4/P5, strides 4/8/16/32 |

The detector saved-output list includes layer 2 and layer 24, and the checkpoint
verifier checks for 26 layers, four detection levels, and the expected strides.

## Gating

The weather gate remains on the P3/P4/P5 neck outputs only. The new P2 branch is
not gated. This keeps the comparison to gate-only clean: the only change is the
P2 detection level.

## Size and compute

Measured with the 5-class trainer build:

- Parameters: **3,270,388** (gate-only: 3,182,543; +87,845)
- GFLOPs at 640: **13.3** (gate-only: 8.789)
- Four detection levels; P2 feature map is 160x160 at 640 input

The compute increase is about 51%. The P2 branch adds many anchor points at
high resolution; training and inference are expected to be slower.

## Training protocol

Same protocol as `EXP-0001`/`EXP-0004`:

- 640x640, batch 4, AMP, `nbs=8`
- AdamW, `lr0=2e-4`, cosine, 3 warmup epochs, weight decay 5e-4
- 100 epochs, patience 20, seed 42, `cls_pw=0.5`
- Mosaic 0.5, MixUp 0.05, `close_mosaic=10`

Expected full-run time is **7-8 h** on the RTX 3050. A 640x640 batch-4 smoke
run (1 epoch, 0.2% of training images) passed architecture verification, checkpoint
reload, and half-precision final validation; peak GPU memory was **1,595 MB**. The
full run should therefore fit the 6 GB budget with batch 4.

## Hypothesis and decision rule

P2 should improve tiny-object recall and AP by giving small objects a stride-4
feature map and more positive assignments. It may also increase false positives
and reduce precision.

Treat it as a win only if one of these holds:

- test mAP50 > 0.5444 (gate-only);
- test mAP50-95 > 0.3301 (gate-only);
- tiny recall > 0.3395 with no meaningful night/rain regression.

If only tiny recall improves while overall mAP is flat or lower, record a neutral
result and stop.
