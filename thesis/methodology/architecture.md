---
title: Architecture — V1 P5 LiteViT
status: draft
date: 2026-11-08
sources:
  - thesis/decisions/ADR-0002-yolo-litevit-wg-architecture.md
  - thesis/related-work/architecture-grounding.md
  - thesis/related-work/lightweight-vit-detection.md
  - ref/idea.md
  - configs/train/litevit_v1_p5.yaml
  - src/litevit/models/blocks.py
  - src/litevit/models/build.py
---

# Architecture — V1 P5 LiteViT

## Scope

V1 is the first, smallest step of YOLO-LiteViT-WG:

```text
YOLOv8n backbone → LiteViT-P5 → PAN/FPN neck → YOLOv8 detect head
```

It keeps the backbone, neck, and head unchanged. One channel-preserving LiteViT
block is inserted after the P5/SPPF backbone layer (index 9), before the first
upsample and PAN/FPN fusion. P3/P4 remain CNN-only. The weather gate is not part
of V1.

V1 is an ablation: it tests whether global context at the deepest, lowest-resolution
feature helps `adverse16k` detection. It is not yet the full proposed model.

## LiteViT block

Input `x` is a BCHW feature map. The block preserves `C`, `H`, and `W`.

### 1. Normalization and positional encoding

```text
y = LayerNorm2d(x)
y = y + CPE(y)
```

`CPE` is a depthwise 3x3 convolution. It supplies local positional information
without fixed-size position embeddings, which is useful across feature scales.

### 2. Linear attention

A 1x1 convolution produces fused QKV:

```text
qkv = Conv1x1(y)                  # 3C channels
q, k, v = split(qkv)
q = ELU(q) + 1
k = ELU(k) + 1
kv = k^T v
out = q (kv) / (q · sum(k) + eps)
```

The attention is computed in float32 for numerical stability under AMP, then cast
back to the input dtype. Four heads are used; the head dimension is `C/4`.

### 3. Residual and ConvFFN

```text
x = x + gamma1 * Conv1x1(attention)
x = x + gamma2 * ConvFFN(LayerNorm2d(x))
```

`ConvFFN` is:

```text
1x1 C→2C → GELU → depthwise 3x3 → GELU → 1x1 2C→C
```

`gamma1` and `gamma2` are LayerScale parameters initialized to zero. Therefore the
block is exactly an identity at initialization, and the wrapped detector starts
from the pretrained YOLOv8n behavior.

## Integration with Ultralytics

`build_v1_p5` loads `yolov8n.pt` first, so all original weights transfer. It then
replaces detector layer 9 with a wrapper:

```python
FeatureWithBlock(base=SPPF, block=LiteViTBlock(256))
```

The wrapper preserves the original `i`, `f`, `type`, and output shape, so the
Ultralytics forward graph and saved-output indices do not change. New blocks are
identity-initialized; no state-dict remapping is needed.

## Size and compute

Measured from the V1 model summary before the 5-class head remap (not a training
run):

- Parameters: **3,692,496**; after the 5-class remap, expect about 3.674 M
  (verify in the training run metadata).
- GFLOPs at 640: **9.3**.
- Added block parameters: **535,296**.

The baseline `EXP-0001` measured **3.139 M** parameters and **8.754 GFLOPs** with
its 5-class head and evaluation path. These values must be re-checked after any
block change.

## Training protocol

V1 uses the same protocol as `EXP-0001`:

- 640x640, batch 4, AMP, `nbs=8`
- AdamW, `lr0=2e-4`, cosine, 3 warmup epochs, weight decay 5e-4
- 100 epochs, patience 20, seed 42, `cls_pw=0.5`
- Mosaic 0.5, MixUp 0.05, `close_mosaic=10`

No freeze schedule is used in V1. The identity initialization is the stability
mechanism; freezing can be added only if the run is unstable.

## Evidence status

The block components are literature-backed (linear attention, CPE, LayerScale,
local CNN path). The exact P5 placement, one-block choice, parameter budget, and
expected mAP gain are hypotheses. See
`../related-work/architecture-grounding.md` for the evidence/assumption map.

## V2/V3 extensions

- V2: add a LiteViT-P4 block after the backbone P4 C2f layer.
- V3: add the Lightweight Weather Gate (L-WG) on neck P3/P4/P5 outputs.
- Optional: gate-only ablation to isolate the weather gate.

These are not implemented in V1.
