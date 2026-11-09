---
title: Lightweight Weather Gate (L-WG)
status: draft
date: 2026-11-09
sources:
  - ref/idea.md
  - thesis/decisions/ADR-0009-weather-gate-design.md
  - thesis/related-work/weather-adaptive-gating.md
  - thesis/related-work/architecture-grounding.md
  - src/litevit/models/gates.py
  - configs/train/weather_gate.yaml
---

# Lightweight Weather Gate (L-WG)

## Purpose

Test the weather-adaptation mechanism on its own, without adding attention
capacity. The gate reweights existing neck channels; it does not create new
spatial features, change resolution, or shift Ultralytics layer indices.

## Placement

The gate is applied after the PAN/FPN neck, on all three detection scales, before
the YOLOv8 detect head:

```text
P3 (80×80, 64)  → Gate-P3  ─┐
P4 (40×40, 128) → Gate-P4  ─┼→ Detect head
P5 (20×20, 256) → Gate-P5  ─┘
```

Layer indices in the YOLOv8n graph: 15 (P3), 18 (P4), 21 (P5). Wrapping these
layers preserves the existing `i`, `f`, `type`, and output shape, so the rest of
the graph and pretrained weights are unchanged.

## Gate equation

For each feature map `F`:

```text
g = global average pool(F)                       # B,C,1,1
h = SiLU(Conv1x1(C → C/4)(g))
w = 2 · sigmoid(Conv1x1(C/4 → C)(h))             # range (0,2)
F' = F × w
```

- The final 1×1 convolution is zero-initialized (weight and bias).
- Therefore `w = 2 · sigmoid(0) = 1` at initialization, so the gate is exactly an
  identity.
- The gate can learn to suppress unreliable channels (`w < 1`) or amplify stable
  ones (`w > 1`).
- It runs in FP32 under AMP and casts back to the input dtype.

## Values and justification

| Choice | Value | Why |
|---|---|---|
| Reduction ratio | **C/4** | P3 has only 64 channels; C/16 leaves a hidden width of 4, too narrow. C/4 gives hidden widths 16/32/64. C/2 would double parameters to ~87k. |
| Activation | **SiLU** | YOLO-native smooth activation; no extra parameters. |
| Gate range | **2·sigmoid** | Exactly 1 at zero initialization; bounded 0–2; supports both suppression and amplification. Plain sigmoid only suppresses. |
| Final-layer init | **zero weight and bias** | Identity start; same stability principle as LayerScale/ReZero. |
| Placement | **P3/P4/P5 after neck** | Weather degrades all scales; P3 protects small objects, P5 carries global context. |
| Spatial attention | **not used** | Keeps the gate cheap and avoids adding the kind of capacity that can overfit. |
| Training | **end-to-end YOLO loss** | No separate weather classifier or two-stage preprocessing. |

Parameter budget:

| Scale | Channels | Hidden C/4 | Parameters |
|---|---:|---:|---:|
| P3 | 64 | 16 | 2,128 |
| P4 | 128 | 32 | 8,352 |
| P5 | 256 | 64 | 33,088 |
| **Total** | | | **43,568** |

Five-class total: **3,182,543 parameters** (baseline 3,138,975; +1.4%). Compute
and VRAM impact are negligible.

## Evidence

- SE and CBAM support global-pooled channel reweighting as a cheap attention
  mechanism.
- WAAM and WAMNet support weather-conditioned feature reweighting, but use heavier
  attention designs.
- ECL-YOLOv11 supports feature refinement under adverse weather, though with edge
  and multi-scale fusion rather than a gate.

The exact gate-only design on a YOLOv8n neck is our hypothesis; the values above
are engineering adaptations for small detector channels, not copied constants.

## Training protocol

The gate-only run uses the baseline protocol unchanged:

- 640×640, batch 4, AMP, `nbs=8`
- AdamW, `lr0=2e-4`, cosine, 3 warmup epochs, weight decay 5e-4
- 100 epochs, patience 20, seed 42, `cls_pw=0.5`
- Mosaic 0.5, MixUp 0.05, `close_mosaic=10`

No extra regularization is applied: the gate is identity-initialized and adds
only 43.6k parameters.

## Status

Implemented. Training and evaluation are pending (`EXP-0004`).
