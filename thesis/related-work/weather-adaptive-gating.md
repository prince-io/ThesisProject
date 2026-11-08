---
title: Weather-Adaptive Gating — Literature Grounding
status: draft
date: 2026-11-08
sources:
  - ref/Real_Time_Object_Detection_in_Adverse_Weather_Conditions_Using_Transformer.pdf
  - ref/sensors-26-00304.pdf
  - thesis/bibliography/references.bib
---

# Weather-Adaptive Gating — Literature Grounding

Scope: evidence for a cheap channel gate on neck features, and how far that evidence
goes toward validating the proposed Lightweight Weather Gate (L-WG).

## Channel attention

Squeeze-and-Excitation [@hu2018senet] pools each channel globally and learns a
channel-wise multiplier. CBAM [@woo2018cbam] adds spatial attention on top. These
are strong evidence that channel reweighting is useful and cheap; they are not
weather-specific.

## Weather-adaptive attention

Nie and Wang [@nie2025transformerweather] propose WAAM, which injects weather cues
into attention and reweights features. Their ablation attributes +3.5 mAP to this
module. The mechanism is heavier than our gate and operates inside a ViT-B/16
pipeline, but it supports the idea of conditioning feature weights on degradation
cues.

WAMNet [@jaiswal2026wamnet] uses cross-dimensional multi-scale attention and a
contextual multi-attention fusion neck for adverse weather; it supports weather-aware
multi-scale reweighting, but at a larger YOLOv8s-scale design.

ECL-YOLOv11 [@liu2026ecl] uses context-guided multi-scale fusion and edge
enhancement rather than an explicit channel gate. It supports the broader claim
that adverse-weather robustness benefits from feature refinement, but it is not a
gate paper.

## Proposed L-WG

For each neck feature `F`:

```text
g = global average pool(F)
w = 2 * sigmoid( FC(C/4 -> C) ( SiLU( FC(C -> C/4)(g) ) ) )
F' = F * w
```

- Final layer zero-initialized; `w = 1` exactly at the start (identity).
- It can suppress unreliable channels (`w < 1`) or amplify stable ones (`w > 1`).
- Extra parameters: roughly 2k (P3), 8k (P4), 33k (P5); about 43k total.
- No spatial size change; negligible memory.

## Evidence status

| Claim | Evidence | Strength |
|---|---|---|
| Channel reweighting is useful | SE [@hu2018senet], CBAM [@woo2018cbam] | Strong |
| Weather-conditioned reweighting can help | WAAM [@nie2025transformerweather], WAMNet [@jaiswal2026wamnet] | Medium |
| A global-pool channel gate is enough | No direct evidence | Weak / our hypothesis |
| Gate after neck P3/P4/P5 is the right placement | No direct evidence | Weak / our hypothesis |
| `2*sigmoid` with zero-init final layer is stable | LayerScale/ReZero-style identity init [@touvron2021deeper; @bachlechner2021rezero] | Medium |
| Expected mAP gain | Not established | Hypothesis |

## What the evidence does not settle

- Whether global average pooling preserves enough weather information; spatial
  attention may be needed.
- Whether the gate should be applied before or after PAN/FPN.
- Whether a gate alone helps, or only in combination with global context.
- Whether gate weights will correlate with fog/rain/snow/night conditions.

## Recommended safeguard

Include a **gate-only ablation** if time permits: baseline YOLOv8n + L-WG on
P3/P4/P5, no LiteViT block. This isolates the gate from the ViT contribution. If
the gate-only model does not help, the main model's gain cannot be attributed to
the gate without further evidence.

## Diagnostic opportunity

Log the mean gate weight per channel and per weather condition during evaluation.
If the gate learns condition-specific channel suppression, that is useful evidence
even when the mAP gain is small. This is an analysis plan, not a result.
