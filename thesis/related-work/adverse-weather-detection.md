---
title: Adverse-Weather Object Detection — Literature Grounding
status: draft
date: 2026-11-08
sources:
  - ref/Real_Time_Object_Detection_in_Adverse_Weather_Conditions_Using_Transformer.pdf
  - ref/sensors-26-00304.pdf
  - thesis/bibliography/references.bib
---

# Adverse-Weather Object Detection — Literature Grounding

Scope: what the verified literature says about detection under fog, rain, snow, and
low light, and which parts of the YOLO-LiteViT-WG proposal it supports. This is a
synthesis, not a claim that our exact architecture has been validated.

The real-time detector family starts with YOLO [@redmon2016yolo]; this thesis uses
YOLOv8n [@ultralytics2023yolov8] as the reference because it fits the 6 GB budget.

## Verified lines of work

### 1. Hybrid CNN-ViT with weather-adaptive attention

Nie and Wang [@nie2025transformerweather] propose a hybrid CNN-ViT detector with a
Weather-Adaptive Attention Module (WAAM). The CNN path supplies local texture and
edge features; the ViT path supplies global context; WAAM reweights features using
visual degradation cues. They evaluate on DAWN, Foggy Cityscapes, and NightOwls.

Reported results are large: 75.8 mAP, 28 FPS on a Jetson device, with ablations of
+5.6 mAP from the ViT block, +3.5 mAP from WAAM, and +2.8 mAP from multi-scale
fusion. The setup is much heavier than ours: ViT-B/16 at 512x512, ImageNet-21k
initialization, 150 epochs, and a different evaluation protocol.

**What it supports:** the general hypothesis that combining local CNN features with
global Transformer context can help in degraded visibility, and that weather-aware
feature reweighting is a plausible mechanism.

**Caveat:** the paper is in IJECS, not a top-tier venue, and its gains are much
larger than the more modest results in the peer-reviewed ECL-YOLOv11 paper below.
Treat its numbers as directional motivation, not as a performance target or proof
for a lightweight YOLOv8n design.

### 2. Edge-enhanced, context-guided YOLOv11

Liu et al. [@liu2026ecl] propose ECL-YOLOv11 with three modules:

- a Convolutional Edge-enhancement (CE) module that fuses Sobel edge features with
  learned convolution features;
- a Context-guided Multi-scale Fusion Network (AENet) for P3/P4/P5 context fusion;
- a Lightweight Shared Convolution Detection Head (LDHead).

On their severe-weather dataset they report 62.7 mAP@50 and 40.5 mAP@50-95, which
is +1.3 and +0.8 over YOLOv11, at 237.8 FPS. The paper also notes that Transformer
global attention can over-rely on global context and lose local detail under severe
contrast degradation.

**What it supports:** preserving edge/local information, multi-scale context fusion,
and lightweight heads are useful under adverse weather; gains are often modest; a
hybrid design should keep a strong local CNN path.

**Caveat:** it is not a ViT-block or weather-gate paper, and its dataset is not
`adverse16k`. It does not test P4/P5 linear attention.

### 3. YOLOv8s-WAMNet

Jaiswal et al. [@jaiswal2026wamnet] propose YOLOv8s-WAMNet: a hybrid ViT backbone,
cross-dimensional multi-scale attention, a contextual multi-attention fusion neck,
and a multi-head dynamic detection head with a hybrid SIoU-MPDIoU loss. They report
robustness on a DALL-E-generated weather dataset and a real adverse-weather dataset,
and compare 13.24 vs 28.7 GFLOPs.

**What it supports:** hybrid attention and multi-scale fusion are active design
directions for adverse-weather YOLO detectors; a lightweight model can remain in the
same compute order as a YOLOv8s baseline.

**Caveat:** it uses YOLOv8s, not YOLOv8n; its modules are heavier than our proposed
P4/P5 blocks; the abstract reports mAP values as formula placeholders, so the exact
numbers should be read from the paper before citation.

## Evidence summary

| Work | Setting | Intervention | Reported outcome | Supports | Main caveat |
|---|---|---|---|---|---|
| Nie and Wang 2025 [@nie2025transformerweather] | ViT-B/16, 512, DAWN/Foggy Cityscapes/NightOwls | Hybrid CNN-ViT + WAAM | Large mAP/FPS gains, +5.6 ViT, +3.5 WAAM | Global context + weather-aware reweighting | Heavy setup; non-top-tier venue; different protocol |
| Liu et al. 2026 [@liu2026ecl] | YOLOv11, severe-weather dataset | Edge enhancement + context fusion + lightweight head | +1.3 mAP50, +0.8 mAP50-95, 237.8 FPS | Local detail + multi-scale context; modest gains | Not ViT/gating; custom data |
| Jaiswal et al. 2026 [@jaiswal2026wamnet] | YOLOv8s, weather datasets | Hybrid ViT backbone + multi-scale attention neck | Robustness claims; 13.24 GFLOPs | Hybrid attention + multi-scale fusion | Heavier model; exact mAP not extracted |

## What the evidence supports for our design

- **Global context can help** when local texture is degraded, but it should not
  replace the local CNN path.
- **Weather-aware reweighting** is a reasonable mechanism; a cheap channel gate is
  a simplification of WAAM/WAMNet-style attention.
- **Multi-scale context** matters; keep P3 for small objects and add global modeling
  only at lower-resolution P4/P5.
- **Evaluate per condition**, not only overall mAP; all three papers report
  condition-specific behavior.

## What the evidence does not settle

- The exact placement of one block at P4/P5 before PAN/FPN.
- Whether linear attention is better than window attention or a pure CNN block at
  these scales.
- Whether a channel gate on neck features is sufficient, or whether spatial/weather
  attention is needed.
- The expected mAP gain on `adverse16k` and the 6 GB memory behavior.
- The novelty of the exact YOLOv8n + P4/P5 LiteViT + gate combination.

## Implication

Use the literature to justify the **direction**, not the result. V1 (P5-only) and
the later P4/P5 + gate variants must be treated as falsifiable experiments, with
per-condition, per-class, and per-size evidence.
