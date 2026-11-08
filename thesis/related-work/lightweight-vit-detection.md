---
title: Lightweight ViT and Hybrid Detection — Literature Grounding
status: draft
date: 2026-11-08
sources:
  - thesis/bibliography/references.bib
---

# Lightweight ViT and Hybrid Detection — Literature Grounding

Scope: evidence for putting Transformer blocks at low-resolution detection features,
using linear attention, and keeping the design trainable at 640x640 on 6 GB.

## Foundations

### Global attention and its cost

The Transformer [@vaswani2017attention] and ViT [@dosovitskiy2021vit] show that
self-attention can model long-range dependencies, but standard attention is
quadratic in the number of tokens. At 640x640, a P3 feature map has 80x80 = 6,400
tokens; P4 has 1,600; P5 has 400. This is the main reason to restrict attention to
P4/P5 and to use linear attention.

### Hierarchical and windowed attention

Swin Transformer [@liu2021swin] uses a hierarchical design and shifted-window
attention, which is strong evidence for two choices: do not put full global
attention on the highest-resolution stage, and use locality-aware attention at
larger scales. Swin is a general vision backbone, not a YOLO neck, so it supports
the principle rather than the exact block.

### Linear attention

Katharopoulos et al. [@katharopoulos2020linear] show that linear attention can
reduce the quadratic cost by rewriting attention as a kernel feature map. EfficientViT
[@cai2022efficientvit] applies multi-scale linear attention to high-resolution dense
prediction and is cited by the ECL-YOLOv11 paper as a lightweight backbone option.
This supports linear attention at P4/P5, but not the exact normalization, head count,
or AMP behavior of our block.

### Positional encoding and stable residual scaling

CPVT [@chu2021cpe] introduces conditional positional encoding via a depthwise
convolution, which avoids fixed-size position embeddings and is suitable for
multi-resolution detection features. LayerScale [@touvron2021deeper] and ReZero
[@bachlechner2021rezero] support small or zero-initialized residual branches so a
new block starts near identity. These are strong supports for our CPE and
zero-initialized LayerScale choices.

### Hybrid CNN-Transformer backbones

MobileViT [@mehta2022mobilevit] combines convolutional locality with Transformer
global modeling in a lightweight backbone, but it is a backbone design, not a
neck block. It supports the general hybrid idea; it does not justify inserting
attention into high-resolution YOLO stages.

## Direct detection work

| Work | Where attention is used | What it reports | Relevance to us | Caveat |
|---|---|---|---|---|
| ViT-YOLO [@zhang2021vityolo] | MHSA-Darknet backbone + BiFPN neck | Transformer-based detection with multi-scale fusion | Supports Transformer features + multi-scale neck | Backbone replacement; UAV-style data |
| ViT-YOLO++ [@yadav2026vityolo] | Light Transformer encoder on YOLOv11 neck | +7.1 mAP@0.5, 36 FPS on MS COCO/Pascal VOC/SKU110K | Direct support for a neck-level Transformer | YOLOv11, not YOLOv8n; different data |
| YOLO-CasViT [@ngo2026yolocasvit] | Replaces C2f with C2f_CasViT (convolutional additive self-attention) | +3.13 mAP@0.5, +0.49 mAP@0.5:0.95 on KITTI | Supports C2f-level attention and local-global mixing | Backbone replacement; KITTI traffic data |
| EF-DETR [@cheng2024efdetr] | Encoder-free lightweight neck | Lightweight Transformer detector with multi-scale feature extraction | Supports lightweight neck design | DETR family; COCO; not YOLO |
| EfficientViT [@cai2022efficientvit] | Multi-scale linear attention backbone | Efficient high-resolution dense prediction | Supports linear attention at scale | Backbone, not a detection neck |

## What is supported

- **Attention at low-resolution stages:** Swin, ViT-YOLO++, and EfficientViT all
  support restricting global modeling to coarser features.
- **Linear attention:** linear attention is a recognized way to reduce the cost of
  global mixing at high resolution.
- **CPE:** depthwise positional encoding is an established alternative to fixed
  position embeddings.
- **Identity initialization:** LayerScale and ReZero support starting a new residual
  branch near identity, which is important when fine-tuning a pretrained detector.
- **Neck-level Transformer:** ViT-YOLO++ and EF-DETR show that Transformer or
  encoder-free necks are viable, though with different detectors and datasets.

## What is our hypothesis

- One channel-preserving LiteViT block each at P4/P5, placed before PAN/FPN.
- A ConvFFN with depthwise 3x3 mixing and expansion ratio 2.
- Four heads, `ELU+1` linear attention, and a specific normalization.
- The predicted parameter count (~3.67 M for V1, ~3.81 M for V2) and GFLOPs.
- That this will improve `adverse16k` mAP or small-object recall.

None of these exact choices is established by the cited papers. They are
engineering hypotheses to test in V1/V2/V3.

## Caveat on “C2f-ViT”

`ref/idea.md` names “C2f-ViT” as related work. We verified **YOLO-CasViT**, which
uses a `C2f_CasViT` block [@ngo2026yolocasvit], but we did not verify a separate
paper titled “C2f-ViT”. The thesis should either cite YOLO-CasViT precisely or
remove the unqualified “C2f-ViT” claim.
