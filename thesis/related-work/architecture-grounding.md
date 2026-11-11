---
title: Architecture Grounding — Evidence vs Assumption
status: draft
date: 2026-11-08
sources:
  - thesis/bibliography/references.bib
  - thesis/related-work/adverse-weather-detection.md
  - thesis/related-work/lightweight-vit-detection.md
  - thesis/related-work/weather-adaptive-gating.md
  - ref/idea.md
---

# Architecture Grounding — Evidence vs Assumption

Purpose: separate what the literature supports from what we are assuming before
implementing V1. Verification used the local PDFs and DOI metadata (OpenAlex) on
2026-11-08. The exact YOLO-LiteViT-WG combination is not directly evaluated by any
verified paper.

## Design-choice table

| Design choice | Evidence | Strength | Remaining assumption | Validation |
|---|---|---|---|---|
| YOLOv8n + P3/P4/P5 neck | YOLOv8 [@ultralytics2023yolov8]; FPN [@lin2017fpn]; PANet [@liu2018panet] | Strong | None for the baseline | Already measured in `EXP-0001` |
| Global context only at P4/P5 | Swin hierarchical attention [@liu2021swin]; ViT-YOLO++ neck encoder [@yadav2026vityolo]; EfficientViT [@cai2022efficientvit] | Medium | One block each, pre-neck, is our choice | V1 (P5), V2 (P4/P5) |
| Linear attention | Katharopoulos et al. [@katharopoulos2020linear]; EfficientViT [@cai2022efficientvit] | Strong for efficiency | Exact normalization/heads/AMP behavior | V1 smoke + full run |
| Depthwise positional encoding (CPE) | CPVT [@chu2021cpe] | Strong | Kernel size 3x3, placement before QKV | V1 |
| ConvFFN with depthwise 3x3 | MobileViT [@mehta2022mobilevit]; EfficientViT [@cai2022efficientvit]; YOLO-CasViT [@ngo2026yolocasvit] | Medium | Expansion ratio 2 and exact layout | V1 |
| Zero-initialized residual scaling | LayerScale [@touvron2021deeper]; ReZero [@bachlechner2021rezero] | Strong for stability | Zero-init LayerScale is our scheme | Identity-check test |
| Channel weather gate | SE [@hu2018senet]; CBAM [@woo2018cbam]; WAAM [@nie2025transformerweather]; WAMNet [@jaiswal2026wamnet] | Medium | GAP + 2-layer MLP + `2*sigmoid` is ours | V3, optional gate-only |
| P3 stays CNN-only | FPN/small-object design [@lin2017fpn]; ECL-YOLOv11 [@liu2026ecl] | Medium | P3 attention might help or hurt | V2/V3 comparison |
| 640, batch 4, AMP on 6 GB | Local baseline/ablation measurements; AMP [@micikevicius2018mixed] | Strong locally | Proposed model memory is unmeasured | Smoke run |
| ~3.67 M / ~3.81 M params; ~9.3–9.5 GFLOPs | Arithmetic estimate from block definitions | Medium | Actual counts/FLOPs may differ | `model.info()` |
| Expected mAP / small-object gain | No direct evidence for this exact design | Weak | The core hypothesis | V1–V3 results |
| `cls_pw=0.5` fixed | Local `EXP-0002` ablation | Local evidence | None for this thesis | Already recorded |

## Verified sources

The following works were checked against local PDFs or DOI metadata:

- Nie and Wang, transformer-based adverse-weather detection [@nie2025transformerweather]
- Liu et al., ECL-YOLOv11 [@liu2026ecl]
- Jaiswal et al., YOLOv8s-WAMNet [@jaiswal2026wamnet]
- Yadav, ViT-YOLO++ [@yadav2026vityolo]
- Zhang et al., ViT-YOLO [@zhang2021vityolo]
- Cheng et al., EF-DETR [@cheng2024efdetr]
- Ngo et al., YOLO-CasViT [@ngo2026yolocasvit]
- Cai et al., EfficientViT [@cai2022efficientvit]
- Chu et al., CPVT/CPE [@chu2021cpe]
- Touvron et al., LayerScale [@touvron2021deeper]
- Bachlechner et al., ReZero [@bachlechner2021rezero]

Foundational entries for ViT, Swin, linear attention, SE, CBAM, FPN, PANet, MobileViT,
YOLO, AMP, AdamW, and gradient checkpointing are also in
`thesis/bibliography/references.bib`.

## Unverified or qualified claims

- **“C2f-ViT”:** no separate paper with that exact title was verified. The closest
  verified work is YOLO-CasViT, which uses `C2f_CasViT` [@ngo2026yolocasvit].
  Do not cite “C2f-ViT” until a source is found; use YOLO-CasViT instead.
- **WAMNet mAP values:** the OpenAlex abstract contains formula placeholders for
  the mAP numbers. The 13.24 vs 28.7 GFLOPs comparison is readable, but exact mAP
  values must be taken from the paper before citation.
- **“P5-only allows batch 6”:** an estimate from `ref/idea.md`, not a measurement.
- **Parameter and GFLOP estimates:** arithmetic predictions; verify with
  `model.info()`.
- **Expected mAP gains:** not established for this architecture on `adverse16k`.
- **Novelty of the exact combination:** the gap is real, but it is a hypothesis
  until V1–V3 produce results.

## What the literature does not settle

1. Whether one block at P5 is enough, or P4 is necessary.
2. Whether the block should be before PAN/FPN, inside the neck, or after it.
3. Whether linear attention beats window attention or a pure CNN block at P4/P5.
4. Whether a channel gate alone is sufficient for weather adaptation.
5. Whether the gate should use feature statistics or image-level weather cues.
6. Whether the gains survive multi-seed evaluation and per-condition analysis.

## V1 status

V1 is implemented as a **falsifiable hypothesis**, not as a validated method:

- P5-only LiteViT block before PAN/FPN (`src/litevit/models/`).
- Identity initialization (zero LayerScale) so the model starts equal to the
  pretrained YOLOv8n baseline.
- Same protocol as `EXP-0001`: 640, batch 4, AMP [@micikevicius2018mixed],
  `nbs=8`, AdamW 2e-4 [@loshchilov2019adamw], 100 epochs, seed 42, `cls_pw=0.5`.
- Gradient checkpointing [@chen2016sublinear] is a fallback only if the P4 block
  exceeds the memory budget.
- Smoke checks: forward/backward at 640 batch 4, save/load, and initial output
  equality with the baseline. Full training results are pending.
- Report per-condition, per-class, per-size, and per-source results; do not claim
  success from overall mAP alone.

The architecture document is `thesis/methodology/architecture.md`. The run record is
`thesis/experiments/EXP-0003-v1-p5.md`.

## Gate-only status

The first V1 result was negative: the P5 block overfit after epoch 18. The next run
isolates the weather gate (`arch: gate_only`) on P3/P4/P5 with C/4, SiLU,
`2*sigmoid`, and zero initialization. It adds 43,568 parameters and is documented in
`thesis/methodology/weather-gate.md` and `thesis/decisions/ADR-0009-weather-gate-design.md`.
The run record is `thesis/experiments/EXP-0004-weather-gate.md`; the result is a small test gain with slightly lower validation, documented in `thesis/findings/weather-gate-analysis.md`.

A+B combines that gate with a regularized P5 block (DropPath 0.2, dropout 0.1,
MLP ratio 1.0). It is complete and neutral on test (`EXP-0005`); see
`thesis/results/gate_p5.md` and `thesis/findings/gate-p5-analysis.md`.

The kitchen-sink model adds Sobel edge residuals at P3/P4 and spatial CBAM
(`EXP-0006`); see `thesis/methodology/gate-edge-cbam.md` and
`thesis/decisions/ADR-0011-gate-edge-cbam-design.md`. It was neutral on test.
TTA was tested and rejected because it reduced accuracy.

The P2 model adds a stride-4 detection level to gate-only (`EXP-0007`); see
`thesis/methodology/p2-head.md`, `thesis/results/gate_p2.md`, and
`thesis/decisions/ADR-0012-p2-head-design.md`. It is the best test result
(+1.27 pp mAP50 and +0.50 pp mAP50-95 over gate-only) but the gain is modest,
non-uniform, and single-seed. It is a resolution result, not evidence for another
attention module.
