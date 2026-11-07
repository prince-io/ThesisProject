---
title: Training and Evaluation Protocol
status: draft
date: 2026-11-07
sources:
  - thesis/decisions/ADR-0006-baseline-protocol.md
  - thesis/decisions/ADR-0007-measurement-and-analysis-protocol.md
  - thesis/experiments/dataset-adverse16k.md
  - configs/train/baseline.yaml
---

# Training and Evaluation Protocol

## Data

`adverse16k`: 16,000 images, 4,000 per condition (fog, rain, snow, night), 5 classes,
split 12,800/1,600/1,600. The dataset is an independent copy with sequence-aware splits
and checksums; see `thesis/experiments/dataset-card-adverse16k.md`.

## Reference model

YOLOv8n, COCO-pretrained, with class-head remapping to the 5 target classes.

## Reference configuration

| Setting | Value |
|---|---|
| Config file | `configs/train/baseline.yaml` |
| Model | `yolov8n.pt`, `pretrained=True`, `cls_remap=True` |
| Data | `configs/datasets/adverse16k.yaml` |
| Input / batch | `imgsz=640`, `batch=4`, `amp=True` |
| Accumulation | `nbs=8` (2 optimizer steps per nominal batch) |
| Optimizer | `AdamW`, `lr0=2e-4`, `lrf=0.01`, `cos_lr=True` |
| Schedule | `warmup_epochs=3`, `weight_decay=5e-4`, `epochs=100`, `patience=20` |
| Augmentation | `mosaic=0.5`, `mixup=0.05`, `close_mosaic=10` |
| Class weighting | `cls_pw=0.5` |
| Reproducibility | `seed=42`, `deterministic=True`, `workers=4`, `cache=False` |
| Output | `runs/train/<run_name>/` |

The authoritative resolved configuration for a completed run is stored in
`runs/train/<run_name>/resolved_config.yaml`.

## Training

- Input 640x640; batch 4; AMP on an RTX 3050 6 GB.
- Nominal batch 8 (gradient accumulation 2).
- AdamW, initial LR 2e-4, cosine schedule, 3 warmup epochs, weight decay 5e-4.
- 100 epochs, early stopping patience 20.
- Mosaic 0.5, MixUp 0.05, mosaic disabled for the final 10 epochs.
- Deterministic training, seed 42.
- Class weighting: `cls_pw=0.5`, i.e. inverse-frequency weights raised to the power 0.5
  and normalized to mean 1. This dampens the severe car dominance.

## Commands

```bash
# Full training
.venv/bin/python scripts/train.py \
  --config configs/train/baseline.yaml \
  --name baseline_yolov8n_seed42

# Evaluation (val + test, per condition/source)
.venv/bin/python scripts/evaluate.py \
  --weights runs/train/baseline_yolov8n_seed42/weights/best.pt \
  --data configs/datasets/adverse16k.yaml \
  --splits val test \
  --out runs/eval/baseline_yolov8n_seed42 \
  --subsets data/processed/adverse16k/subsets \
  --figures-dir thesis/results/figures/baseline

# Error analysis
.venv/bin/python scripts/error_analysis.py \
  --weights runs/train/baseline_yolov8n_seed42/weights/best.pt \
  --out runs/eval/baseline_yolov8n_seed42/error_analysis \
  --splits val test

# Curated records
.venv/bin/python scripts/document_results.py \
  --train-run runs/train/baseline_yolov8n_seed42 \
  --eval-run runs/eval/baseline_yolov8n_seed42 \
  --figures-dir thesis/results/figures/baseline \
  --vram-report runs/train/vram_calibration/vram.json
```

For the ablation, the run writes its own `vram.json`; pass
`--vram-report runs/train/ablation_cls_pw0_seed42/vram.json` to the document step.

## Evaluation

- Overall val/test precision, recall, mAP50, mAP50-95.
- Per-class AP50 and AP50-95, with target counts to expose low-support classes.
- Per-condition and per-source subsets.
- Efficiency: parameters, GFLOPs, inference latency, FPS.
- Error analysis: confidence 0.25 (primary) and 0.001 (ceiling), IoU 0.5; TP/FP/FN
  by condition, source, class, and object-size bin; confusion and background FP.
- Full artifacts in `runs/`; curated facts in `thesis/results/`; interpretation in
  `thesis/findings/`.

## Memory measurement

- Training PyTorch peak allocated and reserved are tracked by callbacks.
- Total GPU used peak is sampled with `nvidia-smi` during a calibration run with
  identical settings when a completed run lacks the external sample.
- Inference peak is measured separately during evaluation.
- Values are reported with explicit labels; see `ADR-0007`.

## Ablation protocol

The class-weighting ablation uses `configs/train/ablation_cls_pw0.yaml`
(`cls_pw=0.0`, all other settings identical) and the same evaluation/error-analysis
commands. It is registered as `EXP-0002` in `thesis/experiments/README.md`.

## Reproducibility

- Config: `configs/train/baseline.yaml`; ablation: `configs/train/ablation_cls_pw0.yaml`.
- Dataset checksums: `data/processed/adverse16k/checksums.sha1`.
- Run metadata records git commit, package versions, GPU, config, and VRAM.
- Commands and full config are recorded in each experiment record.
