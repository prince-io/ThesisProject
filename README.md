# B.Tech Thesis — Lightweight Adverse-Weather Object Detection

Final-year thesis project. Target: YOLO-LiteViT-WG (YOLOv8n backbone + linear-attention ViT at neck P4/P5 + lightweight weather channel gate), trainable at 640x640 on an RTX 3050 6GB.

## Repository layout

- `ref/` — non-authoritative input inbox (ideas, suggestions, papers). Never cited as truth.
- `thesis/` — curated source of truth: methodology, decisions, experiments, results, findings, related work, bibliography.
- `manuscript/` — derived LaTeX thesis, written from `thesis/`.
- `runs/` — full raw artifacts (untracked). Curated summaries live in `thesis/experiments` and `thesis/results`.

## Status

Environment and the `adverse16k` dataset are ready. The YOLOv8n baseline is trained,
evaluated, and error-analyzed (`EXP-0001`). The `cls_pw=0.0` ablation (`EXP-0002`) is
complete: validation is tied, test mAP is slightly lower, and `cls_pw=0.5` is retained
for later comparisons (`ADR-0008`). Architecture literature grounding is complete in
`thesis/bibliography/` and `thesis/related-work/`. V1 (P5-only LiteViT, `EXP-0003`)
early-stopped at epoch 38 and underperformed the baseline on test. Gate-only weather
adaptation (L-WG on P3/P4/P5, `EXP-0004`) is complete: test mAP50 0.5444 /
mAP50-95 0.3301 vs baseline 0.5428 / 0.3287; validation is slightly lower but
training is stable. A+B (gate + regularized P5 attention, `EXP-0005`) is complete: stable but neutral on
test. TTA was worse and is excluded. Kitchen-sink (gate + Sobel edge + spatial
CBAM, `EXP-0006`) is implemented; training/evaluation are pending. V2/V3 attention
work has not started.

## Code layout

- `src/litevit/` — implementation package (editable install).
- `configs/` — training and experiment YAML configs (`configs/datasets/adverse16k.yaml`).
- `scripts/` — entrypoints and utilities.
- `tests/` — pytest suite.
- `data/` — datasets (gitignored). Processed benchmark: `data/processed/adverse16k`.
- `runs/` — full run artifacts (gitignored).
- `pyproject.toml` — dependencies and tooling.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install -e ".[dev,datasets]"
```

Verify CUDA: `.venv/bin/python -c "import torch; print(torch.cuda.is_available())"`.
Run checks: `.venv/bin/ruff check .` and `.venv/bin/pytest`.

## Dataset (adverse16k)

16,000 images, 4,000 per condition (fog, rain, snow, night), 5 classes
(`person, car, bus, bicycle, motorcycle`), split 12,800/1,600/1,600.
Built from ACDC + DAWN + BDD100K + RTTS as an **independent copy** with
`checksums.sha1`, using **sequence-aware splits** and DAWN near-duplicate grouping
(zero sequence leakage). See `thesis/experiments/dataset-card-adverse16k.md`.

```bash
.venv/bin/python scripts/build_adverse16k.py --root . --out data/processed/adverse16k \
  --force --copy-images
.venv/bin/python scripts/verify_adverse16k.py --root . --out data/processed/adverse16k \
  --report runs/dataset_checks/adverse16k --require-independent --checksums
.venv/bin/python scripts/audit_adverse16k.py --root . --out data/processed/adverse16k \
  --report runs/dataset_checks/adverse16k
.venv/bin/python scripts/prepare_acdc_test.py --root . --out data/unlabeled/acdc_test \
  --report runs/dataset_checks/acdc_test --force
```

The unlabeled **ACDC test** set (`data/unlabeled/acdc_test`, 2,000 images) is for
demos and final qualitative testing; its ground truth is withheld by ACDC.

## Baseline training (Stage 2A)

```bash
# 1. Smoke test (1 epoch)
.venv/bin/python scripts/train.py \
  --config configs/train/baseline.yaml \
  --name smoke_yolov8n --epochs 1

# 2. Full baseline (100 epochs)
.venv/bin/python scripts/train.py \
  --config configs/train/baseline.yaml \
  --name baseline_yolov8n_seed42

# 3. Evaluate val + test, per condition/source, with figures
.venv/bin/python scripts/evaluate.py \
  --weights runs/train/baseline_yolov8n_seed42/weights/best.pt \
  --data configs/datasets/adverse16k.yaml \
  --splits val test \
  --out runs/eval/baseline_yolov8n_seed42 \
  --subsets data/processed/adverse16k/subsets \
  --figures-dir thesis/results/figures/baseline

# 4. Error analysis (val + test)
.venv/bin/python scripts/error_analysis.py \
  --weights runs/train/baseline_yolov8n_seed42/weights/best.pt \
  --out runs/eval/baseline_yolov8n_seed42/error_analysis \
  --splits val test

# 5. Generate curated thesis records
.venv/bin/python scripts/document_results.py \
  --train-run runs/train/baseline_yolov8n_seed42 \
  --eval-run runs/eval/baseline_yolov8n_seed42 \
  --figures-dir thesis/results/figures/baseline \
  --vram-report runs/train/vram_calibration/vram.json

# 6. Optional export
.venv/bin/python scripts/export.py \
  --weights runs/train/baseline_yolov8n_seed42/weights/best.pt \
  --format onnx
```

Protocol: YOLOv8n, 640, batch 4, AMP, `nbs=8`, AdamW 2e-4 + cosine, 100 epochs,
seed 42, `cls_pw=0.5`. See `thesis/decisions/ADR-0006-baseline-protocol.md`.
VRAM is reported as training PyTorch peak, training GPU-used peak (`nvidia-smi`
calibration), and inference peak; see `ADR-0007`.

## Ablation: cls_pw=0.0

```bash
.venv/bin/python scripts/train.py \
  --config configs/train/ablation_cls_pw0.yaml \
  --name ablation_cls_pw0_seed42

.venv/bin/python scripts/evaluate.py \
  --weights runs/train/ablation_cls_pw0_seed42/weights/best.pt \
  --data configs/datasets/adverse16k.yaml \
  --splits val test \
  --out runs/eval/ablation_cls_pw0_seed42 \
  --subsets data/processed/adverse16k/subsets \
  --figures-dir thesis/results/figures/ablation_cls_pw0

.venv/bin/python scripts/error_analysis.py \
  --weights runs/train/ablation_cls_pw0_seed42/weights/best.pt \
  --out runs/eval/ablation_cls_pw0_seed42/error_analysis \
  --splits val test

.venv/bin/python scripts/document_results.py \
  --train-run runs/train/ablation_cls_pw0_seed42 \
  --eval-run runs/eval/ablation_cls_pw0_seed42 \
  --results thesis/results/ablation_cls_pw0.md \
  --experiment thesis/experiments/EXP-0002-cls-pw0.md \
  --figures-dir thesis/results/figures/ablation_cls_pw0 \
  --vram-report runs/train/ablation_cls_pw0_seed42/vram.json \
  --title "Ablation Results — cls_pw=0.0" \
  --experiment-title "EXP-0002 — cls_pw=0.0 Ablation"
```

## V1: P5 LiteViT

V1 inserts one identity-initialized LiteViT block after the P5/SPPF backbone layer
(before PAN/FPN). See `thesis/methodology/architecture.md` and
`thesis/experiments/EXP-0003-v1-p5.md`.

```bash
# Optional one-epoch smoke run
.venv/bin/python scripts/train.py \
  --config configs/train/litevit_v1_p5.yaml \
  --name litevit_v1_p5_smoke --epochs 1

# Full V1 run
.venv/bin/python scripts/train.py \
  --config configs/train/litevit_v1_p5.yaml \
  --name litevit_v1_p5_seed42_v2

.venv/bin/python scripts/evaluate.py \
  --weights runs/train/litevit_v1_p5_seed42_v2/weights/best.pt \
  --data configs/datasets/adverse16k.yaml \
  --splits val test \
  --out runs/eval/litevit_v1_p5_seed42_v2 \
  --subsets data/processed/adverse16k/subsets \
  --figures-dir thesis/results/figures/litevit_v1_p5

.venv/bin/python scripts/error_analysis.py \
  --weights runs/train/litevit_v1_p5_seed42_v2/weights/best.pt \
  --out runs/eval/litevit_v1_p5_seed42_v2/error_analysis \
  --splits val test

.venv/bin/python scripts/document_results.py \
  --train-run runs/train/litevit_v1_p5_seed42_v2 \
  --eval-run runs/eval/litevit_v1_p5_seed42_v2 \
  --results thesis/results/litevit_v1_p5.md \
  --experiment thesis/experiments/EXP-0003-v1-p5.md \
  --figures-dir thesis/results/figures/litevit_v1_p5 \
  --vram-report runs/train/litevit_v1_p5_seed42_v2/vram.json \
  --title "V1 Results — P5 LiteViT" \
  --experiment-title "EXP-0003 — V1 P5 LiteViT"
```

## Weather gate only (EXP-0004)

YOLOv8n + Lightweight Weather Gate on neck P3/P4/P5. The gate is identity-initialized
and adds 43,568 parameters. See `thesis/methodology/weather-gate.md` and
`thesis/experiments/EXP-0004-weather-gate.md`.

```bash
# Optional 5-epoch smoke run
.venv/bin/python scripts/train.py \
  --config configs/train/weather_gate.yaml \
  --name weather_gate_smoke --epochs 5

# Full gate-only run
.venv/bin/python scripts/train.py \
  --config configs/train/weather_gate.yaml \
  --name weather_gate_seed42

.venv/bin/python scripts/evaluate.py \
  --weights runs/train/weather_gate_seed42/weights/best.pt \
  --data configs/datasets/adverse16k.yaml \
  --splits val test \
  --out runs/eval/weather_gate_seed42 \
  --subsets data/processed/adverse16k/subsets \
  --figures-dir thesis/results/figures/weather_gate

.venv/bin/python scripts/error_analysis.py \
  --weights runs/train/weather_gate_seed42/weights/best.pt \
  --out runs/eval/weather_gate_seed42/error_analysis \
  --splits val test

.venv/bin/python scripts/document_results.py \
  --train-run runs/train/weather_gate_seed42 \
  --eval-run runs/eval/weather_gate_seed42 \
  --results thesis/results/weather_gate.md \
  --experiment thesis/experiments/EXP-0004-weather-gate.md \
  --figures-dir thesis/results/figures/weather_gate \
  --vram-report runs/train/weather_gate_seed42/vram.json \
  --title "Weather Gate Results" \
  --experiment-title "EXP-0004 — Gate-Only Weather Adaptation"
```

## A+B: gate + regularized P5 attention (EXP-0005)

YOLOv8n with a regularized P5 LiteViT block and P3/P4/P5 weather gates. The block
uses DropPath 0.2, dropout 0.1, MLP ratio 1.0, and FP32 execution. Total: 3,583,951
parameters and 9.099 GFLOPs. See `thesis/methodology/gate-p5.md` and
`thesis/experiments/EXP-0005-gate-p5.md`.

```bash
# Optional 5-epoch smoke run
.venv/bin/python scripts/train.py \
  --config configs/train/gate_p5.yaml \
  --name gate_p5_smoke --epochs 5

# Full A+B run
.venv/bin/python scripts/train.py \
  --config configs/train/gate_p5.yaml \
  --name gate_p5_seed42

.venv/bin/python scripts/evaluate.py \
  --weights runs/train/gate_p5_seed42/weights/best.pt \
  --data configs/datasets/adverse16k.yaml \
  --splits val test \
  --out runs/eval/gate_p5_seed42 \
  --subsets data/processed/adverse16k/subsets \
  --figures-dir thesis/results/figures/gate_p5

.venv/bin/python scripts/error_analysis.py \
  --weights runs/train/gate_p5_seed42/weights/best.pt \
  --out runs/eval/gate_p5_seed42/error_analysis \
  --splits val test

.venv/bin/python scripts/document_results.py \
  --train-run runs/train/gate_p5_seed42 \
  --eval-run runs/eval/gate_p5_seed42 \
  --results thesis/results/gate_p5.md \
  --experiment thesis/experiments/EXP-0005-gate-p5.md \
  --figures-dir thesis/results/figures/gate_p5 \
  --vram-report runs/train/gate_p5_seed42/vram.json \
  --title "A+B Results — Gate + Regularized P5 Attention" \
  --experiment-title "EXP-0005 — Gate + Regularized P5 Attention"
```

## Kitchen sink: gate + Sobel edge + spatial CBAM (EXP-0006)

YOLOv8n with P3/P4 Sobel edge residuals, P3/P4/P5 channel gates, and spatial CBAM.
No P5 attention. Total: 3,203,704 parameters and 8.895 GFLOPs. See
`thesis/methodology/gate-edge-cbam.md` and
`thesis/experiments/EXP-0006-gate-edge-cbam.md`.

```bash
# Optional 5-epoch smoke run
.venv/bin/python scripts/train.py \
  --config configs/train/gate_edge_cbam.yaml \
  --name gate_edge_cbam_smoke --epochs 5

# Full kitchen-sink run
.venv/bin/python scripts/train.py \
  --config configs/train/gate_edge_cbam.yaml \
  --name gate_edge_cbam_seed42

.venv/bin/python scripts/evaluate.py \
  --weights runs/train/gate_edge_cbam_seed42/weights/best.pt \
  --data configs/datasets/adverse16k.yaml \
  --splits val test \
  --out runs/eval/gate_edge_cbam_seed42 \
  --subsets data/processed/adverse16k/subsets \
  --figures-dir thesis/results/figures/gate_edge_cbam

.venv/bin/python scripts/error_analysis.py \
  --weights runs/train/gate_edge_cbam_seed42/weights/best.pt \
  --out runs/eval/gate_edge_cbam_seed42/error_analysis \
  --splits val test

.venv/bin/python scripts/document_results.py \
  --train-run runs/train/gate_edge_cbam_seed42 \
  --eval-run runs/eval/gate_edge_cbam_seed42 \
  --results thesis/results/gate_edge_cbam.md \
  --experiment thesis/experiments/EXP-0006-gate-edge-cbam.md \
  --figures-dir thesis/results/figures/gate_edge_cbam \
  --vram-report runs/train/gate_edge_cbam_seed42/vram.json \
  --title "Gate + Edge + CBAM Results" \
  --experiment-title "EXP-0006 — Gate + Sobel Edge + Spatial CBAM"
```

## Conventions

- Every durable decision is an append-only ADR in `thesis/decisions/`.
- Curated docs use YAML front matter: `title`, `status` (`draft` | `validated` | `superseded`), `date`, `sources`.
- Raw datasets live under `ref/` (untracked) and are retained for now; processed and test copies live under `data/` and are independent.
- `ref/*.pdf` is untracked.
