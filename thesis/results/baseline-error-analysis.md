---
title: Baseline Error Analysis — YOLOv8n
status: validated
date: 2026-11-07
sources:
  - runs/eval/baseline_yolov8n_seed42/error_analysis/error_analysis.json
  - thesis/results/baseline.md
---

# Baseline Error Analysis — YOLOv8n

- Weights: `runs/train/baseline_yolov8n_seed42/weights/best.pt`
- Splits: val and test
- Confidence: 0.25 (ceiling pass: 0.001)
- Matching IoU: 0.5
- Same-class matching for TP/FP/FN; any-class matching for confusion.

## Overall

| Split | GT | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| val | 15,334 | 9,537 | 3,669 | 5,797 | 0.722 | 0.622 |
| test | 16,230 | 10,265 | 4,056 | 5,965 | 0.717 | 0.632 |

## By condition

| Condition | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| fog | val | 2,669 | 2,047 | 601 | 622 | 0.773 | 0.767 | 0.926 |
| fog | test | 3,586 | 2,508 | 733 | 1,078 | 0.774 | 0.699 | 0.893 |
| night | val | 3,745 | 2,083 | 1,049 | 1,662 | 0.665 | 0.556 | 0.775 |
| night | test | 4,055 | 2,435 | 1,143 | 1,620 | 0.681 | 0.600 | 0.838 |
| rain | val | 4,396 | 2,684 | 945 | 1,712 | 0.740 | 0.611 | 0.829 |
| rain | test | 4,113 | 2,506 | 1,052 | 1,607 | 0.704 | 0.609 | 0.816 |
| snow | val | 4,524 | 2,723 | 1,074 | 1,801 | 0.717 | 0.602 | 0.814 |
| snow | test | 4,476 | 2,816 | 1,128 | 1,660 | 0.714 | 0.629 | 0.829 |

## By source

| Source | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ACDC | val | 1,675 | 716 | 333 | 959 | 0.683 | 0.427 | 0.651 |
| ACDC | test | 1,418 | 808 | 415 | 610 | 0.661 | 0.570 | 0.775 |
| BDD100K | val | 10,747 | 6,572 | 2,686 | 4,175 | 0.710 | 0.612 | 0.828 |
| BDD100K | test | 11,113 | 6,811 | 2,845 | 4,302 | 0.705 | 0.613 | 0.830 |
| DAWN | val | 410 | 342 | 85 | 68 | 0.801 | 0.834 | 0.961 |
| DAWN | test | 579 | 473 | 160 | 106 | 0.747 | 0.817 | 0.952 |
| RTTS | val | 2,502 | 1,907 | 565 | 595 | 0.771 | 0.762 | 0.924 |
| RTTS | test | 3,120 | 2,173 | 636 | 947 | 0.774 | 0.696 | 0.895 |

## By class

| Class | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | val | 203 | 51 | 43 | 152 | 0.543 | 0.251 | 0.552 |
| bicycle | test | 162 | 57 | 38 | 105 | 0.600 | 0.352 | 0.617 |
| bus | val | 346 | 147 | 107 | 199 | 0.579 | 0.425 | 0.572 |
| bus | test | 377 | 146 | 146 | 231 | 0.500 | 0.387 | 0.589 |
| car | val | 12,201 | 8,057 | 2,988 | 4,144 | 0.729 | 0.660 | 0.859 |
| car | test | 13,173 | 8,741 | 3,418 | 4,432 | 0.719 | 0.664 | 0.862 |
| motorcycle | val | 119 | 41 | 22 | 78 | 0.651 | 0.345 | 0.546 |
| motorcycle | test | 156 | 83 | 31 | 73 | 0.728 | 0.532 | 0.692 |
| person | val | 2,465 | 1,241 | 509 | 1,224 | 0.709 | 0.503 | 0.748 |
| person | test | 2,362 | 1,238 | 423 | 1,124 | 0.745 | 0.524 | 0.800 |

## By object size (relative area)

| Size | Split | GT | TP | FN | Recall |
|---|---|---|---:|---:|---:|---:|
| tiny (<0.001) | val | 5,193 | 1,596 | 3,597 | 0.307 |
| tiny (<0.001) | test | 5,328 | 1,779 | 3,549 | 0.334 |
| small (0.001–0.01) | val | 6,656 | 4,779 | 1,877 | 0.718 |
| small (0.001–0.01) | test | 7,000 | 4,959 | 2,041 | 0.708 |
| medium (0.01–0.05) | val | 2,380 | 2,123 | 257 | 0.892 |
| medium (0.01–0.05) | test | 2,685 | 2,395 | 290 | 0.892 |
| large (>0.05) | val | 1,105 | 1,039 | 66 | 0.940 |
| large (>0.05) | test | 1,217 | 1,132 | 85 | 0.930 |

## Confusion and background false positives

Top background false positives (unmatched predictions):

| Predicted class | Count |
|---|---:|
| car | 6,240 |
| person | 920 |
| bus | 191 |
| bicycle | 67 |
| motorcycle | 48 |

Top off-diagonal matched confusions:

| GT class | Predicted class | Count |
|---|---|---:|
| bus | car | 175 |
| car | bus | 92 |
| person | car | 21 |
| motorcycle | bicycle | 12 |
| bicycle | motorcycle | 11 |
| motorcycle | person | 10 |
| bicycle | person | 9 |

## Artifacts

- Raw report: `runs/eval/baseline_yolov8n_seed42/error_analysis/error_analysis.json`
- CSVs: `error_by_condition.csv`, `error_by_source.csv`, `error_by_class.csv`, `error_by_size.csv`
- Qualitative overlays: `runs/eval/baseline_yolov8n_seed42/error_analysis/overlays/` (40 images)

_Facts only; interpretation belongs in `thesis/findings/baseline-analysis.md`._
