---
title: Weather Gate Error Analysis
status: validated
date: 2026-11-10
sources:
  - runs/eval/weather_gate_seed42/error_analysis/error_analysis.json
  - thesis/results/weather_gate.md
---

# Weather Gate Error Analysis

- Weights: `runs/train/weather_gate_seed42/weights/best.pt`
- Splits: val and test
- Confidence: 0.25 (ceiling pass: 0.001)
- Matching IoU: 0.5
- Same-class matching for TP/FP/FN; any-class matching for confusion.

## Overall

| Split | GT | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| val | 15,334 | 9,518 | 3,769 | 5,816 | 0.716 | 0.621 |
| test | 16,230 | 10,306 | 4,162 | 5,924 | 0.712 | 0.635 |

## By condition

| Condition | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fog | val | 2,669 | 2,049 | 692 | 620 | 0.748 | 0.768 | 0.924 |
| fog | test | 3,586 | 2,537 | 807 | 1,049 | 0.759 | 0.707 | 0.890 |
| night | val | 3,745 | 2,070 | 1,034 | 1,675 | 0.667 | 0.553 | 0.779 |
| night | test | 4,055 | 2,429 | 1,135 | 1,626 | 0.682 | 0.599 | 0.840 |
| rain | val | 4,396 | 2,701 | 973 | 1,695 | 0.735 | 0.614 | 0.832 |
| rain | test | 4,113 | 2,518 | 1,045 | 1,595 | 0.707 | 0.612 | 0.820 |
| snow | val | 4,524 | 2,698 | 1,070 | 1,826 | 0.716 | 0.596 | 0.819 |
| snow | test | 4,476 | 2,822 | 1,175 | 1,654 | 0.706 | 0.630 | 0.829 |

## By source

| Source | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ACDC | val | 1,675 | 712 | 340 | 963 | 0.677 | 0.425 | 0.650 |
| ACDC | test | 1,418 | 808 | 362 | 610 | 0.691 | 0.570 | 0.779 |
| BDD100K | val | 10,747 | 6,553 | 2,682 | 4,194 | 0.710 | 0.610 | 0.833 |
| BDD100K | test | 11,113 | 6,813 | 2,912 | 4,300 | 0.701 | 0.613 | 0.831 |
| DAWN | val | 410 | 342 | 98 | 68 | 0.777 | 0.834 | 0.978 |
| DAWN | test | 579 | 484 | 188 | 95 | 0.720 | 0.836 | 0.957 |
| RTTS | val | 2,502 | 1,911 | 649 | 591 | 0.746 | 0.764 | 0.921 |
| RTTS | test | 3,120 | 2,201 | 700 | 919 | 0.759 | 0.705 | 0.891 |

## By class

| Class | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | val | 203 | 51 | 32 | 152 | 0.614 | 0.251 | 0.522 |
| bicycle | test | 162 | 63 | 37 | 99 | 0.630 | 0.389 | 0.648 |
| bus | val | 346 | 151 | 114 | 195 | 0.570 | 0.436 | 0.607 |
| bus | test | 377 | 152 | 124 | 225 | 0.551 | 0.403 | 0.613 |
| car | val | 12,201 | 8,030 | 3,028 | 4,171 | 0.726 | 0.658 | 0.861 |
| car | test | 13,173 | 8,748 | 3,506 | 4,425 | 0.714 | 0.664 | 0.861 |
| motorcycle | val | 119 | 40 | 25 | 79 | 0.615 | 0.336 | 0.529 |
| motorcycle | test | 156 | 83 | 32 | 73 | 0.722 | 0.532 | 0.705 |
| person | val | 2,465 | 1,246 | 570 | 1,219 | 0.686 | 0.505 | 0.753 |
| person | test | 2,362 | 1,260 | 463 | 1,102 | 0.731 | 0.533 | 0.799 |

## By object size (relative area)

| Size | Split | GT | TP | FN | Recall |
|---|---|---:|---:|---:|---:|
| tiny (<0.001) | val | 5,193 | 1,614 | 3,579 | 0.311 |
| tiny (<0.001) | test | 5,328 | 1,789 | 3,539 | 0.336 |
| small (0.001–0.01) | val | 6,656 | 4,746 | 1,910 | 0.713 |
| small (0.001–0.01) | test | 7,000 | 4,964 | 2,036 | 0.709 |
| medium (0.01–0.05) | val | 2,380 | 2,121 | 259 | 0.891 |
| medium (0.01–0.05) | test | 2,685 | 2,412 | 273 | 0.898 |
| large (>0.05) | val | 1,105 | 1,037 | 68 | 0.938 |
| large (>0.05) | test | 1,217 | 1,141 | 76 | 0.938 |

## Confusion and background false positives

Top background false positives (unmatched predictions):

| Predicted class | Count |
|---|---:|
| car | 6,381 |
| person | 1,018 |
| bus | 165 |
| bicycle | 57 |
| motorcycle | 45 |

Top off-diagonal matched confusions:

| GT class | Predicted class | Count |
|---|---|---:|
| bus | car | 162 |
| car | bus | 100 |
| person | car | 18 |
| motorcycle | person | 10 |
| motorcycle | bicycle | 10 |
| car | person | 9 |
| bicycle | motorcycle | 8 |
| motorcycle | car | 7 |
| person | motorcycle | 6 |
| bicycle | person | 4 |

## Artifacts

- Raw report: `runs/eval/weather_gate_seed42/error_analysis/error_analysis.json`
- CSVs: `error_by_condition.csv`, `error_by_source.csv`, `error_by_class.csv`, `error_by_size.csv`
- Qualitative overlays: `runs/eval/weather_gate_seed42/error_analysis/overlays/`

_Facts only; interpretation belongs in `thesis/findings/weather-gate-analysis.md`._
