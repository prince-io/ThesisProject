---
title: Ablation Error Analysis — cls_pw=0.0
status: validated
date: 2026-11-08
sources:
  - runs/eval/ablation_cls_pw0_seed42/error_analysis/error_analysis.json
  - thesis/results/ablation_cls_pw0.md
---

# Ablation Error Analysis — cls_pw=0.0

- Weights: `runs/train/ablation_cls_pw0_seed42/weights/best.pt`
- Splits: val and test
- Confidence: 0.25 (ceiling pass: 0.001)
- Matching IoU: 0.5
- Same-class matching for TP/FP/FN; any-class matching for confusion.

## Overall

| Split | GT | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| val | 15,334 | 9,691 | 3,707 | 5,643 | 0.723 | 0.632 |
| test | 16,230 | 10,375 | 3,885 | 5,855 | 0.728 | 0.639 |

## By condition

| Condition | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fog | val | 2,669 | 2,097 | 682 | 572 | 0.755 | 0.786 | 0.925 |
| fog | test | 3,586 | 2,538 | 762 | 1,048 | 0.769 | 0.708 | 0.896 |
| night | val | 3,745 | 2,111 | 1,020 | 1,634 | 0.674 | 0.564 | 0.775 |
| night | test | 4,055 | 2,446 | 1,026 | 1,609 | 0.704 | 0.603 | 0.842 |
| rain | val | 4,396 | 2,739 | 982 | 1,657 | 0.736 | 0.623 | 0.834 |
| rain | test | 4,113 | 2,548 | 1,000 | 1,565 | 0.718 | 0.619 | 0.819 |
| snow | val | 4,524 | 2,744 | 1,023 | 1,780 | 0.728 | 0.607 | 0.816 |
| snow | test | 4,476 | 2,843 | 1,097 | 1,633 | 0.722 | 0.635 | 0.831 |

## By source

| Source | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ACDC | val | 1,675 | 728 | 360 | 947 | 0.669 | 0.435 | 0.648 |
| ACDC | test | 1,418 | 807 | 334 | 611 | 0.707 | 0.569 | 0.779 |
| BDD100K | val | 10,747 | 6,663 | 2,619 | 4,084 | 0.718 | 0.620 | 0.832 |
| BDD100K | test | 11,113 | 6,883 | 2,717 | 4,230 | 0.717 | 0.619 | 0.833 |
| DAWN | val | 410 | 347 | 90 | 63 | 0.794 | 0.846 | 0.963 |
| DAWN | test | 579 | 484 | 168 | 95 | 0.742 | 0.836 | 0.967 |
| RTTS | val | 2,502 | 1,953 | 638 | 549 | 0.754 | 0.781 | 0.921 |
| RTTS | test | 3,120 | 2,201 | 666 | 919 | 0.768 | 0.705 | 0.896 |

## By class

| Class | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | val | 203 | 50 | 37 | 153 | 0.575 | 0.246 | 0.552 |
| bicycle | test | 162 | 53 | 39 | 109 | 0.576 | 0.327 | 0.660 |
| bus | val | 346 | 152 | 103 | 194 | 0.596 | 0.439 | 0.587 |
| bus | test | 377 | 157 | 138 | 220 | 0.532 | 0.416 | 0.602 |
| car | val | 12,201 | 8,162 | 2,978 | 4,039 | 0.733 | 0.669 | 0.860 |
| car | test | 13,173 | 8,832 | 3,240 | 4,341 | 0.732 | 0.670 | 0.864 |
| motorcycle | val | 119 | 43 | 33 | 76 | 0.566 | 0.361 | 0.513 |
| motorcycle | test | 156 | 82 | 34 | 74 | 0.707 | 0.526 | 0.712 |
| person | val | 2,465 | 1,284 | 556 | 1,181 | 0.698 | 0.521 | 0.753 |
| person | test | 2,362 | 1,251 | 434 | 1,111 | 0.742 | 0.530 | 0.796 |

## By object size (relative area)

| Size | Split | GT | TP | FN | Recall |
|---|---|---:|---:|---:|---:|
| tiny (<0.001) | val | 5,193 | 1,668 | 3,525 | 0.321 |
| tiny (<0.001) | test | 5,328 | 1,809 | 3,519 | 0.340 |
| small (0.001–0.01) | val | 6,656 | 4,846 | 1,810 | 0.728 |
| small (0.001–0.01) | test | 7,000 | 5,013 | 1,987 | 0.716 |
| medium (0.01–0.05) | val | 2,380 | 2,132 | 248 | 0.896 |
| medium (0.01–0.05) | test | 2,685 | 2,411 | 274 | 0.898 |
| large (>0.05) | val | 1,105 | 1,045 | 60 | 0.946 |
| large (>0.05) | test | 1,217 | 1,142 | 75 | 0.938 |

## Confusion and background false positives

Top background false positives (unmatched predictions):

| Predicted class | Count |
|---|---:|
| car | 6,062 |
| person | 972 |
| bus | 176 |
| bicycle | 68 |
| motorcycle | 59 |

Top off-diagonal matched confusions:

| GT class | Predicted class | Count |
|---|---|---:|
| bus | car | 162 |
| car | bus | 94 |
| person | car | 15 |
| bicycle | motorcycle | 12 |
| motorcycle | person | 10 |
| motorcycle | car | 9 |
| motorcycle | bicycle | 8 |
| bicycle | person | 8 |
| car | person | 6 |
| bicycle | car | 4 |

## Artifacts

- Raw report: `runs/eval/ablation_cls_pw0_seed42/error_analysis/error_analysis.json`
- CSVs: `error_by_condition.csv`, `error_by_source.csv`, `error_by_class.csv`, `error_by_size.csv`
- Qualitative overlays: `runs/eval/ablation_cls_pw0_seed42/error_analysis/overlays/` (40 images)

_Facts only; interpretation belongs in `thesis/findings/ablation-analysis.md`._
