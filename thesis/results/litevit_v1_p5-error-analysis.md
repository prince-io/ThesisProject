---
title: V1 Error Analysis — P5 LiteViT
status: validated
date: 2026-11-09
sources:
  - runs/eval/litevit_v1_p5_seed42_v2/error_analysis/error_analysis.json
  - thesis/results/litevit_v1_p5.md
---

# V1 Error Analysis — P5 LiteViT

- Weights: `runs/train/litevit_v1_p5_seed42_v2/weights/best.pt`
- Splits: val and test
- Confidence: 0.25 (ceiling pass: 0.001)
- Matching IoU: 0.5
- Same-class matching for TP/FP/FN; any-class matching for confusion.

## Overall

| Split | GT | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| val | 15,334 | 8,997 | 3,856 | 6,337 | 0.700 | 0.587 |
| test | 16,230 | 9,687 | 4,223 | 6,543 | 0.696 | 0.597 |

## By condition

| Condition | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fog | val | 2,669 | 1,971 | 798 | 698 | 0.712 | 0.738 | 0.910 |
| fog | test | 3,586 | 2,413 | 949 | 1,173 | 0.718 | 0.673 | 0.875 |
| night | val | 3,745 | 1,958 | 1,049 | 1,787 | 0.651 | 0.523 | 0.748 |
| night | test | 4,055 | 2,289 | 1,051 | 1,766 | 0.685 | 0.564 | 0.826 |
| rain | val | 4,396 | 2,552 | 988 | 1,844 | 0.721 | 0.581 | 0.813 |
| rain | test | 4,113 | 2,350 | 1,058 | 1,763 | 0.690 | 0.571 | 0.796 |
| snow | val | 4,524 | 2,516 | 1,021 | 2,008 | 0.711 | 0.556 | 0.795 |
| snow | test | 4,476 | 2,635 | 1,165 | 1,841 | 0.693 | 0.589 | 0.808 |

## By source

| Source | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ACDC | val | 1,675 | 684 | 414 | 991 | 0.623 | 0.408 | 0.610 |
| ACDC | test | 1,418 | 765 | 431 | 653 | 0.640 | 0.539 | 0.751 |
| BDD100K | val | 10,747 | 6,138 | 2,581 | 4,609 | 0.704 | 0.571 | 0.811 |
| BDD100K | test | 11,113 | 6,356 | 2,747 | 4,757 | 0.698 | 0.572 | 0.813 |
| DAWN | val | 410 | 343 | 115 | 67 | 0.749 | 0.837 | 0.956 |
| DAWN | test | 579 | 476 | 222 | 103 | 0.682 | 0.822 | 0.955 |
| RTTS | val | 2,502 | 1,832 | 746 | 670 | 0.711 | 0.732 | 0.905 |
| RTTS | test | 3,120 | 2,090 | 823 | 1,030 | 0.717 | 0.670 | 0.874 |

## By class

| Class | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | val | 203 | 39 | 20 | 164 | 0.661 | 0.192 | 0.453 |
| bicycle | test | 162 | 47 | 29 | 115 | 0.618 | 0.290 | 0.636 |
| bus | val | 346 | 98 | 54 | 248 | 0.645 | 0.283 | 0.448 |
| bus | test | 377 | 105 | 77 | 272 | 0.577 | 0.279 | 0.438 |
| car | val | 12,201 | 7,669 | 3,222 | 4,532 | 0.704 | 0.629 | 0.846 |
| car | test | 13,173 | 8,277 | 3,525 | 4,896 | 0.701 | 0.628 | 0.848 |
| motorcycle | val | 119 | 30 | 24 | 89 | 0.556 | 0.252 | 0.437 |
| motorcycle | test | 156 | 70 | 21 | 86 | 0.769 | 0.449 | 0.667 |
| person | val | 2,465 | 1,161 | 536 | 1,304 | 0.684 | 0.471 | 0.720 |
| person | test | 2,362 | 1,188 | 571 | 1,174 | 0.675 | 0.503 | 0.777 |

## By object size (relative area)

| Size | Split | GT | TP | FN | Recall |
|---|---|---:|---:|---:|---:|
| tiny (<0.001) | val | 5,193 | 1,290 | 3,903 | 0.248 |
| tiny (<0.001) | test | 5,328 | 1,436 | 3,892 | 0.270 |
| small (0.001–0.01) | val | 6,656 | 4,616 | 2,040 | 0.694 |
| small (0.001–0.01) | test | 7,000 | 4,768 | 2,232 | 0.681 |
| medium (0.01–0.05) | val | 2,380 | 2,073 | 307 | 0.871 |
| medium (0.01–0.05) | test | 2,685 | 2,356 | 329 | 0.877 |
| large (>0.05) | val | 1,105 | 1,018 | 87 | 0.921 |
| large (>0.05) | test | 1,217 | 1,127 | 90 | 0.926 |

## Confusion and background false positives

Top background false positives (unmatched predictions):

| Predicted class | Count |
|---|---:|
| car | 6,516 |
| person | 1,097 |
| bus | 98 |
| bicycle | 46 |
| motorcycle | 36 |

Top off-diagonal matched confusions:

| GT class | Predicted class | Count |
|---|---|---:|
| bus | car | 223 |
| car | bus | 63 |
| person | car | 26 |
| bicycle | motorcycle | 11 |
| motorcycle | person | 10 |
| motorcycle | car | 10 |
| bicycle | person | 8 |
| motorcycle | bicycle | 7 |
| bicycle | car | 5 |
| person | motorcycle | 3 |

## Artifacts

- Raw report: `runs/eval/litevit_v1_p5_seed42_v2/error_analysis/error_analysis.json`
- CSVs: `error_by_condition.csv`, `error_by_source.csv`, `error_by_class.csv`, `error_by_size.csv`
- Qualitative overlays: `runs/eval/litevit_v1_p5_seed42_v2/error_analysis/overlays/`

_Facts only; interpretation belongs in `thesis/findings/v1-p5-analysis.md`._
