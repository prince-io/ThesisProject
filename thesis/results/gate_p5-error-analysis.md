---
title: A+B Error Analysis — Gate + Regularized P5 Attention
status: validated
date: 2026-11-10
sources:
  - runs/eval/gate_p5_seed42/error_analysis/error_analysis.json
  - thesis/results/gate_p5.md
---

# A+B Error Analysis — Gate + Regularized P5 Attention

- Weights: `runs/train/gate_p5_seed42/weights/best.pt`
- Splits: val and test
- Confidence: 0.25 (ceiling pass: 0.001)
- Matching IoU: 0.5

## Overall

| Split | GT | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| val | 15,334 | 9,505 | 3,559 | 5,829 | 0.728 | 0.620 |
| test | 16,230 | 10,195 | 4,063 | 6,035 | 0.715 | 0.628 |

## By condition

| Condition | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fog | val | 2,669 | 2,057 | 646 | 612 | 0.761 | 0.771 | 0.927 |
| fog | test | 3,586 | 2,504 | 794 | 1,082 | 0.759 | 0.698 | 0.890 |
| night | val | 3,745 | 2,078 | 984 | 1,667 | 0.679 | 0.555 | 0.772 |
| night | test | 4,055 | 2,398 | 1,081 | 1,657 | 0.689 | 0.591 | 0.835 |
| rain | val | 4,396 | 2,675 | 935 | 1,721 | 0.741 | 0.609 | 0.828 |
| rain | test | 4,113 | 2,506 | 1,045 | 1,607 | 0.706 | 0.609 | 0.818 |
| snow | val | 4,524 | 2,695 | 994 | 1,829 | 0.731 | 0.596 | 0.816 |
| snow | test | 4,476 | 2,787 | 1,143 | 1,689 | 0.709 | 0.623 | 0.825 |

## By class

| Class | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bicycle | val | 203 | 49 | 31 | 154 | 0.613 | 0.241 | 0.542 |
| bicycle | test | 162 | 58 | 32 | 104 | 0.644 | 0.358 | 0.667 |
| bus | val | 346 | 142 | 95 | 204 | 0.599 | 0.410 | 0.561 |
| bus | test | 377 | 146 | 116 | 231 | 0.557 | 0.387 | 0.568 |
| car | val | 12,201 | 8,022 | 2,900 | 4,179 | 0.734 | 0.657 | 0.859 |
| car | test | 13,173 | 8,669 | 3,443 | 4,504 | 0.716 | 0.658 | 0.859 |
| motorcycle | val | 119 | 42 | 32 | 77 | 0.568 | 0.353 | 0.513 |
| motorcycle | test | 156 | 81 | 36 | 75 | 0.692 | 0.519 | 0.679 |
| person | val | 2,465 | 1,250 | 501 | 1,215 | 0.714 | 0.507 | 0.752 |
| person | test | 2,362 | 1,241 | 436 | 1,121 | 0.740 | 0.525 | 0.800 |

## By object size (relative area)

| Size | Split | GT | TP | FN | Recall |
|---|---|---:|---:|---:|---:|
| tiny (<0.001) | val | 5,193 | 1,583 | 3,610 | 0.305 |
| tiny (<0.001) | test | 5,328 | 1,729 | 3,599 | 0.325 |
| small (0.001–0.01) | val | 6,656 | 4,773 | 1,883 | 0.717 |
| small (0.001–0.01) | test | 7,000 | 4,921 | 2,079 | 0.703 |
| medium (0.01–0.05) | val | 2,380 | 2,113 | 267 | 0.888 |
| medium (0.01–0.05) | test | 2,685 | 2,405 | 280 | 0.896 |
| large (>0.05) | val | 1,105 | 1,036 | 69 | 0.938 |
| large (>0.05) | test | 1,217 | 1,140 | 77 | 0.937 |

## Confusion and background false positives

Top background false positives (unmatched predictions):

| Predicted class | Count |
|---|---:|
| car | 6,174 |
| person | 927 |
| bus | 157 |
| motorcycle | 63 |
| bicycle | 49 |

Top off-diagonal matched confusions:

| GT class | Predicted class | Count |
|---|---|---:|
| bus | car | 175 |
| car | bus | 86 |
| person | car | 15 |
| motorcycle | bicycle | 12 |
| motorcycle | person | 8 |
| motorcycle | car | 8 |
| bicycle | motorcycle | 8 |
| bicycle | person | 6 |
| person | motorcycle | 4 |
| car | person | 4 |

## Artifacts

- Raw report: `runs/eval/gate_p5_seed42/error_analysis/error_analysis.json`
- CSVs: `error_by_condition.csv`, `error_by_source.csv`, `error_by_class.csv`, `error_by_size.csv`
- Qualitative overlays: `runs/eval/gate_p5_seed42/error_analysis/overlays/`

_Facts only; interpretation belongs in `thesis/findings/gate-p5-analysis.md`._
