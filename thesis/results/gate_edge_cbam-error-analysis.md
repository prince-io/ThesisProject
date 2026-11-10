---
title: Gate + Edge + CBAM Error Analysis
status: validated
date: 2026-11-10
sources:
  - runs/eval/gate_edge_cbam_seed42/error_analysis/error_analysis.json
  - thesis/results/gate_edge_cbam.md
---

# Gate + Edge + CBAM Error Analysis

- Weights: `runs/train/gate_edge_cbam_seed42/weights/best.pt`
- Splits: val and test
- Confidence: 0.25
- Matching IoU: 0.5

## Overall

| Split | GT | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| val | 15,334 | 9,612 | 3,751 | 5,722 | 0.719 | 0.627 |
| test | 16,230 | 10,291 | 4,265 | 5,939 | 0.707 | 0.634 |

## By condition

| Condition | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fog | val | 2,669 | 2,083 | 674 | 586 | 0.756 | 0.780 | 0.928 |
| fog | test | 3,586 | 2,531 | 839 | 1,055 | 0.751 | 0.706 | 0.892 |
| rain | val | 4,396 | 2,698 | 951 | 1,698 | 0.739 | 0.614 | 0.837 |
| rain | test | 4,113 | 2,506 | 1,073 | 1,607 | 0.700 | 0.609 | 0.821 |
| snow | val | 4,524 | 2,708 | 1,068 | 1,816 | 0.717 | 0.599 | 0.818 |
| snow | test | 4,476 | 2,808 | 1,197 | 1,668 | 0.701 | 0.627 | 0.825 |
| night | val | 3,745 | 2,123 | 1,058 | 1,622 | 0.667 | 0.567 | 0.774 |
| night | test | 4,055 | 2,446 | 1,156 | 1,609 | 0.679 | 0.603 | 0.841 |

## By class

| Class | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| person | val | 2,465 | 1,267 | 521 | 1,198 | 0.709 | 0.514 | 0.754 |
| person | test | 2,362 | 1,244 | 456 | 1,118 | 0.732 | 0.527 | 0.801 |
| car | val | 12,201 | 8,104 | 3,057 | 4,097 | 0.726 | 0.664 | 0.862 |
| car | test | 13,173 | 8,752 | 3,605 | 4,421 | 0.708 | 0.664 | 0.861 |
| bus | val | 346 | 147 | 115 | 199 | 0.561 | 0.425 | 0.598 |
| bus | test | 377 | 154 | 135 | 223 | 0.533 | 0.408 | 0.592 |
| bicycle | val | 203 | 52 | 31 | 151 | 0.627 | 0.256 | 0.527 |
| bicycle | test | 162 | 57 | 41 | 105 | 0.582 | 0.352 | 0.673 |
| motorcycle | val | 119 | 42 | 27 | 77 | 0.609 | 0.353 | 0.521 |
| motorcycle | test | 156 | 84 | 28 | 72 | 0.750 | 0.538 | 0.692 |

## By object size

| Size | Split | GT | TP | FN | Recall |
|---|---|---:|---:|---:|---:|
| tiny | val | 5,193 | 1,635 | 3,558 | 0.315 |
| tiny | test | 5,328 | 1,809 | 3,519 | 0.340 |
| small | val | 6,656 | 4,823 | 1,833 | 0.725 |
| small | test | 7,000 | 4,938 | 2,062 | 0.705 |
| medium | val | 2,380 | 2,113 | 267 | 0.888 |
| medium | test | 2,685 | 2,399 | 286 | 0.893 |
| large | val | 1,105 | 1,041 | 64 | 0.942 |
| large | test | 1,217 | 1,145 | 72 | 0.941 |

## Artifacts

- Raw report: `runs/eval/gate_edge_cbam_seed42/error_analysis/error_analysis.json`
- CSVs: `error_by_condition.csv`, `error_by_source.csv`, `error_by_class.csv`, `error_by_size.csv`
- Qualitative overlays: `runs/eval/gate_edge_cbam_seed42/error_analysis/overlays/`

_Facts only; interpretation belongs in the findings docs._
