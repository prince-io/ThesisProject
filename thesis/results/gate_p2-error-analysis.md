---
title: Gate + P2 Error Analysis
status: validated
date: 2026-11-11
sources:
  - runs/eval/gate_p2_seed42/error_analysis/error_analysis.json
  - thesis/results/gate_p2.md
---

# Gate + P2 Error Analysis

- Weights: `runs/train/gate_p2_seed42/weights/best.pt`
- Splits: val and test
- Confidence: 0.25
- Matching IoU: 0.5
- Same-class matching for TP/FP/FN.

## Overall

| Split | GT | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| val | 15,334 | 9,426 | 3,252 | 5,908 | 0.743 | 0.615 |
| test | 16,230 | 10,247 | 3,601 | 5,983 | 0.740 | 0.631 |

## By condition

| Condition | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fog | val | 2,669 | 2,051 | 660 | 618 | 0.757 | 0.768 | 0.940 |
| fog | test | 3,586 | 2,526 | 772 | 1,060 | 0.766 | 0.704 | 0.926 |
| rain | val | 4,396 | 2,686 | 836 | 1,710 | 0.763 | 0.611 | 0.911 |
| rain | test | 4,113 | 2,516 | 915 | 1,597 | 0.733 | 0.612 | 0.898 |
| snow | val | 4,524 | 2,672 | 938 | 1,852 | 0.740 | 0.591 | 0.900 |
| snow | test | 4,476 | 2,825 | 1,056 | 1,651 | 0.728 | 0.631 | 0.906 |
| night | val | 3,745 | 2,017 | 818 | 1,728 | 0.711 | 0.539 | 0.861 |
| night | test | 4,055 | 2,380 | 858 | 1,675 | 0.735 | 0.587 | 0.924 |

## By class

| Class | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| person | val | 2,465 | 1,259 | 469 | 1,206 | 0.729 | 0.511 | 0.834 |
| person | test | 2,362 | 1,237 | 401 | 1,125 | 0.755 | 0.524 | 0.873 |
| car | val | 12,201 | 7,940 | 2,632 | 4,261 | 0.751 | 0.651 | 0.930 |
| car | test | 13,173 | 8,714 | 3,011 | 4,459 | 0.743 | 0.662 | 0.932 |
| bus | val | 346 | 132 | 86 | 214 | 0.606 | 0.382 | 0.598 |
| bus | test | 377 | 144 | 123 | 233 | 0.539 | 0.382 | 0.637 |
| bicycle | val | 203 | 54 | 32 | 149 | 0.628 | 0.266 | 0.611 |
| bicycle | test | 162 | 65 | 31 | 97 | 0.677 | 0.401 | 0.716 |
| motorcycle | val | 119 | 41 | 33 | 78 | 0.554 | 0.345 | 0.597 |
| motorcycle | test | 156 | 87 | 35 | 69 | 0.713 | 0.558 | 0.750 |

## By object size

| Size | Split | GT | TP | FN | Recall |
|---|---|---:|---:|---:|---:|
| tiny | val | 5,193 | 1,595 | 3,598 | 0.307 |
| tiny | test | 5,328 | 1,817 | 3,511 | 0.341 |
| small | val | 6,656 | 4,696 | 1,960 | 0.706 |
| small | test | 7,000 | 4,900 | 2,100 | 0.700 |
| medium | val | 2,380 | 2,102 | 278 | 0.883 |
| medium | test | 2,685 | 2,398 | 287 | 0.893 |
| large | val | 1,105 | 1,033 | 72 | 0.935 |
| large | test | 1,217 | 1,132 | 85 | 0.930 |

## By source

| Source | Split | GT | TP | FP | FN | Precision | Recall | Recall @0.001 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ACDC | val | 1,675 | 719 | 288 | 956 | 0.714 | 0.429 | 0.759 |
| ACDC | test | 1,418 | 803 | 308 | 615 | 0.723 | 0.566 | 0.862 |
| BDD | val | 10,747 | 6,454 | 2,254 | 4,293 | 0.741 | 0.601 | 0.911 |
| BDD | test | 11,113 | 6,767 | 2,443 | 4,346 | 0.735 | 0.609 | 0.913 |
| DAWN | val | 410 | 339 | 90 | 71 | 0.790 | 0.827 | 0.968 |
| DAWN | test | 579 | 486 | 167 | 93 | 0.744 | 0.839 | 0.978 |
| RTTS | val | 2,502 | 1,914 | 620 | 588 | 0.755 | 0.765 | 0.938 |
| RTTS | test | 3,120 | 2,191 | 683 | 929 | 0.762 | 0.702 | 0.924 |

## Background false positives

Unmatched predictions by class across the evaluated splits:

| Predicted class | Count |
|---|---:|
| car | 5,462 |
| person | 856 |
| bus | 156 |
| motorcycle | 58 |
| bicycle | 54 |

## Matched confusion

Counts are aggregated over the evaluated splits. Rows are GT classes; columns are predicted classes.

| GT | person | car | bus | bicycle | motorcycle |
|---|---:|---:|---:|---:|---:|
| person | 2491 | 20 | 0 | 0 | 2 |
| car | 5 | 16619 | 82 | 1 | 5 |
| bus | 0 | 184 | 247 | 0 | 0 |
| bicycle | 4 | 3 | 0 | 117 | 10 |
| motorcycle | 10 | 9 | 0 | 10 | 121 |

## Artifacts

- Raw report: `runs/eval/gate_p2_seed42/error_analysis/error_analysis.json`
- CSVs: `error_by_condition.csv`, `error_by_source.csv`, `error_by_class.csv`, `error_by_size.csv`
- Qualitative overlays: `runs/eval/gate_p2_seed42/error_analysis/overlays/`

_Facts only; interpretation belongs in `thesis/findings/gate-p2-analysis.md`._
