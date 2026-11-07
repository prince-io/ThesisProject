---
title: Dataset Card — adverse16k
status: validated
date: 2026-11-07
sources:
  - ref/acdc
  - ref/dawn
  - ref/bdd100k
  - ref/rtts
---

# Dataset Card — adverse16k

## Motivation

Adverse-weather object detection needs a benchmark that is balanced across weather conditions, large enough for a lightweight detector, and free of frame-level leakage. The base paper's dataset table did not match the official benchmarks, so this dataset combines four public sources and documents its construction.

## Composition

- **16,000 images**, **4,000 per condition**: fog, rain, snow, night.
- **5 classes**: `0 person, 1 car (incl. truck), 2 bus, 3 bicycle, 4 motorcycle`.
- **Split**: 12,800 train / 1,600 val / 1,600 test (3,200/400/400 per condition).

| Source | Images | Share | Role |
|---|---:|---:|---|
| BDD100K | 10,118 | 63.2% | Large-scale urban driving; weather/time attributes |
| RTTS (RESIDE) | 3,228 | 20.2% | Real fog; 5 classes |
| ACDC | 1,970 | 12.3% | Adverse train/val COCO detection |
| DAWN | 684 | 4.3% | Real fog/rain/snow; Pascal VOC |

## Class instances

| Class | Train | Val | Test | Total | Share |
|---|---:|---:|---:|---:|---:|
| person | 17,711 | 2,465 | 2,362 | 22,538 | 14.0% |
| car (incl. truck) | 106,284 | 12,201 | 13,173 | 131,658 | 81.7% |
| bus | 2,972 | 346 | 377 | 3,695 | 2.3% |
| bicycle | 1,393 | 203 | 162 | 1,758 | 1.1% |
| motorcycle | 1,235 | 119 | 156 | 1,510 | 0.9% |
| **Total** | **129,595** | **15,334** | **16,230** | **161,159** | 100% |

Per-condition instances:

| Condition | person | car | bus | bicycle | motorcycle |
|---|---:|---:|---:|---:|---:|
| Fog | 8,962 | 22,913 | 2,009 | 594 | 974 |
| Rain | 4,257 | 36,132 | 656 | 369 | 176 |
| Snow | 5,897 | 37,605 | 697 | 416 | 113 |
| Night | 3,422 | 35,008 | 333 | 379 | 247 |

## Label mapping

| Target | Source labels merged in |
|---|---|
| person | `person`, `pedestrian`, `rider` |
| car | `car`, `truck` |
| bus | `bus` |
| bicycle | `bicycle`, `bike` |
| motorcycle | `motorcycle`, `motorbike`, `motor` |

Dropped annotations: `train`, `traffic light`, `traffic sign`, `lane/*`, `area/*`. Counts are in the audit report.

## Split methodology

- **ACDC:** official train+val pooled and split by **sequence**; whole sequences are assigned globally to one split.
- **RTTS:** all images pooled and split by **sequence**; the official frame-level split was not used because it leaks sequences.
- **DAWN:** near-duplicate images are grouped by 64-bit dHash (Hamming ≤ 6, union-find) and each group is assigned to one split.
- **BDD100K:** official train/val splits are kept; each image is a distinct video keyframe, so no sequence leakage.
- **Fillers:** shortfalls are filled by RTTS (fog) or BDD100K (rain/snow/night). Seed 42.
- **Verification:** zero sequences span splits; zero cross-split DAWN near-duplicates; zero duplicate image hashes.

## Quality checks

- `scripts/verify_adverse16k.py`: counts, class ids, box validity, image decode, duplicate hashes, sequence leakage, independent files, checksums.
- `scripts/audit_adverse16k.py`: distributions, dropped/merged annotations, DAWN near-duplicates, plots.
- Reports: `runs/dataset_checks/adverse16k/report.json` and `audit.json` (both `passed`/clean).
- Checksums: `data/processed/adverse16k/checksums.sha1` (16,000 images).

## Distribution and biases

- **Class imbalance:** car is ~82% of instances; strategies such as Copy-Paste, Mosaic, and class-aware sampling are recommended during training.
- **Source imbalance:** BDD100K supplies 63% of images; report per-source results and avoid tuning only on BDD.
- **Resolution:** 1280x720 (10,201), 1920x1080 (1,970), 896x504 (281), long tail.
- **Objects/image:** median 9, mean 10.1, p90 19, max 144.
- **Box area (relative):** median 0.0024, p90 0.0347; small objects are common.

## Intended use

- Train and evaluate lightweight adverse-weather detectors at 640x640.
- Per-condition and per-source robustness analysis.
- Cross-domain experiments (e.g., train on ACDC/RTTS, test on BDD adverse subsets).

## Out of scope

- Sandstorm (DAWN sand excluded), conditions outside fog/rain/snow/night.
- Quantitative ACDC test evaluation (test GT is withheld); the 2,000-image unlabeled ACDC test copy is for demos/qualitative testing.
- Commercial use.

## Licenses and ethics

- ACDC: research/non-commercial; anonymized images.
- DAWN: CC BY-NC 3.0.
- BDD100K: academic/non-commercial; privacy-sensitive driving scenes.
- RTTS/RESIDE: research use.
- Raw data is untracked; processed copies do not redistribute raw image bundles beyond local research use.

## Versioning and maintenance

- Version: `v1.0`, 2026-11-07.
- Rebuild: `scripts/build_adverse16k.py --force --copy-images` (seed 42).
- Verify: `scripts/verify_adverse16k.py --require-independent --checksums`.
- Audit: `scripts/audit_adverse16k.py`.
- Config: `configs/datasets/adverse16k.yaml`.

## Limitations

- Class and source imbalance remain.
- ACDC/RTTS official splits are re-pooled; results are not directly comparable to ACDC-val or RTTS random-split numbers.
- DAWN near-duplicate grouping uses a fixed dHash threshold (6) and may merge visually similar but distinct scenes.
- Dropped-class objects become background; no explicit ignore regions.

## Citations

- Yu et al., BDD100K, CVPR 2020.
- Sakaridis et al., ACDC, ICCV 2021.
- Kenk & Hassaballah, DAWN, 2020.
- Li et al., RESIDE/RTTS, IEEE TIP 2018.
