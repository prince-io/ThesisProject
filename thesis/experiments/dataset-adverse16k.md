---
title: Dataset Record — adverse16k
status: validated
date: 2026-11-07
sources:
  - ref/acdc
  - ref/dawn
  - ref/bdd100k
  - ref/rtts
---

# Dataset Record — adverse16k

## Summary

Balanced adverse-weather detection benchmark: **16,000 images**, **4,000 per condition** (fog, rain, snow, night), **5 classes**, split **12,800 / 1,600 / 1,600** (80/10/10). Built deterministically with seed 42 using **sequence-aware splits** and **near-duplicate grouping**; the audit found **zero sequence leakage** and **zero cross-split near-duplicates**. The processed set is an **independent copy** (link count 1) with checksums. A separate unlabeled **ACDC test** set (2,000 images) is kept for demos and final qualitative testing. Full datasheet: `dataset-card-adverse16k.md`.

## Sources

| Source | Images used | Access | Notes |
|---|---:|---|---|
| RTTS (RESIDE) | 3,228 | Google Drive / RESIDE | Real fog, 5 classes |
| BDD100K | 10,118 | local copy + 2020 labels (archive.org mirror) | Boxes from Detection-2020; weather/time from 2018 release |
| ACDC | 1,970 | local copy | Adverse train/val COCO detection |
| DAWN | 684 | Mendeley Data | Pascal VOC; exact duplicates removed |

## Condition and split counts

| Condition | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Fog | 3,200 | 400 | 400 | 4,000 |
| Rain | 3,200 | 400 | 400 | 4,000 |
| Snow | 3,200 | 400 | 400 | 4,000 |
| Night | 3,200 | 400 | 400 | 4,000 |
| **Total** | **12,800** | **1,600** | **1,600** | **16,000** |

Source mix by condition:

| Condition | ACDC | DAWN | BDD100K | RTTS |
|---|---:|---:|---:|---:|
| Fog | 490 | 282 | — | 3,228 |
| Rain | 493 | 199 | 3,308 | — |
| Snow | 497 | 203 | 3,300 | — |
| Night | 490 | — | 3,510 | — |

## Classes

| ID | Name | Train instances | Val instances | Test instances | Total |
|---:|---|---:|---:|---:|---:|
| 0 | person | 17,711 | 2,465 | 2,362 | 22,538 |
| 1 | car (incl. truck) | 106,284 | 12,201 | 13,173 | 131,658 |
| 2 | bus | 2,972 | 346 | 377 | 3,695 |
| 3 | bicycle | 1,393 | 203 | 162 | 1,758 |
| 4 | motorcycle | 1,235 | 119 | 156 | 1,510 |
| | **Total** | **129,595** | **15,334** | **16,230** | **161,159** |

Mapping: `pedestrian/person/rider -> person`, `truck -> car`, `bicycle/bike -> bicycle`, `motorcycle/motorbike/motor -> motorcycle`; `train`, `traffic light`, `traffic sign`, `lane/*`, `area/*` dropped. Images with no target-class boxes are excluded.

Source counts by split (train/val/test): ACDC 1,565/199/206; DAWN 546/69/69; RTTS 2,530/369/329; BDD100K 8,159/963/996.

## Split policy

- ACDC: official train+val pooled; whole sequences are assigned globally per source to train/val/test.
- RTTS: all images pooled; sequence-aware split (the official frame-level split leaks sequences).
- DAWN: near-duplicate groups (64-bit dHash, Hamming ≤ 6) are assigned as a unit to one split.
- BDD100K: official train/val kept; each image is a distinct video keyframe, so no sequence leakage.
- Shortfalls are filled by the condition's filler source: RTTS for fog, BDD100K for rain/snow/night.
- Seed 42; verification requires zero sequences spanning splits and zero cross-split near-duplicates.

## Condition assignment

- ACDC: condition folder.
- DAWN: weather folder (fog/rain/snow; sand excluded).
- RTTS: fog.
- BDD100K: `foggy -> fog`, `snowy -> snow`, `rainy -> rain`; `night` only if `timeofday=night` and weather is clear / partly cloudy / overcast / undefined. This makes conditions mutually exclusive.

## Unlabeled ACDC test set

- **2,000 images** (500 per condition), anonymized ACDC test split.
- Location: `data/unlabeled/acdc_test/` (`images/`, `manifest.csv`, `labels_info/`, `stats.json`).
- **No ground-truth boxes** (ACDC withholds test annotations); use for demos and qualitative testing.
- Independent copies (link count 1); report: `runs/dataset_checks/acdc_test/report.json` (`passed: true`).

## Verification

- `data/processed/adverse16k` is an independent copy: every image has link count 1; `checksums.sha1` covers all 16,000 images.
- Report: `runs/dataset_checks/adverse16k/report.json` (`passed: true`).
- Checks: 16,000 total; 4,000/condition; 3,200/400/400 per condition; 5 classes only; valid normalized boxes; images decode; no duplicate hashes; **zero sequence leakage**; **zero cross-split DAWN near-duplicates**; BDD weather/time coverage complete.
- Audit: `runs/dataset_checks/adverse16k/audit.json` and `audit.md`, with plots in `runs/dataset_checks/adverse16k/plots/`.
- Raw `ref/` datasets are retained for now; the processed and test copies do not depend on them.

## Known limitations

- Class imbalance remains: car ~81.7% of instances; person 14.0%; bus 2.3%; bicycle 1.1%; motorcycle 0.9%.
- ACDC/RTTS official splits are re-pooled for sequence safety; results are not directly comparable to their official split numbers.
- BDD100K Detection-2020 labels were obtained from the archive.org mirror because the official ETH host was unreachable. The labels use legacy aliases (`person`, `bike`, `motor`), now mapped correctly.
- DAWN near-duplicate grouping uses a fixed dHash threshold (6) and may merge visually similar but distinct scenes.
- No explicit ignore regions; dropped-class objects become background.
- `DAWN sand` is excluded (fifth condition, cannot be balanced to 4k).
- ACDC test has no GT and cannot give quantitative mAP locally.

## Reproduction

```bash
.venv/bin/python scripts/build_adverse16k.py --root . --out data/processed/adverse16k \
  --force --copy-images
.venv/bin/python scripts/verify_adverse16k.py --root . --out data/processed/adverse16k \
  --report runs/dataset_checks/adverse16k --require-independent --checksums
.venv/bin/python scripts/prepare_acdc_test.py --root . --out data/unlabeled/acdc_test \
  --report runs/dataset_checks/acdc_test --force
.venv/bin/python scripts/audit_adverse16k.py --root . --out data/processed/adverse16k \
  --report runs/dataset_checks/adverse16k
```

Config: `configs/datasets/adverse16k.yaml`.

## Output layout

```text
data/processed/adverse16k/
├── images/{train,val,test}/     # independent copies (link count 1)
├── labels/{train,val,test}/     # YOLO labels, class ids 0-4
├── coco/{train,val,test}.json   # COCO export for pycocotools
├── manifest.csv                 # per-image provenance, sequence, counts
├── stats.json                   # split/condition/source/class counts
├── checksums.sha1               # SHA-1 for all processed images
└── dataset.yaml                 # Ultralytics config

data/unlabeled/acdc_test/
├── images/{fog,night,rain,snow}/test/...  # 2,000 unlabeled copies
├── labels_info/                            # ACDC test image-info JSONs
├── manifest.csv
├── stats.json
└── README.md
```
