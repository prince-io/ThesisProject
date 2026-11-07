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

Balanced adverse-weather detection benchmark: **16,000 images**, **4,000 per condition** (fog, rain, snow, night), **5 classes**, split **12,800 / 1,600 / 1,600** (80/10/10). Built deterministically with seed 42. The processed set is an **independent copy** (link count 1) with checksums. A separate unlabeled **ACDC test** set (2,000 images) is kept for demos and final qualitative testing.

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

| ID | Name | Train instances | Val instances | Test instances |
|---:|---|---:|---:|---:|
| 0 | person | 9,243 | 1,338 | 1,138 |
| 1 | car (incl. truck) | 104,618 | 13,908 | 13,221 |
| 2 | bus | 2,840 | 412 | 432 |
| 3 | bicycle | 811 | 94 | 107 |
| 4 | motorcycle | 1,008 | 145 | 118 |
| | **Total** | **118,520** | **15,897** | **15,016** |

Mapping: `truck -> car`, `rider -> person`; `train`, `traffic light`, `traffic sign` dropped. Images with no target-class boxes are excluded.

## Split policy

- ACDC: official train -> train; official val -> val/test.
- BDD100K: official train -> train; official val -> val/test.
- RTTS: official 3,457 train / 865 held-out; held-out split into val/test.
- DAWN: 80/10/10 by condition (seed 42), after exact-duplicate removal.
- Shortfalls are filled by the condition's filler source: RTTS for fog, BDD100K for rain/snow/night.
- No image is trained on if it comes from a source's official val/test split.

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
- Checks: 16,000 total; 4,000/condition; 3,200/400/400 per condition; 5 classes only; valid normalized boxes; images decode; no duplicate hashes; BDD weather/time coverage complete.
- Raw `ref/` datasets are retained for now; the processed and test copies do not depend on them.

## Known limitations

- Strong class imbalance: car is ~88% of instances; person/bicycle/motorcycle are scarce in rain/snow/night.
- ACDC and DAWN contribute fewer images than planned because of dropped classes and duplicate/no-box removal; fillers compensate.
- BDD100K Detection-2020 labels were obtained from the archive.org mirror because the official ETH host was unreachable.
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
```

Config: `configs/datasets/adverse16k.yaml`.

## Output layout

```text
data/processed/adverse16k/
├── images/{train,val,test}/     # independent copies (link count 1)
├── labels/{train,val,test}/     # YOLO labels, class ids 0-4
├── coco/{train,val,test}.json   # COCO export for pycocotools
├── manifest.csv                 # per-image provenance and counts
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
