---
title: ADR-0004 — Adopt adverse16k (ACDC + DAWN + BDD100K + RTTS)
status: accepted
date: 2026-11-07
sources:
  - ref/acdc
  - ref/dawn
  - ref/bdd100k
  - ref/rtts
---

# ADR-0004 — Adopt adverse16k (ACDC + DAWN + BDD100K + RTTS)

## Context

The base paper's dataset table does not match the official DAWN, Foggy Cityscapes, or NightOwls datasets. A single small dataset (DAWN, 1,027 images) is insufficient for a robust detector, and BDD100K has only 143 foggy images. We need a balanced, defensible benchmark that covers the four thesis conditions and stays trainable on an RTX 3050 6 GB.

## Decision

Build **adverse16k**: 16,000 labeled images, exactly **4,000 per condition** (fog, rain, snow, night), split **80/10/10** into 12,800 train / 1,600 val / 1,600 test (3,200/400/400 per condition).

- **Sources:** ACDC, DAWN, BDD100K, RTTS.
- **Classes (5):** `0 person, 1 car, 2 bus, 3 bicycle, 4 motorcycle`.
- **Merges:** `truck -> car`, `rider -> person`; `train`, `traffic light`, `traffic sign` annotations are dropped.
- **Condition assignment:** mutually exclusive (adverse weather first; night only for night + clear/overcast/partly-cloudy/undefined weather).
- **Splits:** source-native train/val pools are respected; DAWN is split 80/10/10; shortfalls are filled by a designated filler source (RTTS for fog, BDD100K for rain/snow/night). Seed 42.
- **Labels:** YOLO for training plus COCO JSON for evaluation. BDD100K boxes come from Detection-2020 labels; frame weather/time attributes come from the 2018 release.
- **Storage:** processed images are **independent copies** in `data/processed/adverse16k` (link count 1), with `checksums.sha1`. The raw `ref/` datasets are retained for now.
- **Unlabeled test:** the 2,000 ACDC adverse test images (500 per condition) are copied to `data/unlabeled/acdc_test` for demo and final qualitative testing. ACDC test GT is withheld, so this set has **no annotations**.

## Consequences

- The benchmark is balanced by condition but **not by class**: car dominates (~88% of instances); person/bicycle/motorcycle are scarce in rain/snow/night. Class-aware sampling and Copy-Paste/Mosaic are needed during training.
- ACDC loses 36 images whose only labels are dropped classes; DAWN loses duplicates and a few images without target boxes. Fillers cover the deficits.
- The official ETH BDD100K host was unreachable; Detection-2020 labels were taken from the archive.org `bdd100k` mirror and joined to the local 2018 attributes.
- Raw datasets stay uncommitted under `ref/` and are retained for now; the independent processed copy and the ACDC test copy survive their eventual removal.
- `DAWN sand` is excluded because it is a fifth condition and cannot be balanced to 4k.
- Licenses are research/non-commercial: ACDC, DAWN (CC BY-NC 3.0), BDD100K, RTTS/RESIDE.

## Status

Accepted.

## Supersedes

None.
