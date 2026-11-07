---
title: ADR-0005 — Sequence-Aware Splits and Label-Alias Fixes
status: accepted
date: 2026-11-07
sources:
  - ref/acdc
  - ref/dawn
  - ref/bdd100k
  - ref/rtts
---

# ADR-0005 — Sequence-Aware Splits and Label-Alias Fixes

## Context

A dataset audit found:

- ACDC's official train/val split shares sequences within every condition.
- RTTS's official 3,457/865 split shares **all** 145 validation sequences with training.
- DAWN has near-duplicate images (45 pairs crossed the initial random split).
- BDD100K Detection-2020 labels use legacy category aliases (`person`, `bike`, `motor`), which the original map silently dropped.

These issues would inflate validation/test scores through frame and near-duplicate leakage.

## Decision

- **Sequence-aware splits:** pool ACDC train+val and all RTTS images, group by sequence, and assign whole sequences globally per source to train/val/test. Official BDD train/val splits are kept (one keyframe per video); DAWN is assigned by near-duplicate group.
- **Use all usable ACDC/DAWN:** every usable ACDC (1,970) and DAWN (684) image is included; fillers (RTTS for fog, BDD100K for rain/snow/night) absorb the remaining per-split quota.
- **DAWN near-duplicate grouping:** 64-bit dHash, Hamming distance ≤ 6, union-find connected components; each component is assigned to one split.
- **BDD aliases:** `pedestrian`/`person` -> person; `bicycle`/`bike` -> bicycle; `motorcycle`/`motorbike`/`motor` -> motorcycle; `rider` -> person; `truck` -> car. `train`, `traffic light`, `traffic sign`, `lane/*`, and `area/*` are dropped.
- **Verification gates:** no sequence may span splits; no cross-split DAWN near-duplicates; no duplicate image hashes.

## Consequences

- Sequence leakage in the final dataset is **zero**; DAWN cross-split near-duplicates are **zero**.
- Source mix per condition is preserved, but per-split source counts vary; some conditions have little ACDC val/test representation, filled by BDD/RTTS.
- Class balance improves substantially once BDD aliases are mapped: person rises to 22,538 instances (from 11,715), bicycle to 1,758, motorcycle to 1,510. Car still dominates (~82%).
- The ACDC/RTTS official splits are no longer used as-is for this combined benchmark; this is deliberate and documented.
- The split is reproducible with seed 42 and is audited by `scripts/audit_adverse16k.py`.

## Status

Accepted.

## Supersedes

Split policy of ADR-0004.
