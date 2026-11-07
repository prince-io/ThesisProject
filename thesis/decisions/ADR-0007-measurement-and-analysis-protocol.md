---
title: ADR-0007 — Measurement and Error-Analysis Protocol
status: accepted
date: 2026-11-07
sources:
  - thesis/results/baseline.md
  - thesis/results/baseline-error-analysis.md
---

# ADR-0007 — Measurement and Error-Analysis Protocol

## Context

The first baseline run exposed two documentation issues: "peak VRAM" was reported without distinguishing PyTorch allocations from total GPU usage, and overall mAP alone does not explain where the detector fails. The thesis needs reproducible, clearly labeled memory and error measurements.

## Decision

- **Training VRAM:** report all of the following with explicit labels:
  - PyTorch peak allocated and reserved, tracked by training callbacks;
  - total GPU used peak from `nvidia-smi` sampling;
  - inference peak allocated, measured separately during evaluation.
  - When a completed run lacks tracker fields, use an identical 1-epoch calibration run and state that in the documentation.
- **Error analysis:** evaluate val and test at confidence 0.25 (primary) and 0.001 (recall ceiling), with IoU 0.5 matching:
  - same-class TP/FP/FN and precision/recall;
  - per-condition, per-source, per-class, and relative-size-bin breakdowns;
  - any-class confusion and background false positives;
  - qualitative false-negative/false-positive overlays.
- **Separation:** raw measurements go to `runs/`; curated facts to `thesis/results/`; interpretation to `thesis/findings/`.
- **Caveats:** single-seed results are provisional; the `cls_pw=0.0` ablation is required to attribute class-weighting effects.

## Consequences

- Memory claims are now auditable and distinguish training from inference.
- Overall mAP is complemented by per-condition, per-class, and per-size evidence.
- Error analysis is reproducible from the run artifacts and documented thresholds.
- All future runs must record VRAM metrics and support the same analysis.

## Status

Accepted.

## Supersedes

Measurement/reporting details of ADR-0006.
