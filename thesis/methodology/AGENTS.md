# methodology/ — Approach and Design

## Purpose

Hold the validated approach: architecture, training strategy, and design rationale used in the thesis.

## Ownership

- Authoritative for how the method works.

## Local Contracts

- Content may be added as implementation proceeds.
- Each doc carries YAML front matter and cites its sources.
- Describe the method, not measured outcomes; results belong in `../results/` and `../findings/`.

## Work Guidance

- `training.md` records the baseline training/evaluation protocol.
- `architecture.md` records the V1 P5 LiteViT design and integration.
- `weather-gate.md` records the gate-only design, values, and parameter budget.
- `gate-p5.md` records A+B: gate plus regularized P5 attention.
- `gate-edge-cbam.md` records the gate + Sobel edge + spatial CBAM model.
- `p2-head.md` records the gate-only + stride-4 P2 detection head design.
- Add V2 (P4/P5 attention) methodology as implementation proceeds.

## Verification

- No automated checks.

## Child DOX Index

- (none)
