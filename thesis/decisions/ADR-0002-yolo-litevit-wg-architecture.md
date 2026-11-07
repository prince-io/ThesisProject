---
title: ADR-0002 — Adopt YOLO-LiteViT-WG as Target Architecture
status: proposed
date: 2026-11-07
sources:
  - ref/idea.md
---

# ADR-0002 — Adopt YOLO-LiteViT-WG as Target Architecture

## Context

The thesis targets lightweight adverse-weather object detection trainable on an RTX 3050 6GB. `ref/idea.md` proposes YOLO-LiteViT-WG: YOLOv8n backbone, linear-attention ViT blocks at neck P4/P5, and a lightweight weather channel gate.

## Decision

Adopt YOLO-LiteViT-WG as the target architecture for implementation and evaluation.

## Consequences

- Backbone stays YOLOv8n C2f; global attention is restricted to P4/P5.
- Linear attention is required to bound memory.
- Weather gating is a cheap channel reweight on neck features.
- The proposal is a hypothesis until implementation and evaluation validate it.

## Status

Proposed — not yet validated.

## Supersedes

None.
