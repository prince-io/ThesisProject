---
title: Project Overview
status: draft
date: 2026-11-07
sources:
  - ref/idea.md
---

# Project Overview

## Objective

B.Tech final-year thesis on lightweight object detection robust to adverse weather (rain, fog, snow, low light).

## Proposed architecture

YOLO-LiteViT-WG: YOLOv8n backbone + linear-attention ViT blocks at neck P4/P5 + lightweight weather channel gate. Proposed in `ref/idea.md`; adoption recorded in `decisions/ADR-0002-yolo-litevit-wg-architecture.md`.

## Hard constraints

- Train stably at 640x640 on an RTX 3050 6GB.
- Batch size 4, mixed precision (AMP).
- Target ~3.5–4.0 M parameters.

## Non-goals

- Methodology content is not written until implementation starts.

## Open questions

- Dataset selection (e.g., DAWN, RTTS, Foggy Cityscapes, custom) — undecided.
- Implementation stack and tooling — deferred until implementation.
- Whether to seed methodology from `ref/idea.md` — deferred.
