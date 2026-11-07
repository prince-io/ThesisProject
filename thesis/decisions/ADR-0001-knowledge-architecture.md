---
title: ADR-0001 — Adopt DOX Knowledge Architecture
status: accepted
date: 2026-11-07
sources: []
---

# ADR-0001 — Adopt DOX Knowledge Architecture

## Context

The thesis needs a durable, curated source of truth that can be used to draft the final document, kept separate from raw inputs and raw run artifacts.

## Decision

Adopt the DOX AGENTS.md hierarchy and a curated `thesis/` knowledge base with separated fields: methodology, decisions, experiments, results, findings, related work, bibliography. `ref/` is a non-authoritative inbox; `manuscript/` is derived output; `runs/` holds full raw artifacts.

## Consequences

- Every meaningful edit requires a DOX pass.
- Knowledge must be promoted from `ref/` into `thesis/` after validation.
- Curated docs carry YAML front matter.
- Decisions are append-only ADRs.

## Status

Accepted.

## Supersedes

None.
