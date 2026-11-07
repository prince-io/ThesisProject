# ref/ — Non-Authoritative Input Inbox

## Purpose

Capture raw inputs: ideas, suggestions, literature notes, and papers. This folder is an inbox, not a record of truth.

## Ownership

- The user drops material here.
- The agent reads and synthesizes from it, but never cites `ref/` as authoritative.

## Local Contracts

- Everything here is mutable and may be incomplete or wrong.
- Validated conclusions must be promoted to `thesis/` with `sources` pointing back to the relevant `ref/` item.
- Reference PDFs and raw datasets are not committed (`ref/*.pdf`, `ref/*.zip`, `ref/*.tar`, `ref/acdc/`, `ref/bdd100k/`, `ref/dawn/`, `ref/rtts/` are gitignored); `.md` notes are tracked.
- Raw datasets here are inputs only. Processed benchmark and ACDC test copies live under `data/`; curated records go to `thesis/experiments/`.
- Raw datasets are retained for now; the `data/` copies are independent and do not rely on them.
- The unlabeled ACDC test images under `ref/acdc` are copied to `data/unlabeled/acdc_test` for demos.

## Work Guidance

- Treat `idea.md` as a design proposal, not a decided methodology.
- When promoting, separate facts (results), interpretation (findings), and decisions (ADRs) as defined by `thesis/AGENTS.md`.
- The `adverse16k` build consumes these raw datasets; see `thesis/experiments/dataset-adverse16k.md`.

## Verification

- No automated checks. Manual: ensure promoted claims in `thesis/` cite a `ref/` source when derived from here.

## Child DOX Index

- (none)
