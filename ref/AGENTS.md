# ref/ — Non-Authoritative Input Inbox

## Purpose

Capture raw inputs: ideas, suggestions, literature notes, and papers. This folder is an inbox, not a record of truth.

## Ownership

- The user drops material here.
- The agent reads and synthesizes from it, but never cites `ref/` as authoritative.

## Local Contracts

- Everything here is mutable and may be incomplete or wrong.
- Validated conclusions must be promoted to `thesis/` with `sources` pointing back to the relevant `ref/` item.
- Reference PDFs are not committed (`ref/*.pdf` is gitignored); `.md` notes are tracked.
- Do not move content out of `ref/`; promote by writing curated docs in `thesis/`.

## Work Guidance

- Treat `idea.md` as a design proposal, not a decided methodology.
- When promoting, separate facts (results), interpretation (findings), and decisions (ADRs) as defined by `thesis/AGENTS.md`.

## Verification

- No automated checks. Manual: ensure promoted claims in `thesis/` cite a `ref/` source when derived from here.

## Child DOX Index

- (none)
