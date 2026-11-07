# scripts/ — Entrypoints and Utilities

## Purpose

Hold thin CLI entrypoints for training, evaluation, export, and data preparation.

## Ownership

- Authoritative for command-line usage.

## Local Contracts

- Scripts are thin wrappers over `litevit`; business logic lives in `src/litevit/`.
- No hardcoded paths; accept config/CLI arguments.
- Each script documents usage in its docstring.

## Work Guidance

- (none yet)

## Verification

- `.venv/bin/ruff check .`

## Child DOX Index

- (none)
