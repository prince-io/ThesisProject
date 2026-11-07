# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here
- Agent must follow DOX instructions across any edits

## Project

- B.Tech final-year thesis: lightweight object detection robust to adverse weather (fog, rain, snow, night).
- Target architecture (proposed): YOLO-LiteViT-WG — YOLOv8n backbone + linear-attention ViT at neck P4/P5 + lightweight weather channel gate.
- Hard constraint: train stably at 640x640 on an RTX 3050 6GB (batch 4, AMP).
- `thesis/` is the curated source of truth for the final thesis document. `ref/` is a non-authoritative inbox of ideas, suggestions, and papers. `manuscript/` is the derived LaTeX write-up.
- Dataset pipeline is implemented: `adverse16k` (16k images, 4 conditions, 5 classes) from ACDC + DAWN + BDD100K + RTTS. Model/training implementation has not started; methodology details are deferred until modeling begins.

## Core Contract

- AGENTS.md files are binding work contracts for their subtrees
- Work products, source materials, instructions, records, assets, and durable docs must stay understandable from the nearest applicable AGENTS.md plus every parent AGENTS.md above it

## Read Before Editing

1. Read the root AGENTS.md
2. Identify every file or folder you expect to touch
3. Walk from the repository root to each target path
4. Read every AGENTS.md found along each route
5. If a parent AGENTS.md lists a child AGENTS.md whose scope contains the path, read that child and continue from there
6. Use the nearest AGENTS.md as the local contract and parent docs for repo-wide rules
7. If docs conflict, the closer doc controls local work details, but no child doc may weaken DOX

Do not rely on memory. Re-read the applicable DOX chain in the current session before editing.

## Update After Editing

Every meaningful change requires a DOX pass before the task is done.

Update the closest owning AGENTS.md when a change affects:

- purpose, scope, ownership, or responsibilities
- durable structure, contracts, workflows, or operating rules
- required inputs, outputs, permissions, constraints, side effects, or artifacts
- user preferences about behavior, communication, process, organization, or quality
- AGENTS.md creation, deletion, move, rename, or index contents

Update parent docs when parent-level structure, ownership, workflow, or child index changes. Update child docs when parent changes alter local rules. Remove stale or contradictory text immediately. Small edits that do not change behavior or contracts may leave docs unchanged, but the DOX pass still must happen.

## Hierarchy

- Root AGENTS.md is the DOX rail: project-wide instructions, global preferences, durable workflow rules, and the top-level Child DOX Index
- Child AGENTS.md files own domain-specific instructions and their own Child DOX Index
- Each parent explains what its direct children cover and what stays owned by the parent
- The closer a doc is to the work, the more specific and practical it must be

## Child Doc Shape

- Create a child AGENTS.md when a folder becomes a durable boundary with its own purpose, rules, responsibilities, workflow, materials, or quality standards
- Work Guidance must reflect the current standards of the project or user instructions; if there are no specific standards or instructions yet, leave it empty
- Verification must reflect an existing check; if no verification framework exists yet, leave it empty and update it when one exists

Default section order:
- Purpose
- Ownership
- Local Contracts
- Work Guidance
- Verification
- Child DOX Index

## Style

- Keep docs concise, current, and operational
- Document stable contracts, not diary entries
- Put broad rules in parent docs and concrete details in child docs
- Prefer direct bullets with explicit names
- Do not duplicate rules across many files unless each scope needs a local version
- Delete stale notes instead of explaining history
- Trim obvious statements, repeated rules, misplaced detail, and warnings for risks that no longer exist

## Closeout

1. Re-check changed paths against the DOX chain
2. Update nearest owning docs and any affected parents or children
3. Refresh every affected Child DOX Index
4. Remove stale or contradictory text
5. Run existing verification when relevant
6. Report any docs intentionally left unchanged and why

## User Preferences

- `thesis/` is the single source of authority; `ref/` is input-only and never cited as a record of truth.
- Keep full raw artifacts and curated thesis knowledge separate: full outputs live in `runs/`, curated summaries in `thesis/experiments` and `thesis/results`.
- Record every important decision as an append-only, dated ADR in `thesis/decisions/` (never delete; mark superseded).
- Curated docs carry YAML front matter: `title`, `status` (`draft` | `validated` | `superseded`), `date`, `sources`.
- Do not commit reference PDFs or raw datasets; `ref/*.pdf`, archives, and raw dataset folders stay untracked.
- Raw datasets under `ref/` are inputs and are retained for now. Processed benchmark and ACDC test copies under `data/` are independent (no hardlinks).
- Do not add methodology content until model implementation starts.
- Draft the final thesis from `thesis/`; output to `manuscript/`.
- Ask the user for approval before every git commit; never commit unprompted.
- Use plain, simple commit messages that are easy to understand.

## Local Environment

- Python 3.12 virtualenv at `.venv/` (gitignored).
- PyTorch 2.14.1 + torchvision 0.29.1, CUDA 13.0 wheels; RTX 3050 Laptop 6 GB (compute 8.6).
- Dependencies declared in `pyproject.toml`; install with `.venv/bin/pip install -e ".[dev,datasets]"`.
- Run tooling via `.venv/bin/` (`ruff`, `pytest`).
- Dataset build: `.venv/bin/python scripts/build_adverse16k.py --root . --out data/processed/adverse16k --force --copy-images`.
- Dataset verify: `.venv/bin/python scripts/verify_adverse16k.py --root . --out data/processed/adverse16k --report runs/dataset_checks/adverse16k --require-independent --checksums`.
- Dataset audit: `.venv/bin/python scripts/audit_adverse16k.py --root . --out data/processed/adverse16k --report runs/dataset_checks/adverse16k`.
- ACDC test prep: `.venv/bin/python scripts/prepare_acdc_test.py --root . --out data/unlabeled/acdc_test --report runs/dataset_checks/acdc_test --force`.
- `data/` holds datasets and `runs/` holds full run artifacts; both are gitignored.

## Child DOX Index

- `ref/AGENTS.md` — non-authoritative inbox for ideas, suggestions, and papers; promotion rules.
- `thesis/AGENTS.md` — curated source of truth: field separation, ADRs, metadata, citation rules, and its own child index.
- `manuscript/AGENTS.md` — derived LaTeX thesis output; drafting rules.
- `src/AGENTS.md` — implementation package (`litevit`), src layout, code contracts.
- `configs/AGENTS.md` — training and experiment YAML configs.
- `scripts/AGENTS.md` — entrypoints and utilities.
- `tests/AGENTS.md` — verification tests.
