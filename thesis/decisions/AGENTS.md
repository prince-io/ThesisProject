# decisions/ — Architecture Decision Records

## Purpose

Record durable thesis decisions as append-only ADRs.

## Ownership

- Authoritative for what was decided and why.

## Local Contracts

- One file per decision: `ADR-NNNN-slug.md` (zero-padded, sequential).
- Each ADR has YAML front matter: `title`, `status` (`proposed` | `accepted` | `superseded` | `rejected`), `date`, `sources`.
- Sections: Context, Decision, Consequences, Status, Supersedes.
- Never delete or rewrite the decision; mark superseded and link the replacement.

## Work Guidance

- Record decisions that affect architecture, data, training, evaluation, or tooling.

## Verification

- Manual: IDs are unique and sequential; superseded ADRs link their replacement.

## Child DOX Index

- `ADR-0001-knowledge-architecture.md`
- `ADR-0002-yolo-litevit-wg-architecture.md`
- `ADR-0003-python-pytorch-stack.md`
- `ADR-0004-adverse16k-dataset.md`
- `ADR-0005-sequence-aware-splits.md`
- `ADR-0006-baseline-protocol.md`
- `ADR-0007-measurement-and-analysis-protocol.md`
