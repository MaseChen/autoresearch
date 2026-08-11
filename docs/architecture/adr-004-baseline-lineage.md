# ADR-004: Explicit baseline references and lineage

## Status

Accepted

## Context

Today a confirmed History row immediately becomes `get_best()`, while Git and
deployment configuration remain pinned until an operator adopts the source.
That ambiguity prevents safe cross-run campaigns.

## Decision

History stores immutable evidence and promotion decisions; it does not
implicitly choose the baseline for a run.  Every run freezes an explicit
`BaselineRef`.

Campaign discovery stores immutable `BaselineRevision` nodes and advances its
active pointer with compare-and-swap only after confirmation and checkpoint.
The deployment baseline is a separate administrator-owned pin.  Campaign and
OJ code cannot modify Git or the deployment pin.

## Trade-offs

- Promotion means eligible evidence, not automatic deployment.
- Normal runs require an explicit deployment baseline mapping per namespace.
- Rollback moves a pointer to an ancestor and never deletes evidence.
