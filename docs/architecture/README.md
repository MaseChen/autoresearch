# Autoresearch Kernel Platform V2 architecture

This directory records the architectural constraints for the V2 platform.
The decisions are intentionally small and independently reviewable.  A code
change that contradicts an accepted decision must first supersede that ADR.

- [ADR-001: modular monolith with isolated runners](adr-001-modular-monolith.md)
- [ADR-002: explicit research identity and namespace](adr-002-research-identity.md)
- [ADR-003: candidate bundles and content addressing](adr-003-candidate-bundles.md)
- [ADR-004: explicit baseline references and lineage](adr-004-baseline-lineage.md)
- [ADR-005: benchmark and discovery modes](adr-005-run-modes.md)
- [ADR-006: trusted built-in extensions](adr-006-trusted-extensions.md)
- [ADR-007: versioned workflows and bounded-context stores](adr-007-workflows-and-stores.md)
- [ADR-008: trusted CURRENT baseline bootstrap](adr-008-current-baseline-bootstrap.md)
- [V2 implementation and activation status](v2-implementation-status.md)
- [Coordinated V2-to-V3 migration runbook](migration-runbook.md)
- [Disaster recovery runbook](recovery-runbook.md)

The target remains a standard-library-only trusted host.  PyTorch, Triton,
TileLang and vendor runtimes stay inside pinned evaluator environments.
