# ADR-007: Versioned workflows and bounded-context stores

## Status

Accepted

## Context

The V1 stage enum is embedded in Controller SQL triggers.  Adding shadow,
noise, profiling or campaigns directly to that enum would repeatedly migrate
unrelated state.  History and Controller commits are intentionally separate.

## Decision

Run stages come from a trusted, versioned `EvaluationProtocol`; each run stores
the resolved workflow snapshot.  Shadow and holdout cases are case roles inside
the quick gate.  Noise and profile observations are explicit experiment/job
relations, not promotion stages.

Keep separate SQLite stores for scientific History, bounded Run control and
Campaign control.  Allocate an `experiment_uid` in Controller before external
execution, insert History idempotently by that UID, then link Controller state.
Campaign observes only terminal child-run evidence.  No cross-database pseudo
transaction is introduced.

## Trade-offs

- Recovery requires deterministic reconciliation and idempotency keys.
- Database, record, API, proposal and manifest version numbers are named and
  evolved independently.
