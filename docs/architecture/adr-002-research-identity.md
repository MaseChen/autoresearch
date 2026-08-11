# ADR-002: Explicit research identity and namespace

## Status

Accepted

## Context

A source SHA-256 proves byte equality, but cannot prove that two measurements
used the same operator, language, toolchain, cases, protocol or baseline.
Global `candidate_hash` and `backend + suite` queries become unsafe as soon as
a second target is added.

## Decision

Every experiment belongs to a `ResearchNamespace` derived from immutable
operator, language, evaluator, evaluation-protocol and promotion-policy
profiles.  Proposer identity is recorded as an experimental condition but is
excluded from the namespace so models can be compared fairly.

An evaluation receives a host-generated `experiment_uid` before execution.
Artifact identity, condition identity and experiment identity remain separate.
Baseline, feedback, deduplication and best-result queries require a namespace.

## Trade-offs

- Different languages have independent automatic baseline tracks.
- Cross-language results are advisory rankings, not automatic promotion.
- Legacy evidence with incomplete provenance is retained as
  `legacy_unknown`; migration never invents missing facts.
