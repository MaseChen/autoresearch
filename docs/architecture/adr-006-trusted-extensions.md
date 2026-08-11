# ADR-006: Trusted built-in extensions

## Status

Accepted

## Context

Operator, language, device and proposer adapters execute inside the trusted
control or evaluation framework.  Arbitrary runtime imports would expand the
trusted computing base and defeat current allowlists.

## Decision

Profiles select implementations from a built-in, reviewed registry.  They may
configure only declared, bounded options and cannot name Python import paths or
add container capabilities.  Candidate source remains the sole untrusted code.

Privileged profiling, if required by vendor tooling, uses a separate trusted
runner rather than expanding ordinary evaluator permissions.  Native GPU
kernels require compile/execute separation and a dedicated isolation review.

## Trade-offs

- Installing a new plugin requires a framework release.
- Signed external plugin distribution and adversarial GPU isolation are future
  work, not implicit capabilities of V2.
