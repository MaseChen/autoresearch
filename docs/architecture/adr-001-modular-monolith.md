# ADR-001: Modular monolith with isolated runners

## Status

Accepted

## Context

The current deployment is one trusted host and one MetaX C500.  Proposal and
GPU execution require different capabilities, but independent service scaling
and remote scheduling are not current requirements.

## Decision

Keep orchestration, identity, persistence and policy in a modular Python
monolith.  Execute proposers, evaluators and privileged profilers in separate,
pinned, least-privilege containers.  Model resources with stable resource IDs
so a future scheduler can add GPUs or hosts without changing scientific
identity.

## Trade-offs

- We do not add a message broker, RPC control plane or distributed consensus.
- A single host remains an availability boundary.
- Module contracts and durable work records are required so runners can be
  extracted later if independent scaling becomes necessary.
