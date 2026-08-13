# ADR-008: Trusted CURRENT baseline bootstrap

## Status

Accepted

## Context

The first deployment adoption produced a scientifically comparable baseline in
the legacy research namespace.  The CURRENT namespace changes the evaluation
protocol by adding reviewed shadow and holdout routing, so legacy experiments
cannot be relabelled as CURRENT.  At the same time, normal V2 promotion needs a
namespace-local explicit baseline before it can create primary and confirmation
evidence.  This creates a one-time bootstrap boundary rather than permission to
copy historical evidence between namespaces.

The resolved legacy deployment pin retains both sides of the measured
improvement: its exact candidate confirmation and the resolved parent baseline
reference used by that confirmation.  Those immutable coordinates are enough
to repeat the scientific comparison under CURRENT without inventing a new
candidate or weakening deployment adoption.

## Decision

Add one trusted administrator operation, `bootstrap-current-baseline`.

1. The operation accepts only the current deployed `kernel.py` path and SHA-256.
   It takes no caller-selected namespace, baseline, protocol, image, timeout or
   promotion policy.
2. The deployment pin must be the exact built-in LEGACY namespace, resolved,
   and retain a resolved parent `BaselineRef`.  Git HEAD, the committed
   `kernel.py`, the working bytes, History, bundle/source CAS and the pin are
   revalidated before any GPU action.
3. The pinned parent artifact is remeasured byte-for-byte against itself under
   the exact built-in CURRENT protocol and resolved execution environment.  A
   narrow trusted recorder creates a new CURRENT `baseline_qualification` row.
   Its immutable evidence payload names the legacy source namespace, experiment
   ID/UID, artifact and environment digest, but the source experiment is never
   modified or treated as CURRENT evidence.  No cross-namespace History
   relation is created; ordinary experiment relations remain namespace-local.
4. The deployed candidate then traverses the normal CURRENT
   `POLICY -> SMOKE -> QUICK -> FULL_PRIMARY -> CONFIRMATION` workflow against
   that CURRENT baseline seed.  Shadow and holdout quick cases retain their
   protocol-defined roles.
5. Deterministic run and experiment identifiers make reconciliation idempotent.
   An interrupted or unknown GPU action is retained and is never automatically
   replayed.
6. Successful bootstrap produces eligible CURRENT primary and confirmation
   evidence only.  It cannot modify Git, configuration, the deployment pin,
   Campaign lineage, OJ state or profiling state.  Manual `adopt-baseline`
   remains the only deployment authority.

The existing `requalify-adoption` operation remains restricted to the original
LEGACY_UNKNOWN-to-resolved migration and cannot be used for CURRENT bootstrap.

## Rejected alternatives

- Relabel the resolved legacy experiments as CURRENT.  This invents a protocol
  provenance that was never executed.
- Compare the deployed candidate with itself and call the pair a normal
  promotion.  This fabricates an improvement decision.
- Allow an operator to nominate an arbitrary parent artifact.  That creates a
  second baseline authority outside the immutable deployment pin.
- Let a baseline qualification directly publish a deployment pin.  This
  collapses scientific evidence and administrative adoption.

## Validation and rollback

- Host tests must prove namespace, pin, Git, CAS, parent, environment and
  evaluator-echo mismatches fail before the first GPU launch.
- Recovery tests must prove a durable result is reconciled without replay and
  an unknown result remains quarantined.
- Checkpoint creation must compare the supplied Controller config with the
  frozen Run before creating a staging directory.
- Rollback restores the control-plane commit and complete runtime checkpoint;
  it never deletes or rewrites bootstrap experiments.
