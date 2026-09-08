# XPU-OJ-Aligned Scoring V2 Implementation Plan

## Phase summary

1. Add dependency-light score mathematics, identities, and shadow reports.
2. Add a qualified `torch.compile` scoring baseline and device-event timing.
3. Integrate reports into Controller, History, writer feedback, and Console.
4. Calibrate same-hash noise and external OJ alignment before activation in a
   new research namespace.

## Phase 1 success criteria

- Continuous anchored score, arithmetic aggregation, canonical identities,
  null calibration, and dual-gate promotion logic are fully unit tested.
- Existing `evaluate_promotion()` and CURRENT namespace behavior are unchanged.

## Phase 2 success criteria

- Full-graph compiled reference is correctness-qualified without fallback.
- Device-event protocol records balanced raw rounds and excludes compile time.
- Absolute anchors use a compiled-reference self-pair in an anchor-only phase;
  the compiled-versus-eager comparison is a separate qualification proof and
  cannot supply the score anchor.
- All case anchors are measured before any eager performance comparison, and
  the exact phase order, raw-round validator, and thresholds are digest-bound.
- The two phases regenerate and release one deterministic case at a time.  No
  full-case tensor, reference output, or input snapshot may remain live while
  another case is prepared; this memory policy is part of the measurement
  contract digest.
- Baseline qualification produces an immutable descriptor or fails closed.
- The scoring worker runs from an exact Git-materialized scoring snapshot and
  echoes that commit into every probe, qualification, and baseline descriptor;
  it never changes the separately frozen scientific evaluator framework.
- A fixed argparse rejection before worker dispatch may be terminalized only by
  the no-argument trusted finalizer after exact raw-evidence and zero-container
  checks. It never authorizes replay or changes a scientific database.
- The archived `e099b2b` exit-137 incident has a separate no-argument recovery
  command.  It accepts only the exact immutable operation and direct reviewed
  recovery child, runs one fresh compile doctor, preserves the original UNKNOWN
  object, records the confirmed memory-cgroup OOM classification, and forbids
  replay.  A durable successful doctor may be resumed after interruption; an
  uncertain or failed doctor may never be repeated by this command.
- If recovery itself is rejected before Docker, its exact output and return
  code are preserved as a distinct known pre-Docker failure.  A follow-up
  recovery commit may issue one new durable retry intent and use only the
  Controller's existing standalone `preflight` doctor authorization; it may
  not overwrite the first intent or reinterpret the original UNKNOWN result.

## Phase 3 success criteria

- Every new result carries explicit objective and paired-safety evidence.
- Legacy History is not reinterpreted; missing values remain unavailable.
- Writer and Console labels say proxy, not official OJ score.

## Phase 4 success criteria

- At least 20 independent same-hash runs qualify the score threshold.
- Blinded external calibration passes the pre-registered correlation gates.
- A new namespace bootstraps the exact deployed candidate without relabeling
  History 210; any later soak restarts from zero.
