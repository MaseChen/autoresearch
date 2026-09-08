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
- Baseline qualification produces an immutable descriptor or fails closed.

## Phase 3 success criteria

- Every new result carries explicit objective and paired-safety evidence.
- Legacy History is not reinterpreted; missing values remain unavailable.
- Writer and Console labels say proxy, not official OJ score.

## Phase 4 success criteria

- At least 20 independent same-hash runs qualify the score threshold.
- Blinded external calibration passes the pre-registered correlation gates.
- A new namespace bootstraps the exact deployed candidate without relabeling
  History 210; any later soak restarts from zero.
