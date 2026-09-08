# Phase 3 Shadow Scoring Runbook

## Authority boundary

Phase 3 is observational. `XPUOJ_TH0_PROXY_V1` has no promotion, deployment,
lineage or soak authority. The existing evaluator promotion decision remains
unchanged. Correctness baseline, compiled scoring baseline and deployment
incumbent remain three separate identities.

## Measurement path

For each newly completed `full` evaluation whose result is successful and
correctness-qualified:

1. Persist the original `evaluate-raw` result unchanged.
2. Materialize the exact scoring framework snapshot frozen by the Run.
3. Persist a candidate shadow intent before Docker.
4. In a fixed evaluator container, regenerate each full case and self-pair the
   candidate under accelerator device-event timing.
5. Release all case objects, regenerate the cases, and pair the same candidate
   against the exact deployment incumbent for the safety channel.
6. Persist bounded stdout/stderr, receipt, terminal result and report digest.
7. Record the report in new History evidence. Console and Writer receive only
   the bounded case-free summary.

The worker identity includes its exact source SHA-256, and the container must
reproduce the exact resolved environment snapshot captured by the successful
Phase 2 qualification. A locally self-consistent but different environment is
not sufficient. Correctness ratios must be finite numbers in the closed
interval from the frozen minimum through `1.0`.

Smoke, quick, failed correctness, profile mismatch and known unqualified
measurements remain explicit `UNAVAILABLE`; they are never represented as
zero. Legacy History without a report remains `NOT_RECORDED`.

## Failure semantics

- `QUALIFIED` plus exit 0 is a trusted measurement completion.
- `UNQUALIFIED` plus exit 2 is a known, non-scoring completion.
- A last-moment authorization rejection while the durable operation remains in
  `PREPARING` is a known `LAUNCH_REJECTED` completion with
  `gpu_state=NOT_STARTED`; it is never retried automatically.
- Timeout, output overflow, fatal GPU marker, signal termination, malformed
  output, identity drift, interruption or status/exit mismatch is
  `UNKNOWN_OUTCOME`.
- Receipt is the terminal write authority. If final/state publication is
  interrupted after the receipt is durable, reconcile reconstructs the exact
  terminal envelope from that receipt without another Docker launch.
- An UNKNOWN operation and its Controller evaluation attempt are not replayed.
  Its exact container identity remains attached to the iteration, the Run is
  stopped, and the shared GPU lock rejects all later controller, campaign,
  profiler and admin GPU admission while the state is `LAUNCHING` or
  `UNKNOWN_OUTCOME`. A future trusted recovery must preserve the original
  operation evidence; manual container cleanup or state editing is forbidden.

The scientific report binds the candidate operation ID, semantic operation
digest and terminal result digest. Checkpoint verification re-proves the
intent/receipt/final/state chain and the matching History report. Legacy
History without these fields remains legacy evidence and is never upgraded by
inference.

## Deployment acceptance

Before deployment, require the full Python suite, branch coverage at least
80%, zero ResourceWarning, dependency-light host import, compileall, frontend
typecheck/lint/unit coverage/build/E2E, and `git diff --check`.

After Admin update, rerun static, doctor and the complete host regression. Then
run one same-hash CURRENT acceptance evaluation. Archive the raw evaluator,
candidate shadow operation, History report, checkpoint verification, unchanged
deployment pin/Campaign/profiler identities, and zero residual resources.

Only after that acceptance may Phase 4 begin: 20 independent same-hash null
runs followed by preregistered blinded XPU-OJ calibration.
