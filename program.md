# Fused MoE kernel autoresearch protocol

You are an autonomous GPU kernel researcher. Optimize the Triton candidate in
`kernel.py` for the MetaX C500 while preserving the fixed operator contract.

## Non-negotiable boundaries

- Edit only `kernel.py` during experiments.
- Never modify `kernel_research/`, tests, the oracle, case generation, scoring,
  history, or benchmark timing.
- Treat `token_ids` as read-only metadata. `a` and `scale_a` are already routed.
- A 128-row tile has exactly one expert. No Triton program may apply one expert
  ID across rows from another tile.
- INT8 products accumulate into INT32; scale in FP32; write BF16 in place.
- Mock mode has no performance signal. Do not optimize, rank, or commit a kernel
  because of a mock result.

## Setup

1. Confirm the dedicated branch and clean state with `git status --short --branch`.
2. Run `python -m unittest discover -s tests -v`.
3. On macOS, run the mock doctor/evaluation only to verify orchestration.
4. On C500, run `python -m kernel_research doctor --backend c500` and preserve
   its JSON environment fingerprint and compile-probe result before the first
   experiment.
5. Establish a correct seed baseline on `smoke`, then `quick`, then `full`.
   Only a successful `full` run becomes the initial accepted baseline artifact.

## One experiment

1. Read the best accepted result and recent failures:

   ```bash
   python -m kernel_research history --format table
   ```

2. State one falsifiable hypothesis. Change one coherent aspect of `kernel.py`
   (for example block sizes, program mapping, loading order, warps, or stages).
3. Evaluate it, attaching the hypothesis as the note:

   ```bash
   python -m kernel_research evaluate --backend c500 --suite quick --note "<hypothesis>"
   ```

4. If quick correctness fails, stop that experiment. The source was archived by
   hash. Read the latest accepted row's `artifact_path` from JSON history, then
   restore only the controlled candidate (replace the example path exactly):

   ```bash
   python -m kernel_research history --format json
   cp .autoresearch/artifacts/<accepted-sha256>.py kernel.py
   ```

   Do not reset or restore the rest of the repository, and do not begin kernel
   mutation until the seed has produced the initial accepted full artifact.
5. If quick passes, run the full suite and inspect `promotion.phase`:
   - `primary`: do not edit `kernel.py`; run `full` once more with the exact
     same candidate hash and a confirmation note.
   - `confirmation` with `eligible_for_promotion: true`: the candidate is
     accepted.
   - `rejected` or a failed confirmation: restore the accepted artifact and
     begin a new hypothesis.
6. A changed source is never a confirmation, even if the note is similar.
   Commit accepted candidates only. Include the hypothesis and measured
   per-case/aggregate change in the commit message. Rejected candidates remain
   in `.autoresearch/artifacts/` and SQLite, not in Git history.

## Promotion rule

- Every case: matched ratio >= 0.99.
- Equal-weight geometric-mean speedup over the current accepted baseline >= 1.01.
- No individual case slowdown greater than 3%.
- The same conditions must hold in a second confirmation run.

Record raw evidence; do not round borderline measurements into a win. Prefer a
simpler kernel when performance is statistically indistinguishable.

## Failure handling

- `CONTRACT_ERROR`: fix the Python signature or syntax before any GPU work.
- `COMPILE_ERROR`: inspect compiler diagnostics and abandon incompatible Triton
  features instead of weakening the evaluator.
- `PRECISION_FAILED`: reduce the change to isolate indexing, masking, dtype, or
  expert-boundary errors.
- `CRASH` or `TIMEOUT`: the worker is fault-isolated; restore `kernel.py`, verify
  the C500 device is healthy, and continue with a safer hypothesis. The C500
  watchdog enforces 180 seconds for each compile phase and 300 seconds for each
  case phase; timeout JSON identifies the phase, case, and stage.
- `UNSUPPORTED_ENV`: stop. Do not edit the kernel to hide an SDK/runtime problem.

Continue autonomously only on a healthy C500 environment. On mock mode, finish
the workflow check and wait for real hardware rather than generating experiments.
