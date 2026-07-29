# Fused MoE kernel autoresearch protocol

You are an autonomous GPU kernel researcher. Optimize the Triton candidate in
`kernel.py` for the MetaX C500 while preserving the fixed operator contract.

There are two execution modes. In the manual mode, a trusted operator edits
`kernel.py` using the steps below. In staged-candidate mode, OpenCode has no
tools and returns one complete `ProposalV1`; the host controller stores that
source outside Git and performs the same scientific stages. The proposer must
not assume it can inspect files, execute commands, continue a prior chat, or
repair a candidate in place.

## Non-negotiable boundaries

- Edit only `kernel.py` during experiments.
- Never modify `kernel_research/`, tests, the oracle, case generation, scoring,
  history, or benchmark timing.
- Treat `token_ids` as read-only metadata. `a` and `scale_a` are already routed.
- A 128-row tile has exactly one expert. No Triton program may apply one expert
  ID across rows from another tile.
- The fixed full-suite B tensor reaches linear offsets of 7,516,192,767
  elements, beyond signed int32. Every candidate must compute B's expert base
  with 64-bit-safe addressing (cast the expert ID before multiplying by the
  expert stride). Do not remove this invariant while tuning local offsets.
- INT8 products accumulate into INT32; scale in FP32; write BF16 in place.
- C500 mcTriton accepts only a literal power-of-two `num_warps` in
  `{1, 2, 4, 8, 16}`. Never request 32 or a dynamically computed warp count.
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

## Staged-candidate mode

Each proposal must be either one JSON object or one `json` fenced block with
exactly these fields and no others:

```json
{
  "schema_version": 1,
  "parent_candidate_hash": "<accepted 64-character lowercase SHA-256>",
  "hypothesis": "<one falsifiable hypothesis>",
  "rationale": "<evidence and expected effect>",
  "kernel_source": "<complete replacement kernel.py>"
}
```

The controller rejects parent mismatches, duplicate source hashes, extra text,
tool events, unknown fields, capability-expanding Python and missing int64-safe
B expert-base addressing before GPU access. Policy parsing is resource-bounded
and the evaluator repeats it before importing the candidate. A unique proposal
is then evaluated
in order: policy → smoke → quick → full primary → unchanged-hash full
confirmation. Only host-validated results are copied into the trusted history.

The autonomous controller never writes the repository's `kernel.py` and never
runs Git commit or push. A promoted staged artifact remains in runtime state
for a trusted operator to inspect and materialize later.

Compiler caches are isolated by evaluator image digest, framework commit and
candidate hash. The same source may reuse its cache through smoke, quick,
primary and confirmation, but a different candidate never receives it.

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
- `CRASH` or `TIMEOUT`: in manual mode, restore `kernel.py` and verify the
  C500 device. In staged-candidate mode, terminate the entire unattended
  session; do not retry automatically. The C500
  watchdog enforces 180 seconds for each compile phase and 300 seconds for each
  case phase; timeout JSON identifies the phase, case, and stage.
- An illegal-address, ATU, or Xnack `CRASH` can disable the vendor runtime for
  that worker. Confirm `mx-smi` reports `Available` before retrying. If the
  64-bit B expert-base invariant is intact, rerun once with
  `CUDA_LAUNCH_BLOCKING=1` and preserve the complete first-case log before
  changing the SDK or kernel.
- `UNSUPPORTED_ENV`: stop. Do not edit the kernel to hide an SDK/runtime problem.
- SIGINT/SIGTERM/SIGQUIT closes the run as `STOPPED` after exact-container
  cleanup. SIGKILL/OOM requires explicit operator inspection; do not assume an
  in-process cleanup handler ran.

Continue autonomously only on a healthy C500 environment. On mock mode, finish
the workflow check and wait for real hardware rather than generating experiments.
