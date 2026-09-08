# Scoring Baseline Stability Pilot

## Immutable result

The first memory-bounded ten-probe qualification completed on the production
C500 host at control commit
`d6b8a983c1de3b7542d81f75349bf33b10151a87`.

- operation: `score-baseline-c9c8382e07f0e3f158fb2ffc`
- measurement contract:
  `sha256:0a8ac7207dd24d0604c8798a1b937f09bd4da160939f59625f4a63de919d7796`
- terminal status: `UNQUALIFIED`
- reason: `relative_mad_exceeded`
- evidence:
  `/home/mx/autoresearch-evidence/xpuoj-scoring-d6b8a98-qualification-v1`
- evidence manifest SHA-256:
  `fa3a6fc4911978b6f3e05c6dd5da83ff067a59c9eebf01dcb126d830e03f56ad`

All ten probes were individually qualified. Every compiled and eager
correctness ratio was 1.0, the compiled-to-eager proof passed for all four
cases, all identities matched, and no OOM or unknown outcome occurred. The
aggregate failed only because `full_decode_down` had relative MAD
`0.005096581386109984`, above the V1 per-case maximum `0.005`.

The result is permanent. It is not retroactively reclassified and cannot
activate a scoring baseline.

## Pilot diagnosis

The V1 rule accepted or rejected the full score from the noisiest individual
case. That does not directly measure stability of the XPU-OJ-aligned target,
which is the equal-weight arithmetic mean of four case scores.

Recomputing that target from the immutable receipts produced:

- aggregate self-score median: `49.99742018984487`
- aggregate self-score MAD: `0.047678360092760386` points
- relative MAD: `0.000953616404840914`
- minimum: `49.82586213507561`
- maximum: `50.04532530035382`
- range: `0.21946316527820642` points

The raw rounds did not show a persistent AB/BA channel bias. Per-probe block
MAD was small; the rejected variation was primarily cross-container absolute
clock variation in the shortest case.

## V2 preregistered stability contract

The pilot is used only to motivate a new contract. A fresh ten-probe
validation is required; the old probes cannot qualify V2.

- each case relative MAD must be at most 1%;
- aggregate self-score MAD must be at most 0.125 score points;
- aggregate median must be within 0.05 points of parity score 50;
- every probe aggregate must be within 0.5 points of parity score 50.

The aggregate MAD budget is half the existing minimum visible score delta of
0.25 points. The maximum probe deviation equals the existing maximum qualified
noise threshold of 0.5 points. The per-case guard remains much stricter than
the roughly 5% run-to-run latency variation documented by SOL-ExecBench even
for its clock-locked official workers.

The exact contract, thresholds, aggregation formula, case weighting and probe
count are digest-bound and echoed by every probe. Qualification evidence V2
also carries its aggregate score envelope. Idempotent verification
reaggregates all ten receipts instead of trusting serialized summary fields.

## Method references

- PyTorch Benchmark uses replicates and robust median/IQR statistics and offers
  adaptive measurement until a configured variability bound is reached:
  <https://docs.pytorch.org/docs/stable/benchmark_utils.html>
- Triton `do_bench` uses explicit warmup/repetition budgets and exposes raw,
  median and quantile summaries:
  <https://triton-lang.org/main/python-api/generated/triton.testing.do_bench.html>
- SOL-ExecBench documents clock locking, repeated measurements and expected
  run-to-run timing variation:
  <https://github.com/NVIDIA/SOL-ExecBench>
