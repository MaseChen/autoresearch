# autoresearch-kernel-fused-moe

An external-agent research loop for optimizing a Triton implementation of a
quantized Fused MoE operator. The fixed framework owns data generation,
correctness, measurement, history, and process fault isolation. The research
agent edits exactly one file: [`kernel.py`](kernel.py).

The first milestone is intentionally split in two:

- **macOS mock mode** validates the control plane without importing PyTorch or
  Triton. It never invents latency and never promotes a candidate.
- **MetaX C500 mode** is the only mode allowed to claim kernel correctness or
  performance. The seed and adapter are implemented but remain unverified until
  run in a matching MXMACA/mcPyTorch/mcTriton environment.

## Operator contract

`kernel.py` must define:

```python
def run_kernel(
    a, b_col_major, scale_a, scale_b, moe_weights,
    token_ids, expert_ids, topk, out,
) -> None:
    ...
```

For routed row `r` and output column `n`:

```text
out[r, n] = sum_k(a[r, k] * b_col_major[expert(r), n, k])
            * scale_a[r] * scale_b[expert(r), n] * moe_weights[r]
expert(r) = expert_ids[r // 128]
```

`a` and `scale_a` are already expanded into routed-row order. `token_ids` is a
read-only compatibility input and must not trigger another gather. Accumulation
uses INT32, scaling uses FP32, and the in-place output is BF16.

## Quick start on macOS

Python 3.10+ and NumPy are sufficient; no GPU packages or network are required.

```bash
python -m kernel_research doctor --backend mock
python -m kernel_research evaluate --backend mock --suite smoke --note "seed control-plane check"
python -m kernel_research history --format table
python -m unittest discover -s tests -v
```

Runtime state is written below `.autoresearch/` and is intentionally untracked.
Candidate source is archived by SHA-256, and SQLite is the source of truth for
experiment history.

## C500 hand-off

Move the repository into the vendor-provided MXMACA environment. Do not install
stock PyTorch or Triton over that environment. Start with:

```bash
python -m kernel_research doctor --backend c500
python -m kernel_research evaluate --backend c500 --suite smoke --note "C500 seed smoke"
python -m kernel_research evaluate --backend c500 --suite quick --note "C500 quick validation"
```

Vendor references: [MACA installation guide](https://repos.metax-tech.com/gitea/repos/index/wiki/MACA)
and [MXMACA C500 release notes](https://developer.metax-tech.com/api/client/document/file/222/preview/?file_type=pdf).

The C500 doctor runs in an isolated worker and performs a minimal mcTriton
compile/launch probe. A runtime that merely imports, or a non-C500 accelerator,
is reported as `UNSUPPORTED_ENV`. Its stable environment fields cover the SDK,
driver, Torch, mcTriton, device, and compile capability; unavailable SDK/driver
versions remain `null` with an explicit probe error instead of being guessed.
The standalone doctor reports `compile_probe_status: PASSED` and
`compile_probe_passed: true` only after the probe completes. The lightweight
doctor embedded in evaluation intentionally reports `NOT_RUN` and `null`.
`driver_version` and `system_maca_version` are parsed from the supported
no-argument `mx-smi` status output, while the Torch build remains separately
recorded as `sdk_version`; a patch-version difference is evidence, not an
automatic rejection.

Only after the doctor and smaller suites succeed should `--suite full` allocate
the four DeepSeek-style production shapes. Cases are processed one at a time;
an unsupported or insufficient-memory environment fails explicitly rather than
silently shrinking a case. These B tensors exceed signed-int32 linear address
range, so the seed promotes the B expert base to int64 before multiplying by
the expert stride. Candidate kernels must preserve an equivalent 64-bit-safe
expert-base calculation.

The C500 benchmark is warm-cache steady-state: 10 warmups followed by three
blocks of 10 measurements. Raw samples and p20/p50/p80 are retained. A future
candidate is promotable only from the `full` suite, when every case reaches a
0.99 matched ratio, its equal-weight geometric-mean speedup is at least 1.01,
no case regresses more
than 3%, and a second `full` run of the exact same source hash repeats the
result. Compilation phases have a 180-second watchdog and each case phase has
a 300-second watchdog; timeout output records the active phase and case.

## Commands

```text
python -m kernel_research doctor --backend mock|c500
python -m kernel_research evaluate --backend mock|c500 --suite smoke|quick|full [--note TEXT]
python -m kernel_research history --format table|json|tsv
```

All machine-readable evaluation output has `schema_version: 1`. Logs and human
diagnostics go to stderr; JSON goes to stdout.

## Project structure

```text
kernel.py                    agent-owned Triton candidate
kernel_research/             fixed cases, oracle, worker, scoring, history, CLI
program.md                   external-agent operating protocol
tests/                       CPU-only unit and integration tests
.autoresearch/               ignored runtime database and candidate artifacts
```

## Trust boundary

AST validation proves only that the candidate has valid syntax and the required
function signature. A spawned worker protects the controller from ordinary
compiler crashes, illegal device accesses, and timeouts; it is not a security
sandbox. Run untrusted agents or candidates inside an OS/container boundary.

## Current limitations

- C500 compilation and numerical results have not been validated on this Mac.
- Full-suite correctness and performance remain pending until a C500 run
  completes all four production shapes; mock, smoke, or quick results do not
  establish that claim.
- The MVP has no built-in LLM client, MCTS, beam search, remote scheduler, or
  NVIDIA backend.
- Mock results are workflow evidence only and cannot be compared as performance.

See [`program.md`](program.md) for the autonomous experiment protocol.
