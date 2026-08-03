# autoresearch-kernel-fused-moe

An external-agent research loop for optimizing a Triton implementation of a
quantized Fused MoE operator. The fixed framework owns data generation,
correctness, measurement, history, and process fault isolation. It supports a
manual workflow and a bounded OpenCode + DeepSeek proposal workflow.

The first milestone is intentionally split in two:

- **macOS mock mode** validates the control plane without importing PyTorch or
  Triton. It never invents latency and never promotes a candidate.
- **MetaX C500 mode** is the only mode allowed to claim kernel correctness or
  performance. The current accepted `88b9eb6f…` 128×128 tile kernel has
  completed smoke, quick, full primary and full confirmation on C500; each new
  server still establishes its own accepted full baseline before comparisons.

## Project documents

- [项目改造说明](docs/project-overview.md)
- [当前进度报告（2026-07-28）](docs/status-report-2026-07-28.md)

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
The macOS system Python 3.9 is not a supported test interpreter.

```bash
python -m kernel_research doctor --backend mock
python -m kernel_research evaluate --backend mock --suite smoke --note "seed control-plane check"
python -m kernel_research history --format table
python3.10 -m unittest discover -s tests -v
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

kernel-autoresearch doctor --config CONFIG
kernel-autoresearch start --config CONFIG
kernel-autoresearch resume --config CONFIG --run-id ID
kernel-autoresearch status --config CONFIG [--run-id ID]
kernel-autoresearch stop --config CONFIG --run-id ID
kernel-autoresearch checkpoint --config CONFIG --run-id ID

kernel-autoresearch-admin bootstrap --manifest PATH --pro-config PATH --flash-config PATH
kernel-autoresearch-admin sync --manifest PATH
kernel-autoresearch-admin verify --manifest PATH --level static|cpu|doctor
kernel-autoresearch-admin update --manifest PATH [--doctor]
kernel-autoresearch-admin adopt-baseline --manifest PATH --candidate-hash HASH [--doctor]
```

All machine-readable evaluation output has `schema_version: 1`. Logs and human
diagnostics go to stderr; JSON goes to stdout.

`python -m kernel_research evaluate-raw` is an internal controller interface.
It repeats the bounded autonomous policy before candidate import, then emits
evidence without opening SQLite or making a promotion decision. The public
manual `evaluate` command intentionally retains its contract-only policy.

## Controlled OpenCode workflow

The host controller is standard-library-only. OpenCode is a replaceable
`Proposer`: it receives a bounded prompt and returns a complete `ProposalV1`.
It never receives a repository mount, GPU device or Docker socket. The trusted
host validates the response, stages its source outside the repository, and
runs smoke → quick → full primary → same-hash confirmation. It writes accepted
and rejected evidence to the existing history only after validating raw
container output. It never replaces `kernel.py`, commits, or pushes.

Copy [`config/autorun.example.json`](config/autorun.example.json) outside the
repository and replace the proposer image reference and `expected_git_commit`.
Every configured path must be absolute and canonical; both images must use a
repository digest (`name@sha256:...`). The current baseline kernel hash is
`88b9eb6f612dbe47e2e59498fd8c45df305e832524155dafb310cc98b26cf9b9`.
Its accepted provenance commit is
`03fd62cf3a32b907c9e2d88f8b4ac9b1c65a087c`; the controller commit is the newer
exact `git rev-parse HEAD` value placed in the deployment config.

The proposer supports exactly two audited DeepSeek model IDs:

```json
"opencode_model": "deepseek/deepseek-v4-pro"
```

or:

```json
"opencode_model": "deepseek/deepseek-v4-flash"
```

Pro remains the default when the field is omitted. Model selection is part of
the immutable run config: never change it when resuming a run. Unknown models,
aliases and other providers are rejected, and API/network failure never causes
an automatic Pro↔Flash fallback. This keeps every proposal attributable to one
model even when provider connectivity is unstable.

Both audited models use a one-million-token context window and a project-side
65,536-token model-output ceiling. Thinking remains enabled with maximum
reasoning effort. This leaves room for ProposalV1 after a long reasoning trace
while the host still independently limits OpenCode to three steps, 20 minutes,
2 MiB of captured output and a 256 KiB kernel source. A `step_finish` with
`reason=length` and no complete Proposal is reported as
`PROPOSER_OUTPUT_TOKEN_LIMIT`; it is not treated as malformed JSON, retried, or
silently sent to the other model.

The example deliberately sets `acknowledge_gpu_passthrough_risk` to `false`.
`doctor` remains available, but candidate GPU evaluation through `start` or
`resume` is blocked until it is explicitly set to `true` for an authorized
GPU. Direct GPU device mounting is not an adversarial kernel sandbox.

Build the pinned OpenCode 1.17.7 image, push it to a registry you control, and
record the returned repository digest:

```bash
cd /home/mx/workspace/autoresearch
docker build --pull --no-cache \
  -f containers/opencode/Dockerfile \
  -t YOUR_REGISTRY/kernel-autoresearch-opencode:1.17.7 .
docker push YOUR_REGISTRY/kernel-autoresearch-opencode:1.17.7
docker image inspect --format '{{json .RepoDigests}}' \
  YOUR_REGISTRY/kernel-autoresearch-opencode:1.17.7
```

Create a dedicated, provider-budget-limited key without putting it in an
environment variable or project file:

```bash
install -d -m 700 /home/mx/.config/kernel-autoresearch
umask 077
read -r -s -p 'DeepSeek API key: ' DEEPSEEK_KEY
printf '%s' "$DEEPSEEK_KEY" \
  > /home/mx/.config/kernel-autoresearch/deepseek-api-key
unset DEEPSEEK_KEY
chmod 600 /home/mx/.config/kernel-autoresearch/deepseek-api-key
```

Install only the console wrapper (the controller itself does not import the
project's NumPy dependency), validate, then launch inside `tmux`:

```bash
python3 -m pip install --user --no-deps -e .
kernel-autoresearch doctor \
  --config /home/mx/autoresearch-runtime/gpu1/autorun.json
tmux new -s fused-moe-research
kernel-autoresearch start \
  --config /home/mx/autoresearch-runtime/gpu1/autorun.json
```

The defaults stop after five unique proposals that pass strict Proposal
validation, six hours, three consecutive proposer/controller failures, or the
first confirmed promotion. Scientific rejections do not consume the controller
failure budget. `CRASH`, `TIMEOUT`, `UNSUPPORTED_ENV`, exit 137, ATU, Xnack or
illegal-address evidence stops the entire run without an unattended retry.
`resume` always requires an explicit run ID; if primary completed, it resumes
the same candidate at confirmation.

SIGINT, SIGTERM and SIGQUIT use the same controlled shutdown path: the exact
full-run-ID container is killed, the interruption is recorded, and the run
closes as `STOPPED`. SIGKILL and host OOM cannot be handled in-process; use the
recorded exact container name for operator recovery.

For the deployment canary, make exactly one live DeepSeek/OpenCode proposal
without candidate GPU evaluation:

```bash
kernel-autoresearch start \
  --config /home/mx/autoresearch-runtime/gpu1/autorun.json \
  --proposal-only
```

This still runs the trusted C500 doctor, but stops at `PROPOSAL_READY` after
ProposalV1 and research-policy validation. Inspect its prompt, raw NDJSON and
staged source before starting the one-candidate C500 dry run with a copied
config whose `max_candidates` is `1`.

To compare Pro and Flash, create `autorun.pro.json` and
`autorun.flash.json` from the same validated config and change only
`opencode_model`. Start a new run for each model; do not resume a Pro run with
the Flash config or vice versa. Run Flash proposal-only and one-candidate GPU
canaries before a five-candidate session. Keep the accepted baseline, image
digests, budgets and protocol identical when comparing model outcomes.

### Server administration

Git/config maintenance is deliberately outside the research controller. After
installing this version, import the two already validated model configs once:

```bash
kernel-autoresearch-admin bootstrap \
  --manifest /home/mx/autoresearch-runtime/gpu1/admin.json \
  --pro-config /home/mx/autoresearch-runtime/gpu1/autorun.pro.json \
  --flash-config /home/mx/autoresearch-runtime/gpu1/autorun.flash.json

source /home/mx/autoresearch-runtime/gpu1/env.sh
kernel-autoresearch-admin verify \
  --manifest "$AUTORESEARCH_MANIFEST" \
  --level doctor
```

`bootstrap` requires the input configs to differ only by model. It creates a
shared base, formal five-candidate configs, one-candidate canary configs and a
secret-free `env.sh`, all mode 0600. `sync` updates commit pins after a trusted
local Git change but never adopts a new kernel hash. The only supported hash
transition is explicit `adopt-baseline` after the same hash is committed and
recorded as an accepted C500 full confirmation.

Subsequent fast-forward deployments use:

```bash
kernel-autoresearch-admin update \
  --manifest /home/mx/autoresearch-runtime/gpu1/admin.json \
  --doctor
```

The updater takes the GPU1 cooperative lock, rejects active runs/containers,
fetches only the fixed tracking branch, requires a fast-forward, runs fixed-
image CPU tests, reinstalls the host entry point, stages config synchronization
and optionally runs both doctors. Formal configs publish only after all checks
pass. It never resets Git, pulls images, reads the API key value, invokes
DeepSeek, starts a run, commits or pushes. See
[server administration](docs/server-admin.md) for the full procedure.

### Server acceptance sequence

Use a dedicated runtime directory and retain each JSON response. Do not start
the five-candidate session until all of these checks pass:

1. Run `kernel-autoresearch doctor`; confirm `status=SUCCESS`, the compile probe
   is `PASSED`, and the GPU-risk acknowledgement value matches the config.
2. Run the proposal-only canary above with risk acknowledgement still false.
3. Copy the config, set `max_candidates` to `1` and
   `acknowledge_gpu_passthrough_risk` to true, then run one complete staged
   session. Confirm smoke, quick, full-primary and confirmation evidence is
   present or that a scientific rejection stopped that candidate cleanly.
4. Exercise safe controller fault fixtures without submitting a hostile GPU
   program:

   ```bash
   PYTHONPATH=tests python3.10 -m unittest -v \
     test_autorun.StateMachineTests.test_quick_compile_failure_and_full_performance_rejection \
     test_autorun_hardening.CommandRunnerHardeningTests.test_high_rate_output_is_bounded_and_killed_immediately \
     test_autorun_hardening.StoreAndRecoveryTests.test_controller_signal_stops_and_cleans_exact_container
   ```

5. For the canary run ID, require both commands below to return no containers;
   inspect the candidate-hash cache directories and confirm no candidate shares
   another candidate's path:

   ```bash
   docker ps -aq --filter "label=kernel-autoresearch.run=RUN_ID"
   find /home/mx/autoresearch-runtime/gpu1/cache -mindepth 3 -maxdepth 3 -type d
   ```

6. Create a checkpoint, open both copied databases with
   `PRAGMA integrity_check`, and verify every file against `manifest.json`.
   The controller's checkpoint tests perform the same reconstruction and
   SHA-256 checks locally.
7. Only then enable the normal five-candidate, six-hour config. The first
   confirmed promotion or any hard GPU/runtime fault terminates the session.

## Project structure

```text
kernel.py                    agent-owned Triton candidate
kernel_research/             fixed cases, oracle, worker, scoring, history, CLI
kernel_research/autorun/     trusted host controller and proposer adapter
containers/opencode/         pinned no-tool OpenCode proposer image
config/                      secret-free deployment example
program.md                   external-agent operating protocol
tests/                       CPU-only unit and integration tests
.autoresearch/               ignored runtime database and candidate artifacts
```

## Trust boundary

The autonomous path adds a conservative research-policy AST gate, but that gate
is not a Python sandbox. Policy parsing runs in a short-lived worker with
source, CPU, wall-clock, Linux memory and output bounds, and is repeated inside
`evaluate-raw`. The proposer container has network but no repository, devices,
state or Docker socket. The evaluator container is offline and sees only a
commit-keyed framework snapshot, one candidate, an optional accepted baseline,
a candidate-specific compiler cache and the three authorized GPU1 devices.
Cache namespaces include evaluator digest, framework commit and candidate hash.
The evaluator does not see either SQLite database. Both containers are
non-root, read-only, capability-free and resource-bounded.

The trusted host process is the only component allowed to call Docker and
update history. Membership in the Docker group is effectively root-equivalent,
so never expose its shell or socket to the Agent. Device isolation limits
ordinary mistakes; an adversarial accelerator kernel may still crash or attack
the shared driver. Without a VM/IOMMU boundary this design does not claim
adversarial GPU isolation.

Controller state uses a validated stage machine and transactional state/event
writes. Checkpoints use SQLite backup, reopen both copies for `integrity_check`,
and include database counts plus a SHA-256 file manifest. The two databases
remain separate; deterministic run notes and candidate hashes reconcile the
narrow cross-database window.

For local branch-aware coverage evidence:

```bash
python3.10 -m pip install -e '.[dev]'
python3.10 -m coverage run -m unittest discover -s tests -v
python3.10 -m coverage report
```

Acceptance requires an unrounded total of at least 80% and branch-only coverage
of at least 85% for `autorun/controller.py`, `autorun/runtime.py`, and
`autorun/store.py`. The server-administration release also requires at least
85% branch-aware coverage for `autorun/admin.py`.

The Flash-output/admin release passes 133 Python 3.10 tests with 83.71% total
branch-aware coverage; `autorun/admin.py` is 85.36%. The preceding dual-model
release recorded 118 tests and 83.31%.

## Current limitations

- C500 results cannot be reproduced on macOS; every deployment must pass its
  own doctor and establish an accepted full baseline.
- The MVP has no in-process LLM client, MCTS, beam search, remote scheduler, or
  NVIDIA backend. `opencode` is the only proposer adapter in this release.
- Mock results are workflow evidence only and cannot be compared as performance.

See [`program.md`](program.md) for the autonomous experiment protocol.
