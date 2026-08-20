# ADR-009: Bounded MetaX profiler image and pre-soak canary

## Status

Accepted; the A5 image was built, pushed, pulled by exact RepoDigest and
independently runtime/toolchain-qualified, then reached a trusted server
canary. Its untraced hardware warmup was killed by the container memory cgroup
at the frozen 4 GiB limit. The worker's `SIGKILL`, absent sentinel and phase
duration agree with the kernel `CONSTRAINT_MEMCG` record. The action remains
UNKNOWN, RESERVED, quarantined and non-replayable. A6 raises only the profiler
memory/resource contract and adds a pre-Docker host-memory gate. Its image has
now been built, pushed, pulled by exact RepoDigest and independently qualified;
the separate activation profile pins that image without changing build
identity. No A6 canary has run yet.

## Context

The host capability doctor proves that the deployment can see MetaX tools and
devices, but it does not make an arbitrary tracing container scientifically or
operationally trustworthy. The evaluator image contains mcTracer
`3.5.3.20-ef9e10e`, whose help and target exit codes are tool-specific: exit
status alone cannot prove either success or failure. Profiling must therefore
freeze the complete image, worker, recipe, and toolchain identity and produce
durable evidence about whether GPU execution started and completed.

Profiling remains advisory. It must not acquire promotion, lineage, Git, or
deployment authority merely because it uses the same candidate and device.

## Decision

1. The profiler is a separate immutable Linux/amd64 image derived from the
   exact evaluator RepoDigest. Its Dockerfile installs nothing and performs no
   network access. It copies only reviewed repository code and a fixed absolute
   entrypoint. The build fails unless the mcTracer binary, both MetaX support
   libraries, the DEVNULL help signature, and their frozen SHA-256 values match.
2. Profiler identity has two one-way layers. The build profile freezes only
   values knowable inside the corrected build image: base image, worker, entrypoint,
   toolchain, recipes, paths, user policy, timeouts, output/resource limits and
   isolation. Its digest excludes both the activation bit and final profiler
   RepoDigest, and is the only profile identity echoed by the worker. The host
   activation profile binds that build digest to `active` and the final exact
   RepoDigest. The image never claims to know its own registry identity. The
   runtime user is exactly `1000:1000`; its private `/tmp` tmpfs is mode `0700`
   and explicitly owned by the same UID/GID. Neither value is configurable for
   profiling.
3. The build submission retains an all-zero image digest sentinel and
   `active=false`.
   Every production profiling launch rejects this state before Docker. Submit
   B may change only those two activation values after a clean-tree image build
   is pushed and independently inspected. The build profile digest must remain
   byte-for-byte unchanged across A and B; Docker still launches the exact
   host-pinned RepoDigest with `--pull=never`.
4. The worker accepts exactly `metax-compile-metadata-v1` and
   `metax-hardware-counters-v1`. Candidate bundle identity and entrypoint bytes,
   CURRENT namespace, fixed case, limits, output paths, and request echo are
   mandatory. Unknown argv, path aliases, symlinks, and excessive output fail.
5. Compile metadata emits a non-empty canonical manifest trace. Unproven
   register, shared-memory, and spill metrics are
   `UNAVAILABLE/COUNTER_NOT_EXPOSED`; they are never inferred from unstable
   compiler text.
6. Hardware collection runs only `quick_decode_gate_up`, with ten warmups and
   ten tracked launches. mcTracer receives only fixed `--mctx`, `--odname`, and
   `--name` values. Exit 0 or 1 is accepted only with the exact target sentinel,
   passing correctness, no fatal/loader/exec marker, and a non-empty bounded
   trace. Trace files are packed into a deterministic metadata-normalized tar.
7. `outcome.json` is written atomically before GPU execution, after trusted
   target completion, and after evidence commit. A missing or incomplete
   outcome after GPU start is UNKNOWN and quarantines the lease. A fatal GPU
   marker is a hard failure. Proven pre-GPU or completed failures settle budget
   and release the lease. Container return code alone is never authoritative.
8. `profile image-doctor` creates or verifies a dedicated no-child CURRENT
   Campaign. It derives the candidate exclusively from the deployment pin,
   Git, History/CAS, the unique CURRENT bootstrap confirmation, and resolved
   execution environment. One `PROFILE_IMAGE_CANARY` action reserves 1,800
   seconds wall and 900 seconds GPU. Lock order is Campaign maintenance fence,
   `gpu1.lock`, then Campaign resource lease. Both recipes must pass.
9. Both canary recipes are individually bounded and their combined raw trace
   byte size must not exceed 64 MiB. The aggregate check occurs before READY;
   overflow is a known completed failure that settles budget and releases the
   lease.
10. Canary and ordinary collection write raw traces only to the controller
   private CAS and a whitelisted advisory summary to scientific CAS. They never
   write History or change a baseline. Collection schema V2 binds the complete
   toolchain and trace descriptor; there is no V1 dual-write because no
   production V1 profiling evidence exists.
11. The soak invariant contains both complete build and activation profiles and
   their digests. Changing the RepoDigest, activation bit, worker, recipe,
   toolchain, path, or limit invalidates prior soak completion and restarts the
   24/72/168-hour sequence.
12. Before every canary recipe launch, the Campaign ledger stores an immutable
    intent containing the fixed argv, recipe, image profiles, timeout and exact
    resource fence. Before the temporary output directory is removed, success
    and failure paths copy bounded stdout, stderr, worker outcome/result,
    available warmup/target sentinels and a non-symlink file inventory into the
    controller private CAS. The Campaign ledger links that object to the intent
    and records return code, timeout/output-limit flags and the unchanged
    known/UNKNOWN/hard classification. Diagnostics remain private and advisory.
13. A quarantined canary action is never resumed. A dedicated operator command
    first runs the ordinary trusted C500 doctor against the exact quarantined
    resource and fencing epoch, then atomically records an immutable
    abandonment, terminalizes the old Campaign, and changes only that old lease
    from QUARANTINED to RELEASED. The original action stays RESERVED and its
    UNKNOWN evidence stays intact. A later soak collector excludes this
    reservation from leak accounting only when the complete Campaign, action,
    doctor, lease, abandonment and diagnostic proof agrees; any missing or
    altered link fails closed. The next image doctor always uses a new Campaign.
14. Hardware warmup and tracked subprocesses each write one fixed atomic phase
    diagnostic before the worker parses a sentinel:
    `/output/warmup-process.json` and `/output/tracked-process.json`. Each binds
    the worker/build identity, phase, argv digest, timeout, duration, return
    code or termination signal, full stdout/stderr hashes, bounded output
    excerpts, and the observed sentinel state. A phase diagnostic is diagnostic
    only: it cannot establish GPU completion. Missing or inconsistent sentinel
    evidence remains UNKNOWN; fatal GPU markers retain their independent hard
    failure authority. The host copies available phase files into the same
    private canary diagnostic CAS object before temporary output is removed.
15. The profiler container memory limit is exactly 24 GiB, matching the
    already-qualified evaluator ceiling. Before any profiling state is created,
    and again immediately before every Docker launch, the trusted host reads
    `/proc/meminfo` and requires both `MemTotal` and `MemAvailable` to be at
    least 24 GiB. Parsing is strict, bounded, duplicate-free and unit-checked.
    The immutable canary attempt intent records the exact host observation, but
    that operational observation is private and never becomes scientific
    evidence. Failure at the first gate has zero Campaign/Docker side effects;
    failure after reservation but before a launch is a known pre-execution
    operator pause, settles wall use with zero GPU use and releases the lease.
    There is no caller-controlled memory value, swap extension, soft reserve,
    OOM disable, or worker-side cgroup telemetry in A6.

## Rejected alternatives

- Use the evaluator image directly for profiling. Its read-only HOME and
  entrypoint contract are not a trace-output contract, and doing so would
  broaden the evaluator trust domain.
- Treat mcTracer exit 1 as success. Both `/bin/true` and `/bin/false` can
  produce exit 1; only the target sentinel, correctness, and trace prove it.
- Parse undocumented counters as zero. Zero is a scientific measurement;
  absence is `UNAVAILABLE` with a fixed reason.
- Activate a tag or image ID. Only a registry RepoDigest is stable enough for
  the soak identity and offline `--pull=never` execution.
- Repair or replay an UNKNOWN canary in place. A fresh doctor may establish
  resource health and authorize explicit abandonment, but it cannot convert the
  old action into a known outcome. Any subsequent qualification uses a new
  Campaign. If the worker, image or recipe changes, it also requires a new
  image/activation identity and a fresh soak clock.
- Infer a known outcome from subprocess return code or phase diagnostic. A
  normal nonzero exit, signal, timeout, partial output, or apparently clean
  process still lacks completion authority without the exact sentinel.
- Reclassify the A5 action after identifying its OOM root cause. Operational
  diagnosis does not supply a target completion sentinel and therefore cannot
  grant completion or replay authority.
- Add cgroup-version-specific worker telemetry or weaken the container memory
  boundary. A6 keeps one host-owned, fail-closed admission gate and one exact
  hard limit instead of expanding the worker trust surface.

## Validation and rollout

- Local tests cover contract drift, worker output and trace bounds, Dockerfile
  restrictions, inactive pre-Docker rejection, V2 host evidence, Campaign
  budget/lease/failure semantics, the A-to-B identity transition, aggregate
  64 MiB enforcement, and explicit soak identity.
- The clean Linux/amd64 build from independent runtime-identity correction A4
  `92469041b4786d970ef93c05c14ef09ece8628a2` was built with the classic builder,
  pushed using a temporary Docker credential directory, pulled by digest, and
  requalified as the same image ID. Submit B pins
  `ghcr.io/masechen/autoresearch-metax-profiler@sha256:9d5516991a89945e7ad008c33ee847667831f5f7fc39fccf7030745f4ffa9acb`
  without changing the A4 build profile digest
  `sha256:a122359bc9c7d13356587964f51a4bdb68676f0841856d005cb380f1bd5facc7`.
  Deployment still uses the normal Admin update path.
- The superseded `e19cecc` Submit A must not be built: it coupled activation
  state to the worker echo. Submit A2 `559e8e9` fixed that identity but its
  `COPY --chmod` instruction cannot be parsed by the reviewed Docker 28.2.2
  classic builder. Submit A3 `1339fcc` fixed native image assembly, but its
  build-time toolchain probe ran as root and its runtime `0700` tmpfs remained
  root-owned; the resulting image failed non-root runtime qualification and was
  not pushed. Submit A4 binds runtime UID/GID and tmpfs ownership into a new
  build profile, upgrades the worker identity, and runs the build-time probe
  only after `USER 1000:1000`. It is never an amend or an in-place server
  workaround.
- The first trusted server action reached the compile recipe and stored its
  private CAS trace, then stopped during the hardware recipe without trusted
  completion. The Campaign remains the immutable record of that UNKNOWN result;
  it is not replayed or edited. A host-only diagnostic/recovery correction does
  not change the reviewed A4 image or build profile. After trusted doctor plus
  explicit abandonment, deployment of that control-plane correction, and a new
  Campaign, both compile manifest and hardware mctx archive must pass before
  soak begins. Any worker, recipe or image correction still requires a new
  build and activation commit.
- The host-only diagnostic/recovery correction was then exercised by a new
  canary. It successfully preserved the outer Docker diagnostic, proving the
  hardware worker reached untraced warmup, but the A4 worker discarded the
  inner subprocess result before sentinel parsing. A5 raises the worker
  revision to `metax-bounded-profiler-worker-v3`, adds the two bounded phase
  diagnostics to the build profile, and returns activation to `active=false`
  with an all-zero image digest. Its build profile digest is
  `sha256:13411bcfe57bcb74e1820e8ea60d30b6f5245dceabfb8785a2573ddaff43b49f`.
  A5 does not change the candidate, mcTracer contract, recipe launch counts,
  History, or baseline. The clean A5 source commit
  `28d499a789c9ae7d4485a1d7116e499c399eb3a0` produced image ID
  `sha256:f48545e69c4e41f98f942513508160554e4e28880fe272a6b8e0a227ab833d98`.
  The image was pushed, pulled and runtime/toolchain-qualified as
  `ghcr.io/masechen/autoresearch-metax-profiler@sha256:d88465d8ce23fb3edd2af5e610ea46b0ca174045b9bf671e8fe1029571dfb684`.
  The independent activation commit pins only that RepoDigest and
  `active=true`; the A5 build digest remains exactly
  `sha256:13411bcfe57bcb74e1820e8ea60d30b6f5245dceabfb8785a2573ddaff43b49f`.
  The next GPU action is a new, one-shot canary Campaign only after the old A4
  quarantine is explicitly abandoned and this activation is deployed.
- The A5 one-shot canary then reached the untraced hardware warmup. The kernel
  killed its UID 1000 Python process in the Docker memory cgroup at
  `2026-08-19T03:21:40Z`; the phase diagnostic recorded return code `-9`,
  `SIGKILL`, 8,191 ms, no timeout and no sentinel. mcTracer never started. The
  immutable root-cause archive is
  `/home/mx/autoresearch-evidence/profiler-a5-canary-oom-root-cause-20260819T032140Z`
  with manifest digest
  `54adaa64afee67f69b2cb06ef5a603104f699874295c6047395fe291cf626198`.
  The Campaign remains `PAUSED_UNKNOWN_OUTCOME`, its action remains RESERVED,
  and fencing epoch 3 remains QUARANTINED until the existing trusted abandon
  workflow is executed from the matching A5 recovery code.
- A6 is a new, direct-child control-plane build submission. It does not modify
  the worker, recipes, cases, toolchain, Dockerfile, candidate, History,
  baseline or `kernel.py`. It changes the exact memory limit from 4 GiB to
  24 GiB and freezes the `/proc/meminfo` admission thresholds into the build
  profile. Its inactive build profile digest is
  `sha256:432756dac8d6a00b7221ea5d39f09d8f33ec4ef48fa42badb35bdc20f2e59ab7`;
  its inactive activation digest is
  `sha256:0812342ab6a513319dd95a3d2b4e5a35751560a52a5c86b62584cd56695e9d05`.
  The clean A6 build commit
  `35f065d787bbf3c77bfe55c4ec5a1bd5bc1c3162` produced image ID
  `sha256:e955857c2c548e047149f786fc568f11bad2ad780380f2f30bac7f0e267983b4`.
  It was pushed, pulled and requalified as
  `ghcr.io/masechen/autoresearch-metax-profiler@sha256:2c817daef35c634b398209fce2d3c40b16d1f37acc9dbf00349e2540477719bb`.
  The separate activation commit changes only the activation bit and exact
  RepoDigest; the build digest remains unchanged. Its activation digest is
  `sha256:bd6dba397b68b7dde0f254ea12a916662c1bd10b1e0d6a99dd74545d3ad0c7ee`.
  The sealed build evidence is stored at
  `/home/mx/autoresearch-evidence/profiler-image-a6-20260820T064356Z`
  with manifest digest
  `4500eb4403dd3130e0b717d895dd7dc348c4ef1cba1f3b7167b84d357ef0f42a`.
  Only after the old A5 canary is abandoned, the A6 activation is deployed and
  a wholly new canary reaches READY may the 24/72/168-hour soak clock start.
