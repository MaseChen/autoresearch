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
identity. Its one-shot canary then failed safely because Docker materialized
the frozen `/tmp` tmpfs as `noexec`, preventing Triton JIT shared objects from
being mapped. The failure is known, settled and released, and its Campaign was
terminalized by the trusted A7 finalizer without GPU or replay. The A7 native
build then failed before its non-root toolchain probe because a checkout made
under umask `0077` reached legacy Docker `COPY` as mode 0600/0700. A8 adds a
build-only, fail-closed library filesystem normalization contract. Its native
image has now been built, pushed, pulled by exact RepoDigest and independently
qualified; the separate activation profile pins that image without changing
build identity. Its one-shot canary completed compile and ten correct warmups,
then lost trusted completion when mcTracer could not initialize shared-memory
RPC under Docker `--ipc=none`. A9 replaces only that isolation axis with an
explicit private, bounded shared-memory namespace. No A9 image or canary
evidence exists yet.

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
16. The general private `/tmp` mount is explicitly `noexec`, remains 64 MiB,
    and retains `nosuid`, `nodev`, mode `0700` and UID/GID 1000. Triton JIT
    output receives a separate nested `/tmp/triton-cache` tmpfs that is
    explicitly `exec`, bounded to 1 GiB, and retains the same ownership and
    isolation flags. The parent mount must precede the child in the exact
    Docker argv. Neither mount is caller-configurable, and both are frozen into
    the worker-visible build profile and therefore the soak invariant.
17. A canary that already reached `PAUSED_OPERATOR` through a trusted known
    failure is never resumed or routed through GPU quarantine abandonment. A
    dedicated host-only finalizer accepts only the exact historical A6 image,
    build/activation identities and deployment snapshot; exactly one compile
    SUCCESS and one hardware KNOWN_FAILURE attempt with immutable diagnostics;
    one SETTLED action; one RELEASED lease; and zero child runs. It records an
    immutable finalization and terminalizes only the Campaign to `CANCELLED`.
    It invokes no doctor, GPU lock, Docker or evaluator and cannot mutate the
    action, lease, diagnostics, History or baseline. Any identity or state
    mismatch fails closed, and repeated calls only return the same proof.
18. The copied Python library tree is normalized before the runtime `USER`
    switch. A fixed build-only normalizer first rejects every symlink,
    non-directory/non-regular object and non-`.py` file, then uses
    `O_NOFOLLOW` descriptors to set root ownership, directory mode `0555` and
    Python-file mode `0444`. It re-enumerates and revalidates the complete tree
    before removal. The normalizer SHA-256, root, ownership, modes, allowed
    object types and runtime import probe are part of the build profile. A
    successful root build step is insufficient: after `USER 1000:1000`, the
    Dockerfile must import the package and run `verify-toolchain`. Host checkout
    modes therefore cannot change the image contract.
19. Profiler containers use exactly one private IPC namespace and one bounded
    `/dev/shm`: Docker argv contains `--ipc=private --shm-size 1g`. Host IPC,
    shareable IPC and container namespace reuse are permanently forbidden. The
    expected mount is root-owned mode `1777`, so the exact UID/GID 1000 runtime
    can create, read and delete private RPC objects; its capacity is exactly
    1 GiB and remains inside the 24 GiB container memory ceiling. IPC mode,
    size, bytes, path, ownership, mode, runtime identity and qualification probe
    are worker-visible Build Profile identity and soak invariants. No CLI can
    override them.

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
- Make all of `/tmp` executable. Triton requires executable JIT storage, but
  extending that authority to HOME, candidate staging and unrelated temporary
  files unnecessarily broadens the container attack surface.
- Reuse quarantine abandonment for a known failure. The A6 action has no
  quarantine and already has settled budget plus a released fence; creating a
  doctor or changing those rows would falsify its preserved outcome.
- Repair the A7 checkout with `chmod` and retry the same build. Checkout umask
  is not an image identity, and an operator-side repair is neither reviewable
  nor reproducible. A8 makes the normalization an immutable build step and
  changes the build digest.
- Use host, shareable or another container's IPC namespace. Those modes create
  cross-workload authority and can leak profiler RPC/shared-memory state. A9
  permits only a fresh private namespace. Retaining `--ipc=none` is also
  rejected because the A8 tracked probe demonstrated that mcTracer cannot
  establish its required RPC/shared-memory transport under that mode.

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
- The A6 canary `profile-image-canary-a6-20260821T030244Z` completed the compile
  recipe, then returned a trusted hardware warmup KNOWN_FAILURE. The exact
  `/tmp` mount was `rw,nosuid,nodev,noexec`: a test executable returned 126 and
  `ctypes` could not map a shared object from that mount. An explicitly
  executable tmpfs passed both probes, and a noexec parent with a nested
  executable `/tmp/triton-cache` also passed without GPU or database action.
  The Campaign is `PAUSED_OPERATOR`, its action is SETTLED, fencing epoch 4 is
  RELEASED, and replay remains forbidden. The sealed archive is
  `/home/mx/autoresearch-evidence/profile-image-canary-a6-20260821T030244Z`
  with manifest digest
  `f51411f52e681d237deffec4ba1de0cb8d73c7904f8b3dff23fc4cd597f96bc7`.
- A7 leaves worker v3, mcTracer, recipes, candidate, memory cap, toolchain,
  Dockerfile and `kernel.py` unchanged. It makes `/tmp` explicitly noexec,
  adds the exact nested executable cache mount, records a new build profile
  digest
  `sha256:cfc19d55524c5032d9b23200e6bc5b6a92c988a8cc01ec154867540a3101d734`,
  and returns activation to `active=false` with the inactive activation digest
  `sha256:d5a3c6ef8f8e03276d6d75bec9517adf0fa79d3c8a48c218a5ec445f2fe57345`.
  The exact A6 known failure must be finalized before deploying a separately
  qualified A7 activation. A new Campaign ID is mandatory for the next canary.
- The A6 finalizer was executed once from the exact A7 recovery code and its
  immutable archive is
  `/home/mx/autoresearch-evidence/profile-image-canary-a6-known-finalization-9362d96f-v1`
  with manifest digest
  `4880b3611d46ab6d59614b544d51fa130f279b3981094838d4e53ce8b9843686`.
  The subsequent A7 native build from `9362d96f073e7a080720351b7922e544046a2ed0`
  failed safely before the non-root toolchain probe. A checkout created under
  umask `0077` materialized tracked Python files/directories as 0600/0700;
  classic Docker `COPY` preserved those root-owned modes and UID 1000 could not
  read `kernel_research/__init__.py`. No final tag, push, GPU or database action
  occurred. The failure archive is
  `/home/mx/autoresearch-evidence/profiler-image-a7-9362d96f073e-build-v1`
  with manifest digest
  `2c0fd536e4f3076b57e492b2561ec89b18e5cbad044865d0e5cd31afa5b9ff39`.
  The failed intermediate image remains audit evidence and must not be reused.
- A8 is a direct-child build submission. It keeps the A7 dual-tmpfs,
  worker v3, recipes, toolchain, candidate, 24 GiB memory cap and `kernel.py`
  unchanged. It adds only deterministic build-time Python-tree normalization
  plus the non-root import proof. Its build profile digest is
  `sha256:bea1abedea118afbaa6206f4826b042a5ce57773fcf73c1a8f2a99ead1c9e549`;
  its inactive activation digest was
  `sha256:d4b3c5923c3aac6f933580fe687e72413b17cd51650cc9a71cc315fb03c9d34c`.
  A new clean checkout with its original hostile umask was built without
  operator-side chmod. The resulting image ID is
  `sha256:eba06d9807fcdbc0c3cdde61be6cf0b9f109779ab13f723ccff06fea94062bba`
  and the qualified RepoDigest is
  `ghcr.io/masechen/autoresearch-metax-profiler@sha256:4a118982bc868b9e0acd50ae0d3af8b8db1768802a7b096e220f801b131a9c35`.
  Local-tag and RepoDigest toolchain snapshots agree at
  `sha256:bd9d5e20698fc00c0a56bca122f4fe7be5e9a0528fbf98952ddfc6233c434293`.
  The control archive is
  `/home/mx/autoresearch-evidence/profiler-a8-control-preflight-6387fea2b1d7-v1`
  with manifest digest
  `81d193f605aed55d0f0a6efb4676039673eef8b326d2bae0eee7607fa9c4f784`;
  the image archive is
  `/home/mx/autoresearch-evidence/profiler-image-a8-6387fea2b1d7-build-v1`
  with manifest digest
  `cfd0d9ed9021a09cedd0a08330dfbaeac6ba0ec9f1d3edecc43098c9ef674cce`.
  The separate activation profile changes only the active bit and exact
  RepoDigest, preserves the build digest, and has activation digest
  `sha256:073f667748fc7d867e7333988c5daed7f574a0072e11e8bf731ded0e6b0138da`.
  Deployment verification completed with static, doctor and 519 host tests.
- The one-shot A8 canary `profile-image-canary-a8-20260823T173630Z`
  completed compile plus ten hardware warmups with correctness 1.0. The tracked
  mcTracer subprocess returned 0 without timeout or signal, but emitted
  `ftruncate`/`mmap` bad-file-descriptor errors and `Rpc connect timeout!`; it
  produced neither target sentinel nor trace. Completion therefore remains
  UNKNOWN, the action remains RESERVED, and gpu1 fencing epoch 5 remains
  QUARANTINED. It must not be replayed or manually cleaned. The full sealed
  manifest digest is retained with the server evidence and must be copied
  verbatim into the abandonment archive before A9 deployment.
- A9 changes only the IPC/shared-memory isolation contract. It restores
  `active=false` and the zero RepoDigest, keeps worker v3, mcTracer rules,
  recipes, candidate, toolchain, memory, dual tmpfs and `kernel.py` unchanged,
  and records build profile digest
  `sha256:560b4a3176a5b77c324b0453d3032a05700c2e7dae48c46be4cb9c0817ef7e7c`
  plus inactive activation digest
  `sha256:f1e4d0940d47ab25601f8e7853171a1eccb1d6b7bef099bf7022a6eb1f7c0282`.
  Before any A9 deployment, the exact activated A8 recovery worktree must run
  the trusted image-doctor abandonment for epoch 5 and archive the unchanged
  UNKNOWN/RESERVED diagnostics. A9 then requires a new image, separate
  activation commit and wholly new canary Campaign.
