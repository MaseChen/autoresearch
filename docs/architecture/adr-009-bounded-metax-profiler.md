# ADR-009: Bounded MetaX profiler image and pre-soak canary

## Status

Accepted; image profile inactive until a reviewed RepoDigest is published.

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
2. Submit A retains an all-zero digest sentinel and `active=false`. Every
   production profiling launch rejects this state before Docker. Submit B is
   allowed only after a clean-tree build has been pushed to the private GHCR
   repository and the returned RepoDigest has been independently inspected.
3. The worker accepts exactly `metax-compile-metadata-v1` and
   `metax-hardware-counters-v1`. Candidate bundle identity and entrypoint bytes,
   CURRENT namespace, fixed case, limits, output paths, and request echo are
   mandatory. Unknown argv, path aliases, symlinks, and excessive output fail.
4. Compile metadata emits a non-empty canonical manifest trace. Unproven
   register, shared-memory, and spill metrics are
   `UNAVAILABLE/COUNTER_NOT_EXPOSED`; they are never inferred from unstable
   compiler text.
5. Hardware collection runs only `quick_decode_gate_up`, with ten warmups and
   ten tracked launches. mcTracer receives only fixed `--mctx`, `--odname`, and
   `--name` values. Exit 0 or 1 is accepted only with the exact target sentinel,
   passing correctness, no fatal/loader/exec marker, and a non-empty bounded
   trace. Trace files are packed into a deterministic metadata-normalized tar.
6. `outcome.json` is written atomically before GPU execution, after trusted
   target completion, and after evidence commit. A missing or incomplete
   outcome after GPU start is UNKNOWN and quarantines the lease. A fatal GPU
   marker is a hard failure. Proven pre-GPU or completed failures settle budget
   and release the lease. Container return code alone is never authoritative.
7. `profile image-doctor` creates or verifies a dedicated no-child CURRENT
   Campaign. It derives the candidate exclusively from the deployment pin,
   Git, History/CAS, the unique CURRENT bootstrap confirmation, and resolved
   execution environment. One `PROFILE_IMAGE_CANARY` action reserves 1,800
   seconds wall and 900 seconds GPU. Lock order is Campaign maintenance fence,
   `gpu1.lock`, then Campaign resource lease. Both recipes must pass.
8. Canary and ordinary collection write raw traces only to the controller
   private CAS and a whitelisted advisory summary to scientific CAS. They never
   write History or change a baseline. Collection schema V2 binds the complete
   toolchain and trace descriptor; there is no V1 dual-write because no
   production V1 profiling evidence exists.
9. The soak invariant contains the profiler profile digest, exact image
   RepoDigest, worker revision, activation bit, recipes, toolchain, and limits.
   Changing any of them invalidates prior soak completion and restarts the
   24/72/168-hour sequence.

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
- Repair an UNKNOWN canary in place. The resource remains quarantined and a
  changed image or worker is qualified in a new Campaign.

## Validation and rollout

- Local tests cover contract drift, worker output and trace bounds, Dockerfile
  restrictions, inactive pre-Docker rejection, V2 host evidence, Campaign
  budget/lease/failure semantics, and explicit soak identity.
- A clean Linux/amd64 build from Submit A is pushed using a temporary Docker
  credential directory. Submit B pins the returned RepoDigest and is deployed
  through the normal Admin update path.
- The first trusted server action is one image-doctor Campaign. Soak does not
  begin until its compile manifest and hardware mctx archive both pass. Any
  canary fix produces a new image and activation commit.
