# Autoresearch Console V1 Runbook

## Authority and topology

The production path is Browser → Mac `127.0.0.1` Gateway → fixed SSH →
standard-library GPU-host Agent.  The GPU host opens no port and receives no
FastAPI, Node, browser dependency, Docker socket, or Console credential.  The
browser cannot submit SQL, shell, argv, paths, image/device names, cases, or
timeouts.  Administration, adoption, migration, restore, profiling, OJ, soak
observation, and quarantine release remain CLI/operator-only.

Every write is prepared against an exact `RuntimeIdentityV1`, expires after
five minutes, and requires the operation digest plus a fixed confirmation
phrase.  The detached worker has a deterministic domain identity.  Disconnects
are reconciled from durable ledgers and never authorize replay.

## Mac installation

1. Install CPython 3.13 and Node 24 LTS (`>=24.11,<25`) independently of the
   GPU host.
2. Create a dedicated virtual environment and install the final clean tree
   with `pip install '.[console]'`.
3. In `console/web`, run `npm ci`, `npm run release-preflight`, and
   `npm run sbom`, `npm run audit:release`, and `npm run build`.  The SBOM is
   deterministically bound to `package-lock.json`; the build publishes only
   local assets into `kernel_research/console/static`.
4. Create `~/.config/kernel-research-console/config.json` mode `0600` with the
   fixed SSH binary/target and the fixed remote Python/Admin-manifest paths.
   The local data directory is created mode `0700`; SQLite/CAS files are
   `0600`.
5. Run `kernel-autoresearch-console`.  Open only the emitted localhost URL;
   its fragment contains the one-use bootstrap token and is immediately
   removed from browser history.

Example configuration (identities only, never a secret):

```json
{
  "schema_version": 1,
  "ssh_binary": "/usr/bin/ssh",
  "ssh_target": "mx@node01",
  "remote_python": "/home/mx/miniforge3/bin/python3.13",
  "remote_admin_manifest": "/home/mx/autoresearch-runtime/gpu1/admin.json",
  "data_dir": "/Users/OPERATOR/.local/share/kernel-research-console",
  "poll_interval_ms": 2000
}
```

SSH must already have a trusted `known_hosts` entry and an available agent.
The Gateway always uses `BatchMode=yes`, `StrictHostKeyChecking=yes`, and
`ClearAllForwardings=yes`; it never accepts a new host key.

## Server deployment

The GPU host installs the final C5 commit with the existing `pip --no-deps`
control-plane procedure.  Verify that importing
`kernel_research.console.agent` does not import FastAPI, Uvicorn, NumPy,
Torch, or Triton.  Only the final C5 commit is deployed; intermediate Console
commits are never installed on the server.

Before Admin update, generate the inheritance proof:

```text
kernel-autoresearch-console-release \
  --repository /path/to/clean/final/tree \
  --protected-manifest /path/to/tree/docs/console/a9-protected-tree-v1.json \
  --static-dir /path/to/tree/kernel_research/console/static \
  --output /absolute/evidence/console-a9-inheritance.json
```

Failure, a protected-file mismatch, profiler identity drift, or a commit not
descending from `562d272...` forbids A9 canary inheritance.  In that case run
a new profiler canary; do not edit the proof.

After a successful proof: Admin update the final commit, run static/doctor and
the complete host suite, perform Agent handshake and SSE reconnect checks, and
prove read-only pages leave all three databases byte/logically unchanged.

## Acceptance and first scientific action

Use fake GPU fixtures for all four task families.  Then create one Console
manual evaluation using the exact currently deployed CandidateBundle.  It
must produce CURRENT POLICY→SMOKE→QUICK→FULL_PRIMARY→CONFIRMATION evidence,
with no proposer attempt, no duplicate GPU launch, no lease/container residue,
and no Deployment/Campaign/profiler mutation.  Archive the Console commit,
asset/protocol/proof digests, tests, handshake, receipt, and scientific chain.

Only then create a new Discovery Campaign and start a new 24/72/168 hour soak
from zero.  Any server Agent, protocol, controller, evaluator, profiler, or
soak-collector change requires a fresh invariant review.  A Mac-only asset
change may be released independently only when the Agent protocol and server
commit remain byte-identical.

## Failure and recovery

- `CHANGING`/identity reset disables all writes until a new handshake.
- Lost SSH/Gateway sessions use `GET /api/v1/operations/{id}`; never prepare a
  second operation.
- `UNKNOWN` means replay forbidden.  Use existing doctor/finalizer processes;
  the Web UI has no release escape hatch.
- Hard/unknown Campaign resume runs the pinned real doctor.  No evidence digest
  is accepted from the browser.
- Local drafts may be deleted after 30 days by a future offline retention
  command.  Audit/receipt records are append-only and are not auto-deleted.
