# ADR-006: Autoresearch Console V1

Status: accepted for implementation before the first production soak.

## Decision

The Console is a local operator adapter, not a new scientific authority.
The browser connects only to a Mac-local Gateway.  The Gateway reaches one
dependency-light Python agent through a fixed SSH command.  The GPU host does
not expose an HTTP port, install Web/Node dependencies, mount a Docker socket,
or retain a Console credential.

The remote agent reads History, Controller, and Campaign SQLite databases with
`mode=ro` and `query_only=ON`.  It never constructs a Store for a read request.
Writes use fixed domain operations and the existing Controller/Campaign locks,
leases, budgets, fencing, evidence, and no-replay semantics.  SQL, shell, raw
argv, paths, image names, devices, evaluator cases, and timeouts are not wire
inputs.

Console client identity and GPU-host control-plane identity are distinct.  A
Mac-only UI release may remain compatible with a frozen server agent without
changing a running soak.  Any server-side agent, controller, evaluator,
profile, or soak implementation change remains a new soak invariant.

## V1 authority boundary

V1 may create and operate autonomous Runs, manual CURRENT evaluations,
Discovery Campaigns, and Benchmark Campaigns.  Lineage advancement is allowed
only when the remote service independently re-proves the complete evidence
chain.  Manual evaluations can produce promotable scientific evidence but may
not change a Deployment or Campaign baseline.

Administration, adoption, migration, recovery, OJ submission, profiling
collection, trusted soak observations, and arbitrary quarantine release remain
outside the Web authority boundary.

## Security and recovery

The Gateway binds only to `127.0.0.1`, uses a one-time bootstrap token, a
SameSite session, CSRF and Origin checks, and a strict nonce CSP.  Candidate
bytes are bounded and content-addressed locally, then revalidated remotely.
Every mutation is a prepare/confirm operation with an immutable digest,
five-minute expiry, deterministic domain identity, and explicit reconciliation.
Disconnects never authorize a replay.

## A9 inheritance

Console development starts from Git object
`562d272aecf935c662abdc987df3aa5531328b73`.  The protected files and A9
identities in `docs/console/a9-protected-tree-v1.json` must remain byte-exact.
The final deployment changes the server Git invariant, so the 24/72/168 hour
soak starts only after Console acceptance.  The prior A9 profiler canary may be
inherited only after the protected-tree proof succeeds.
