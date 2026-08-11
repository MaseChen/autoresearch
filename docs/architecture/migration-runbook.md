# Coordinated V2-to-V3 migration runbook

History and Controller are one migration recovery unit even though they remain
separate bounded contexts at runtime. Opening a persisted V1 or V2 store does
not migrate it. Only the coordinated migration command holds the opaque
capability required to cross that boundary.

A persisted database marked `user_version=0` is considered new only when it
contains no user-defined SQLite objects. Resetting the pragma on an existing
schema cannot bypass the coordinator; both Stores reject that state before
enabling WAL or another write-affecting pragma.

## Preconditions

1. Stop Controller, Campaign, evaluator and proposer processes. Verify there
   are no active containers or Campaign resource leases.
2. Verify every Controller Run and iteration is terminal.
3. Use the exact deployment configuration whose repository path and pinned Git
   commit describe the code currently operating the databases.
4. The repository must be a clean, non-symlink Git worktree at that pinned
   commit. Runtime state and credentials must remain outside it.
5. Select an empty checkpoint parent on durable storage with enough space for
   both databases, all legacy artifacts, the configuration and a full Git
   bundle.

## Execute

```text
kernel-autoresearch migrate-v3 \
  --config /absolute/runtime/controller.json \
  --checkpoint-root /absolute/migration-checkpoints \
  --operation-id planned-v3-migration
```

Before either live Store is opened, the coordinator:

- inspects both databases read-only and requires exact V2 schemas, SQLite
  integrity, foreign-key integrity and terminal workflow state;
- verifies every referenced legacy artifact and inventories the complete
  legacy artifact tree;
- verifies the raw configuration contains no inline secret field, binds its
  canonical path, repository and expected commit;
- requires a clean repository at the expected commit and creates a full Git
  bundle that passes an offline `git bundle verify` check;
- publishes an exact-cover, SHA-256 manifest through a private staging
  directory; and
- migrates throw-away copies before revalidating the live sources.

Immediately before live migration it publishes matching intent fences beside
both databases. Ordinary History, Controller, legacy CLI and admin access is
then rejected. On success both stores are V3 and the matching fences are
removed.

## Failure and rollback

There is no one-database, one-file or schema-downgrade rollback. If failure
occurs after live migration begins, preserve the intent fences and stop all
writers. The structured error identifies the immutable migration checkpoint
and observed schema versions.

Recover the checkpoint as one unit in an inactive area:

- restore `history.sqlite3`, `controller.sqlite3` and the entire `artifacts/`
  tree together;
- restore the exact raw configuration;
- recreate or reset an inactive repository from `code/repository.bundle` at
  the manifest's expected commit; and
- verify the checkpoint manifest, SQLite integrity, artifact digests, config
  digest and Git bundle again before an operator switches the complete runtime
  unit active.

Do not remove a migration intent fence merely to make a mixed V2/V3 pair open.
The fence is recovery evidence. Remove it only as part of a verified whole-unit
restore or after the coordinator reports complete success.
