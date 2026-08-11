# Disaster recovery runbook

This runbook covers disaster restore of an immutable Controller checkpoint.
It is not the normal Run or Campaign resume path. Normal resume reconciles the
live databases by immutable run, experiment, and budget-action identities.

## Safety contract

- Stop the trusted host and all Campaign supervisors before recovery. Verify
  that no evaluator or proposer container is still running and no resource
  lease is active.
- Treat one checkpoint as an indivisible recovery unit. A non-Campaign
  checkpoint contains Controller and History databases. A Campaign checkpoint
  additionally contains the matching Campaign database. Never mix databases,
  CAS objects, artifacts, or configuration from different checkpoints.
- Restore only into a new, canonical, inactive directory. The restore command
  refuses to overwrite an existing tree and never changes the active runtime.
- Do not edit `resolved-config.json`, database rows, artifact bytes, manifests,
  or digests to make a restore fit another location. The immutable run snapshot
  still names its original deployment paths and execution environment.
- `READY_FOR_MANUAL_SWITCH` means that copied bytes and SQLite integrity have
  been verified. It does not mean that the restored tree has become active or
  that a Run is safe to resume.

## Restore and verify

Choose a sibling directory on the same filesystem as the intended active
runtime root:

```text
kernel-autoresearch restore-checkpoint \
  --checkpoint /absolute/checkpoints/RUN-TIMESTAMP \
  --destination /absolute/runtime.restore-TIMESTAMP
```

The command verifies exact manifest coverage and SHA-256 for every file, opens
each included SQLite database read-only for `PRAGMA integrity_check`, copies
through a private staging directory, verifies the copied databases, and then
publishes only the new inactive directory. A Campaign manifest must include its
Campaign database and a non-Campaign manifest must not contain one.

Verification also proves ownership across all three databases: the Controller
Run, frozen BaselineRef, Campaign child intent, active baseline revision,
namespace, budget bounds, and History artifact/evaluation links must agree.
Adding an otherwise valid Campaign database to a non-Campaign checkpoint is
rejected.

Before switching, an operator must additionally verify:

1. the checkpoint `run_id`, `campaign_id`, namespace, baseline revision, and
   resolved profile digests are the intended recovery point;
2. the deployment profile, evaluator image, framework commit, toolchain and
   device allocation still match the frozen execution-environment digest;
3. the configured stable runtime path can be restored without rewriting the
   frozen snapshot;
4. GPU doctor succeeds after the old runtime is quiescent; and
5. unknown GPU outcomes remain quarantined rather than being replayed.

## Manual switch

Perform the switch during a maintenance window. Move the entire old runtime
tree aside and move the verified restored tree into the exact stable path named
by the existing deployment configuration. The two moves, resource fencing, and
service restart are operator-owned actions; this code intentionally does not
perform them.

After the switch, open all restored databases read-only once more, verify the
restore manifest, then run status and doctor before allowing resume. Resume must
use the same immutable `ResolvedRunSnapshot`; aliases are not re-resolved.
Campaign recovery must also reconcile its child intent, budget reservations,
lease fencing epoch, and staged baseline pointer before any external action.

## Rollback

If any post-switch verification fails, stop the trusted host, quarantine the
resource, and switch the complete previous runtime tree back as one unit. Never
roll back only one database or copy selected rows/artifacts between roots.
Preserve the failed restored tree and logs for audit.
