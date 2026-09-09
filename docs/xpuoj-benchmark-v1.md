# XPU-OJ four-case benchmark protocol V1

This protocol exists only for a bounded cross-system comparison.  It does not
replace the formal CURRENT Deployment baseline and it has no adoption or
Campaign-lineage authority.

## Frozen identity

- Protocol: `fused-moe-c500-xpuoj-four-case-v1`
- External seed: accepted XPU-OJ submission `#141440`, score `84`
- Source SHA-256:
  `fd7fba3a4a16275af6f3367e5d0bf9a1d717066d7c93434ed291842254a2bf7f`
- Result receipt SHA-256:
  `2875e8235f445c0a95da66b73f292f1c83ba9d11176e9e6186df4facdb09af1d`
- Required local compatibility proof candidate SHA-256:
  `423c0bbae93fea33dbe1ee5695677df8acfc595c5c637949676b04e49d5a8626`

The evaluator runs only these existing C500 cases:

1. `full_decode_gate_up`
2. `full_prefill_gate_up`
3. `full_decode_down`
4. `full_prefill_down`

The trusted stage sequence is:

```text
PROPOSE -> POLICY -> FULL_PRIMARY -> CONFIRMATION
```

Every full evaluation measures the candidate and exact external seed in the
same interleaved local C500 execution.  The external score and timings are
provenance and proposer context; they are not substituted for local timing.

## Trusted commands

Registration accepts no caller-selected profile, protocol, cases, image,
device, or timeout:

```bash
python -m kernel_research.autorun register-xpuoj-baseline \
  --config "$CONFIG" \
  --candidate "$EXACT_SOURCE" \
  --receipt "$EXACT_RECEIPT" \
  --proof-experiment-id 294
```

The returned History row is a non-promotable benchmark seed with
`promotion_authority=false`.  Registration is idempotent for the frozen
source and receipt.

Start one run with a unique ID:

```bash
python -m kernel_research.autorun start-xpuoj-benchmark \
  --config "$CONFIG" \
  --run-id "$RUN_ID" \
  --baseline-experiment-id "$XPUOJ_BASELINE_EXPERIMENT_ID"
```

The Controller config still owns the six-hour ceiling, candidate limit,
failure limit, model, images, resources, and stop-after-promotion behavior.
Never repeat `start-xpuoj-benchmark` for an existing Run ID.  Use `status` or
`resume` only under the existing trusted outcome and fencing rules.

## Acceptance boundary

Before execution, verify the deployed commit, clean worktree, formal
Deployment pin, History proof, resolved environment, absence of active Runs or
Campaigns, and absence of blocking/quarantined leases.  After execution,
archive the Controller Run, History chain, candidate bundles, exact parent
diffs, resource cleanup, and a schema V2 checkpoint.
