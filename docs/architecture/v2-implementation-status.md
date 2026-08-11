# V2 implementation and activation status

This document separates implemented software contracts from evidence that can
only be produced on the intended MetaX C500 deployment. A profile or adapter
being present in the trusted registry does not by itself make it eligible for
autonomous evaluation or promotion.

| Release | Software status | Activation or evidence status |
| --- | --- | --- |
| R0 | Accepted ADRs and legacy golden compatibility are implemented. | Existing deployment and accepted legacy artifact remain unchanged. |
| R1 | The current protocol adds explicit shadow and hidden holdout quick-case roles. Strict CURRENT noise collection/reporting freezes the namespace, baseline, environment and replicate identity, while legacy hash reports remain read-only. A bounded profiling doctor is implemented. | Ten independent C500 full remeasurements, the 99% null-distribution decision, and the real profiler capability probe have not been performed by the test suite. Current-protocol autonomous promotion remains closed until those checks pass. |
| R2 | Profile references, namespace identity, tagged artifacts, experiment UID, explicit baselines, scientific/private CAS, History V3, Controller V3, and coordinated V2-to-V3 migration are implemented. | Production migration requires an offline terminal V2 pair and a verified whole-unit checkpoint. |
| R3 | The existing Fused MoE, Triton Python, C500, protocol, promotion, and OpenCode paths have trusted component wrappers. Production Run and evaluator paths resolve those exact registered components before start, resume and every GPU stage. Proposal V2 and V1 conversion, frozen snapshots, strict evaluator identity echo, idempotent cross-store reconciliation, graft-resistant UID reproof, and an environment-bound manual deployment pin are implemented. | The legacy compatibility path remains scientifically non-comparable to resolved V2 execution evidence. A CURRENT deployment pin is issued only from a complete primary/confirmation proof and is revalidated against CAS, Git, and the resolved runtime before every ordinary Run. |
| R4 | Trusted OpenCode, Direct API, and Pi proposer contracts are implemented. Benchmark Campaigns freeze a trusted cohort snapshot, execute deterministic interleaved child Runs, and derive aggregate reports from Campaign, Controller and History ledgers rather than caller-supplied observations. | Only the two reviewed OpenCode profiles are executable. Direct API and Pi adapters remain inert planning/request contracts; no network credential or live-provider execution is enabled here. |
| R5 | TileLang and MACA CUDA probe/contract skeletons, Ragged Prefill CPU oracle, and descriptors for the planned operator sequence are implemented fail-closed. | Every R5 language/operator profile is `INACTIVE`. No real TileLang/MACA compiler, device, or two-full-run activation evidence has been produced. |
| R6 | Durable five-axis Campaign budgets, per-action deadline/fencing checks, environment-bound staged lineage, trusted GPU-doctor quarantine clearance, OJ nomination/export/manual feedback, evidence-collected soak gates, three-database checkpoints, inactive-root recovery, a shared Admin/Campaign maintenance fence, and bounded advisory profiling collection are implemented. OJ keeps the Campaign-local revision row separate from the imported scientific baseline revision, and profiling binds its subject to the exact Campaign child and Controller run. | No 24/72/168-hour qualification soak has been run. Profiling remains gated by a continuously observed soak ledger with qualifying workload activity, and the release-candidate profiler image digest must be replaced by the built MetaX image before activation. |

## Non-negotiable runtime gates

- A Run freezes its namespace, proposer, baseline, execution environment,
  workflow, History cutoff, and bounded budget before an external action.
- Holdout measurements remain complete host-side scientific evidence but are
  removed from proposer feedback and per-case tuning summaries.
- Candidate bytes, scientific condition, one execution attempt, and baseline
  lineage use separate identities.
- A Campaign may advance only its staged baseline and only from trusted primary
  plus confirmation evidence and a verified checkpoint. Deployment adoption is
  a separate manual administrator action.
- Manual deployment adoption publishes one immutable `deployment-baseline.json`
  beside the three durable runtime contexts. Its namespace, parent baseline,
  primary/confirmation relation, execution environment, authoritative bundle
  CAS, Git commit, and `kernel.py` bytes must continue to agree at Run start.
- OJ feedback and profiling evidence are advisory. Neither may change local
  promotion, Campaign lineage, Git, or a deployment pin.
- Benchmark execution consumes its ledger-derived frozen feedback snapshot;
  its public report surface cannot accept caller-supplied observations.
- Unknown GPU outcomes and hard failures are quarantined and are not replayed
  automatically. Clearing a quarantine requires a fresh successful trusted
  doctor result bound to the exact resource and fencing epoch.
- Production Campaign commands use the single conventional database path
  `<runtime-root>/campaign/campaign.sqlite3`; path aliases and custom live
  database locations fail closed so deployment administration can find every
  active Campaign.
- Campaign lifecycle activation and every deployment publisher share the
  canonical `<runtime-root>/campaign/maintenance.lock` fence. Admin sync,
  update, adoption and post-update publication recheck the Campaign ledger
  while that fence is held, so a Campaign cannot start inside the
  check-to-publish window.
- Soak time accrues only across contiguous, at-most-five-minute trusted
  observations of Campaign, Controller, Git, and Docker state. Empty ledgers,
  unavailable sources, invariant changes, and missing stage-specific workload
  evidence cannot qualify a release.
- A restored or migration checkpoint is switched and rolled back only as a
  complete runtime unit; a one-database rollback is unsupported.

## Activation checklist

Before enabling the current C500 protocol for autonomous discovery:

1. run and record at least ten independent same-hash full remeasurements in one
   resolved execution environment;
2. confirm the read-only noise report's 99th percentile remains below the 1%
   promotion threshold, otherwise keep automatic promotion paused;
3. establish a manually reviewed current-namespace baseline with primary and
   confirmation evidence bound to the resolved execution environment;
4. run the profiler doctor in its pinned environment and retain explicit
   `UNAVAILABLE + reason` values for missing capabilities; and
5. complete the applicable Campaign soak from zero after every invariant
   violation or implementation change that affects the gate.

## Repository verification snapshot

The final host-only verification run for this implementation completed 450
unit and integration tests with 80.51% branch-aware coverage. `compileall` and
the Git whitespace check also passed. These checks use mocks and isolated
SQLite/runtime fixtures; they do not claim MetaX C500 execution, provider/OJ
network access, or elapsed 24/72/168-hour soak evidence.
