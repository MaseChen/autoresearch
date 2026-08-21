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
| R3 | The existing Fused MoE, Triton Python, C500, protocol, promotion, and OpenCode paths have trusted component wrappers. Production Run and evaluator paths resolve those exact registered components before start, resume and every GPU stage. Proposal V2 and V1 conversion, frozen snapshots, strict evaluator identity echo, idempotent cross-store reconciliation, graft-resistant UID reproof, an explicit legacy-adoption requalification bridge, a trusted CURRENT baseline bootstrap, and an environment-bound manual deployment pin are implemented. | Migrated `LEGACY_UNKNOWN` evidence is never relabelled. The requalification bridge must first reproduce the deployed baseline under a resolved environment. The CURRENT bootstrap then repeats the immutable deployed improvement chain under the CURRENT protocol and produces fresh namespace-local primary and confirmation evidence. A CURRENT deployment pin is issued only by manual adoption from that complete proof and is revalidated against CAS, Git, and the resolved runtime before every ordinary Run. |
| R4 | Trusted OpenCode, Direct API, and Pi proposer contracts are implemented. Benchmark Campaigns freeze a trusted cohort snapshot, execute deterministic interleaved child Runs, and derive aggregate reports from Campaign, Controller and History ledgers rather than caller-supplied observations. | Only the two reviewed OpenCode profiles are executable. Direct API and Pi adapters remain inert planning/request contracts; no network credential or live-provider execution is enabled here. |
| R5 | TileLang and MACA CUDA probe/contract skeletons, Ragged Prefill CPU oracle, and descriptors for the planned operator sequence are implemented fail-closed. | Every R5 language/operator profile is `INACTIVE`. No real TileLang/MACA compiler, device, or two-full-run activation evidence has been produced. |
| R6 | Durable five-axis Campaign budgets, deadline/fencing checks, environment-bound staged lineage, trusted GPU-doctor quarantine clearance, OJ nomination/export/manual feedback, evidence-collected soak gates, three-database checkpoints, recovery, the Admin/Campaign maintenance fence, and bounded advisory profiling are implemented. ADR-009 adds a separate offline-built MetaX profiler image, split build/activation identities, durable worker outcome protocol, V2 toolchain/trace evidence, a no-child pre-soak canary, immutable per-recipe launch/host diagnostics, and terminalization paths that never replay an action. A5 persists bounded phase diagnostics; A6 freezes a 24 GiB hard limit plus host memory admission; A7 preserves a noexec general tmpfs while granting only a bounded nested Triton cache execution; A8 normalizes the copied Python tree before the non-root build probe and freezes that filesystem contract into build identity. | A5's OOM action was safely abandoned. A6 was built, activated and run once; its noexec-cache known failure was settled and later terminalized without GPU/replay. A7's native build failed safely because classic Docker preserved hostile 0600/0700 checkout modes; it produced no final image or runtime action. A8 is an inactive build submission only: no A8 image, activation or canary exists. The 24/72/168-hour soak has not started and profiling collection remains gated. |

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
- The full deployment commit and evaluator framework commit are independent
  pins. Adoption accepts only one direct `kernel.py`-only Git commit and keeps
  the evaluator framework, cache, and execution-environment binding frozen to
  the exact framework Git tree materialized inside the container.
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
- Profiler image, worker, recipe, toolchain, limit, or activation changes are
  explicit soak invariants. Any change after a gate starts invalidates prior
  elapsed time and restarts qualification from the first stage.
- A restored or migration checkpoint is switched and rolled back only as a
  complete runtime unit; a one-database rollback is unsupported.

## Activation checklist

Before enabling the current C500 protocol for autonomous discovery:

1. establish a manually reviewed current-namespace baseline with primary and
   confirmation evidence bound to the resolved execution environment, using
   the trusted bootstrap defined by ADR-008 rather than relabelling legacy
   evidence;
2. run and record at least ten independent same-hash full remeasurements in one
   resolved execution environment;
3. confirm the read-only noise report's 99th percentile remains below the 1%
   promotion threshold, otherwise keep automatic promotion paused;
4. run the profiler doctor in its pinned environment and retain explicit
   `UNAVAILABLE + reason` values for missing capabilities; and
5. complete the applicable Campaign soak from zero after every invariant
   violation or implementation change that affects the gate.

## Repository verification snapshot

The A8 inactive build-submission verification run completed 519 unit and
integration tests with 80.15% branch-aware coverage. `compileall` and the Git
whitespace check also passed. These checks use mocks and isolated SQLite/runtime
fixtures; they do not claim an A8 image build or MetaX C500 canary, provider/OJ
network access, or elapsed 24/72/168-hour soak evidence.
