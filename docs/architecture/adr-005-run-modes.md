# ADR-005: Benchmark and discovery modes

## Status

Accepted

## Context

Model comparisons are biased if later arms see newer candidates or a moving
baseline.  Continuous discovery, conversely, needs feedback and staged lineage
advancement.

## Decision

Support two explicit modes:

- `BENCHMARK` freezes namespace, baseline, prompt protocol, feedback snapshot
  and History cutoff.  It never advances lineage.
- `DISCOVERY` may consume bounded feedback and advance a campaign baseline only
  between child runs.

One run always has one immutable proposer profile and baseline.  Provider or
Harness failures never trigger silent model fallback.

## Trade-offs

- Benchmark evidence cannot be ranked by a single best candidate; reports use
  repeated protocol, cost and funnel metrics.
- Campaign model selection occurs only at child-run boundaries.
