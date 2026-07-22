"""Deterministic latency summaries and conservative promotion decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence


MIN_AGGREGATE_SPEEDUP = 1.01
MAX_CASE_REGRESSION = 0.03

LatencyValues = Mapping[str, float] | Sequence[float]


def _finite_positive(value: object, *, label: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be numeric") from exc
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return converted


def percentiles(samples: Sequence[float]) -> dict[str, float]:
    """Return linearly interpolated p20/p50/p80 latency statistics."""

    values = [
        _finite_positive(sample, label=f"samples[{index}]")
        for index, sample in enumerate(samples)
    ]
    if not values:
        raise ValueError("samples must not be empty")
    values.sort()

    def interpolate(percentile: float) -> float:
        rank = (len(values) - 1) * percentile / 100.0
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return values[lower]
        fraction = rank - lower
        return values[lower] + (values[upper] - values[lower]) * fraction

    return {
        "p20": interpolate(20.0),
        "p50": interpolate(50.0),
        "p80": interpolate(80.0),
    }


def latency_percentiles(samples: Sequence[float]) -> dict[str, float]:
    """Return percentile keys named for serialized microsecond results."""

    summary = percentiles(samples)
    return {f"{name}_us": value for name, value in summary.items()}


def _paired_latencies(
    baseline: LatencyValues,
    candidate: LatencyValues,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    if isinstance(baseline, Mapping) or isinstance(candidate, Mapping):
        if not isinstance(baseline, Mapping) or not isinstance(candidate, Mapping):
            raise TypeError("baseline and candidate must both be mappings or sequences")
        if not baseline:
            raise ValueError("latency mappings must not be empty")
        baseline_keys = {str(key) for key in baseline}
        candidate_keys = {str(key) for key in candidate}
        if baseline_keys != candidate_keys:
            missing = sorted(baseline_keys - candidate_keys)
            extra = sorted(candidate_keys - baseline_keys)
            raise ValueError(
                f"candidate case keys differ from baseline; missing={missing}, extra={extra}"
            )
        # String keys are the persisted case identity used throughout the
        # framework.  Reject ambiguous mappings such as {1: ..., "1": ...}.
        if len(baseline_keys) != len(baseline) or len(candidate_keys) != len(candidate):
            raise ValueError("case keys must have unique string representations")
        baseline_by_name = {str(key): value for key, value in baseline.items()}
        candidate_by_name = {str(key): value for key, value in candidate.items()}
        names = tuple(sorted(baseline_keys))
        baseline_values = tuple(
            _finite_positive(baseline_by_name[name], label=f"baseline[{name!r}]")
            for name in names
        )
        candidate_values = tuple(
            _finite_positive(candidate_by_name[name], label=f"candidate[{name!r}]")
            for name in names
        )
        return names, baseline_values, candidate_values

    try:
        baseline_sequence = tuple(baseline)
        candidate_sequence = tuple(candidate)
    except TypeError as exc:
        raise TypeError("baseline and candidate must be mappings or sequences") from exc
    if not baseline_sequence:
        raise ValueError("latency sequences must not be empty")
    if len(baseline_sequence) != len(candidate_sequence):
        raise ValueError("baseline and candidate must have the same number of cases")
    names = tuple(str(index) for index in range(len(baseline_sequence)))
    baseline_values = tuple(
        _finite_positive(value, label=f"baseline[{index}]")
        for index, value in enumerate(baseline_sequence)
    )
    candidate_values = tuple(
        _finite_positive(value, label=f"candidate[{index}]")
        for index, value in enumerate(candidate_sequence)
    )
    return names, baseline_values, candidate_values


def case_speedups(
    baseline: LatencyValues,
    candidate: LatencyValues,
) -> dict[str, float]:
    """Return normalized speedup ``baseline_latency / candidate_latency``."""

    names, baseline_values, candidate_values = _paired_latencies(baseline, candidate)
    return {
        name: baseline_value / candidate_value
        for name, baseline_value, candidate_value in zip(
            names, baseline_values, candidate_values
        )
    }


def geometric_mean_speedup(
    baseline: LatencyValues,
    candidate: LatencyValues,
) -> float:
    """Compute the equal-weight geometric mean of per-case speedups."""

    speedups = case_speedups(baseline, candidate)
    return math.exp(math.fsum(math.log(value) for value in speedups.values()) / len(speedups))


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    """Result of primary and optional confirmation comparisons."""

    promoted: bool
    needs_confirmation: bool
    aggregate_speedup: float
    confirmation_speedup: float | None
    worst_case_regression: float
    confirmation_worst_case_regression: float | None
    per_case_speedups: Mapping[str, float]
    confirmation_case_speedups: Mapping[str, float] | None
    reason: str

    @property
    def eligible(self) -> bool:
        return self.promoted

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _run_metrics(
    baseline: LatencyValues,
    candidate: LatencyValues,
    *,
    min_speedup: float,
    max_case_regression: float,
) -> tuple[bool, float, float, dict[str, float]]:
    names, baseline_values, candidate_values = _paired_latencies(baseline, candidate)
    speedups = {
        name: baseline_value / candidate_value
        for name, baseline_value, candidate_value in zip(
            names, baseline_values, candidate_values
        )
    }
    aggregate = math.exp(
        math.fsum(math.log(value) for value in speedups.values()) / len(speedups)
    )
    regressions = [
        max(0.0, candidate_value / baseline_value - 1.0)
        for baseline_value, candidate_value in zip(baseline_values, candidate_values)
    ]
    worst_regression = max(regressions)
    qualifies = (
        aggregate + 1.0e-12 >= min_speedup
        and worst_regression <= max_case_regression + 1.0e-12
    )
    return qualifies, aggregate, worst_regression, speedups


def evaluate_promotion(
    baseline: LatencyValues,
    candidate: LatencyValues,
    confirmation: LatencyValues | None = None,
    *,
    confirmation_baseline: LatencyValues | None = None,
    min_speedup: float = MIN_AGGREGATE_SPEEDUP,
    max_case_regression: float = MAX_CASE_REGRESSION,
) -> PromotionDecision:
    """Apply aggregate, per-case, and second-run promotion gates.

    A primary run that clears both performance thresholds requests a second
    measurement but is not yet promotable. Promotion occurs only if that
    independent confirmation also clears the same thresholds. When supplied,
    ``confirmation_baseline`` is the current-best measurement interleaved with
    the confirmation candidate, so each run is normalized against its own
    contemporaneous baseline.
    """

    if not math.isfinite(min_speedup) or min_speedup <= 0.0:
        raise ValueError("min_speedup must be finite and positive")
    if (
        not math.isfinite(max_case_regression)
        or max_case_regression < 0.0
    ):
        raise ValueError("max_case_regression must be finite and non-negative")

    qualifies, aggregate, worst, speedups = _run_metrics(
        baseline,
        candidate,
        min_speedup=min_speedup,
        max_case_regression=max_case_regression,
    )
    if not qualifies:
        reason = (
            "aggregate_speedup_below_threshold"
            if aggregate + 1.0e-12 < min_speedup
            else "per_case_regression_exceeded"
        )
        return PromotionDecision(
            promoted=False,
            needs_confirmation=False,
            aggregate_speedup=aggregate,
            confirmation_speedup=None,
            worst_case_regression=worst,
            confirmation_worst_case_regression=None,
            per_case_speedups=speedups,
            confirmation_case_speedups=None,
            reason=reason,
        )

    if confirmation is None:
        return PromotionDecision(
            promoted=False,
            needs_confirmation=True,
            aggregate_speedup=aggregate,
            confirmation_speedup=None,
            worst_case_regression=worst,
            confirmation_worst_case_regression=None,
            per_case_speedups=speedups,
            confirmation_case_speedups=None,
            reason="confirmation_required",
        )

    confirmed, confirmation_aggregate, confirmation_worst, confirmation_speedups = (
        _run_metrics(
            baseline if confirmation_baseline is None else confirmation_baseline,
            confirmation,
            min_speedup=min_speedup,
            max_case_regression=max_case_regression,
        )
    )
    return PromotionDecision(
        promoted=confirmed,
        needs_confirmation=False,
        aggregate_speedup=aggregate,
        confirmation_speedup=confirmation_aggregate,
        worst_case_regression=worst,
        confirmation_worst_case_regression=confirmation_worst,
        per_case_speedups=speedups,
        confirmation_case_speedups=confirmation_speedups,
        reason="promoted" if confirmed else "confirmation_failed",
    )


__all__ = [
    "MAX_CASE_REGRESSION",
    "MIN_AGGREGATE_SPEEDUP",
    "PromotionDecision",
    "case_speedups",
    "evaluate_promotion",
    "geometric_mean_speedup",
    "latency_percentiles",
    "percentiles",
]
