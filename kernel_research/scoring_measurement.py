"""Canonical role binding for scoring-baseline device-event evidence."""

from __future__ import annotations

import math
import statistics
from typing import Mapping, Sequence

from .device_timing import device_event_protocol_snapshot
from .platform.canonical import canonical_sha256


SCORING_BASELINE_MEASUREMENT_SCHEMA_VERSION = 2
MIN_SCORING_REFERENCE_MATCHED_RATIO = 0.99
MAX_COMPILED_TO_EAGER_RATIO = 1.01
SCORING_BASELINE_QUALIFICATION_RUNS = 10
SCORING_BASELINE_STABILITY_PROTOCOL_ID = "xpuoj-th0-aggregate-stability-v2"
MAX_CASE_RELATIVE_MAD = 0.01
MAX_AGGREGATE_SCORE_MAD_POINTS = 0.125
MAX_AGGREGATE_PARITY_BIAS_POINTS = 0.05
MAX_AGGREGATE_PROBE_DEVIATION_POINTS = 0.50


def scoring_baseline_stability_contract_snapshot() -> dict[str, object]:
    """Freeze the score-aligned cross-probe qualification rule."""

    snapshot: dict[str, object] = {
        "protocol_id": SCORING_BASELINE_STABILITY_PROTOCOL_ID,
        "probe_count": SCORING_BASELINE_QUALIFICATION_RUNS,
        "case_statistic": "relative-mad-of-probe-anchor-medians",
        "maximum_case_relative_mad": MAX_CASE_RELATIVE_MAD,
        "aggregate_formula": "arithmetic-mean-of-th0-self-scores",
        "aggregate_case_weighting": "equal",
        "aggregate_parity_score": 50.0,
        "aggregate_statistic": "median-and-mad-of-probe-self-scores",
        "maximum_aggregate_score_mad_points": (
            MAX_AGGREGATE_SCORE_MAD_POINTS
        ),
        "maximum_aggregate_parity_bias_points": (
            MAX_AGGREGATE_PARITY_BIAS_POINTS
        ),
        "maximum_aggregate_probe_deviation_points": (
            MAX_AGGREGATE_PROBE_DEVIATION_POINTS
        ),
    }
    return {**snapshot, "digest": canonical_sha256(snapshot)}


def scoring_baseline_measurement_contract_snapshot() -> dict[str, object]:
    """Freeze which callable pair supplies each scoring-baseline claim."""

    snapshot: dict[str, object] = {
        "schema_version": SCORING_BASELINE_MEASUREMENT_SCHEMA_VERSION,
        "anchor_pairing": "compiled-reference-vs-compiled-reference",
        "anchor_statistic": "median-of-12-self-paired-block-latencies",
        "anchor_channel_count": 2,
        "anchor_blocks_per_channel": 6,
        "phase_order": "all-case-anchors-before-performance-proof",
        "case_memory_policy": "regenerate-and-release-each-case",
        "performance_pairing": "compiled-reference-vs-eager-reference",
        "performance_statistic": "ratio-of-channel-medians",
        "performance_role": "qualification-only-non-regression-proof",
        "minimum_correctness_ratio": MIN_SCORING_REFERENCE_MATCHED_RATIO,
        "maximum_compiled_to_eager_ratio": MAX_COMPILED_TO_EAGER_RATIO,
        "qualification_stability": scoring_baseline_stability_contract_snapshot(),
        "timing_protocol": device_event_protocol_snapshot(),
    }
    return {**snapshot, "digest": canonical_sha256(snapshot)}


def _number(value: object, *, field: str, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (number <= 0.0 if positive else number < 0.0):
        requirement = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be finite and {requirement}")
    return number


def validate_device_event_measurement(
    value: object, *, field: str
) -> tuple[float, float, float]:
    """Recompute both channel medians and their combined block median."""

    expected_protocol = device_event_protocol_snapshot()
    expected_fields = {
        "protocol_id",
        "warmup_iterations",
        "measurement_rounds",
        "launches_per_round",
        "candidate_median_ms",
        "candidate_mad_ms",
        "incumbent_median_ms",
        "incumbent_mad_ms",
        "rounds",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError(f"{field} fields are not exact")
    for key in (
        "protocol_id",
        "warmup_iterations",
        "measurement_rounds",
        "launches_per_round",
    ):
        if value.get(key) != expected_protocol[key]:
            raise ValueError(f"{field} changed the device-event protocol")
    rounds = value.get("rounds")
    if not isinstance(rounds, Sequence) or isinstance(rounds, (str, bytes)):
        raise ValueError(f"{field}.rounds must be an array")
    expected_orders = tuple(expected_protocol["round_order"])
    if len(rounds) != len(expected_orders):
        raise ValueError(f"{field} has the wrong round count")
    candidate_values: list[float] = []
    incumbent_values: list[float] = []
    for index, (round_value, expected_order) in enumerate(
        zip(rounds, expected_orders, strict=True)
    ):
        if not isinstance(round_value, Mapping) or set(round_value) != {
            "round_index",
            "order",
            "candidate_latency_ms",
            "incumbent_latency_ms",
        }:
            raise ValueError(f"{field} round fields are not exact")
        if (
            round_value.get("round_index") != index
            or round_value.get("order") != expected_order
        ):
            raise ValueError(f"{field} round identity changed")
        candidate_values.append(
            _number(
                round_value.get("candidate_latency_ms"),
                field=f"{field}.candidate_latency_ms",
                positive=True,
            )
        )
        incumbent_values.append(
            _number(
                round_value.get("incumbent_latency_ms"),
                field=f"{field}.incumbent_latency_ms",
                positive=True,
            )
        )
    candidate_median = statistics.median(candidate_values)
    incumbent_median = statistics.median(incumbent_values)
    candidate_mad = statistics.median(
        abs(item - candidate_median) for item in candidate_values
    )
    incumbent_mad = statistics.median(
        abs(item - incumbent_median) for item in incumbent_values
    )
    derived = {
        "candidate_median_ms": (candidate_median, True),
        "candidate_mad_ms": (candidate_mad, False),
        "incumbent_median_ms": (incumbent_median, True),
        "incumbent_mad_ms": (incumbent_mad, False),
    }
    for key, (expected, positive) in derived.items():
        actual = _number(value.get(key), field=f"{field}.{key}", positive=positive)
        if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError(f"{field} derived statistics changed")
    combined_median = statistics.median(candidate_values + incumbent_values)
    return candidate_median, incumbent_median, combined_median


__all__ = [
    "MAX_AGGREGATE_PARITY_BIAS_POINTS",
    "MAX_AGGREGATE_PROBE_DEVIATION_POINTS",
    "MAX_AGGREGATE_SCORE_MAD_POINTS",
    "MAX_CASE_RELATIVE_MAD",
    "MAX_COMPILED_TO_EAGER_RATIO",
    "MIN_SCORING_REFERENCE_MATCHED_RATIO",
    "SCORING_BASELINE_QUALIFICATION_RUNS",
    "SCORING_BASELINE_MEASUREMENT_SCHEMA_VERSION",
    "SCORING_BASELINE_STABILITY_PROTOCOL_ID",
    "scoring_baseline_measurement_contract_snapshot",
    "scoring_baseline_stability_contract_snapshot",
    "validate_device_event_measurement",
]
