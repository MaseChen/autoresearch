"""Trusted-host aggregation for independent scoring-baseline probes."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .device_timing import device_event_protocol_snapshot
from .objective_scoring import ObjectiveScoreReport, ScoringBaselineDescriptor
from .platform.canonical import (
    canonical_json_text,
    canonical_sha256,
    require_sha256_digest,
)
from .scoring_measurement import (
    MAX_AGGREGATE_PARITY_BIAS_POINTS,
    MAX_AGGREGATE_PROBE_DEVIATION_POINTS,
    MAX_AGGREGATE_SCORE_MAD_POINTS,
    MAX_CASE_RELATIVE_MAD,
    MAX_COMPILED_TO_EAGER_RATIO,
    MIN_SCORING_REFERENCE_MATCHED_RATIO,
    SCORING_BASELINE_QUALIFICATION_RUNS,
    scoring_baseline_measurement_contract_snapshot,
    scoring_baseline_stability_contract_snapshot,
    validate_device_event_measurement,
)


# Retained only to verify schema-V1 evidence.  New qualifications use the
# score-aligned stability contract embedded in the measurement identity.
MAX_RELATIVE_MAD = 0.005


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass(frozen=True, slots=True)
class ScoringBaselineQualification:
    descriptor: ScoringBaselineDescriptor
    probe_digests: tuple[str, ...]
    case_envelopes_ms: Mapping[str, Mapping[str, float]]
    qualified: bool
    reason: str
    aggregate_score_envelope: Mapping[str, float] | None = None
    qualification_stability: Mapping[str, object] | None = None
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version not in {1, 2}:
            raise ValueError("unsupported scoring baseline qualification schema")
        if type(self.qualified) is not bool:
            raise TypeError("qualified must be a boolean")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("qualification reason must be a non-empty string")
        if self.qualified != (self.reason == "qualified"):
            raise ValueError("qualification status and reason disagree")
        if len(self.probe_digests) != SCORING_BASELINE_QUALIFICATION_RUNS:
            raise ValueError("qualification requires exactly ten probe digests")
        for digest in self.probe_digests:
            require_sha256_digest(digest, field="probe_digest")
        if set(self.case_envelopes_ms) != set(self.descriptor.case_baseline_ms):
            raise ValueError("qualification and descriptor case sets differ")
        expected_envelope_keys = {"p01", "p50", "p99", "mad", "relative_mad"}
        normalized: dict[str, Mapping[str, float]] = {}
        for case_id, envelope in sorted(self.case_envelopes_ms.items()):
            if not isinstance(case_id, str) or not case_id:
                raise ValueError("qualification case ID must be a non-empty string")
            if (
                not isinstance(envelope, Mapping)
                or set(envelope) != expected_envelope_keys
            ):
                raise ValueError("qualification case envelope fields are not exact")
            values: dict[str, float] = {}
            for key in sorted(expected_envelope_keys):
                raw = envelope[key]
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    raise TypeError("qualification envelope values must be numeric")
                value = float(raw)
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        "qualification envelope values must be finite and non-negative"
                    )
                values[key] = value
            if (
                values["p01"] <= 0.0
                or values["p50"] <= 0.0
                or values["p99"] <= 0.0
                or values["p01"] > values["p50"]
                or values["p50"] > values["p99"]
                or not math.isclose(
                    values["p50"],
                    float(self.descriptor.case_baseline_ms[case_id]),
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
                or not math.isclose(
                    values["relative_mad"],
                    values["mad"] / values["p50"],
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
            ):
                raise ValueError("qualification case envelope is inconsistent")
            normalized[case_id] = MappingProxyType(values)
        object.__setattr__(self, "case_envelopes_ms", MappingProxyType(normalized))
        if self.schema_version == 1:
            if (
                self.aggregate_score_envelope is not None
                or self.qualification_stability is not None
            ):
                raise ValueError("schema-V1 qualification has V2 stability fields")
            return
        expected_stability = scoring_baseline_stability_contract_snapshot()
        if self.qualification_stability != expected_stability:
            raise ValueError("qualification stability contract changed")
        raw_aggregate = self.aggregate_score_envelope
        expected_aggregate_keys = {
            "p01",
            "p50",
            "p99",
            "mad",
            "relative_mad",
            "parity_bias_points",
            "maximum_probe_deviation_points",
        }
        if (
            not isinstance(raw_aggregate, Mapping)
            or set(raw_aggregate) != expected_aggregate_keys
        ):
            raise ValueError("aggregate score envelope fields are not exact")
        aggregate: dict[str, float] = {}
        for key in sorted(expected_aggregate_keys):
            raw = raw_aggregate[key]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError("aggregate score envelope values must be numeric")
            value = float(raw)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "aggregate score envelope values must be finite and non-negative"
                )
            aggregate[key] = value
        if (
            aggregate["p01"] <= 0.0
            or aggregate["p01"] > aggregate["p50"]
            or aggregate["p50"] > aggregate["p99"]
            or not math.isclose(
                aggregate["relative_mad"],
                aggregate["mad"] / aggregate["p50"],
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                aggregate["parity_bias_points"],
                abs(aggregate["p50"] - 50.0),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            )
            or aggregate["maximum_probe_deviation_points"]
            < max(
                abs(aggregate["p01"] - 50.0),
                abs(aggregate["p50"] - 50.0),
                abs(aggregate["p99"] - 50.0),
            )
        ):
            raise ValueError("aggregate score envelope is inconsistent")
        case_stability_exceeded = any(
            envelope["relative_mad"] > MAX_CASE_RELATIVE_MAD
            for envelope in normalized.values()
        )
        if case_stability_exceeded:
            expected_reason = "case_relative_mad_exceeded"
        elif aggregate["mad"] > MAX_AGGREGATE_SCORE_MAD_POINTS:
            expected_reason = "aggregate_score_mad_exceeded"
        elif (
            aggregate["parity_bias_points"]
            > MAX_AGGREGATE_PARITY_BIAS_POINTS
        ):
            expected_reason = "aggregate_score_parity_bias_exceeded"
        elif (
            aggregate["maximum_probe_deviation_points"]
            > MAX_AGGREGATE_PROBE_DEVIATION_POINTS
        ):
            expected_reason = "aggregate_score_probe_deviation_exceeded"
        else:
            expected_reason = "qualified"
        if self.reason != expected_reason:
            raise ValueError("qualification result disagrees with stability evidence")
        object.__setattr__(
            self,
            "aggregate_score_envelope",
            MappingProxyType(aggregate),
        )
        object.__setattr__(
            self,
            "qualification_stability",
            MappingProxyType(dict(expected_stability)),
        )

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "descriptor": self.descriptor.to_dict(),
            "probe_digests": list(self.probe_digests),
            "case_envelopes_ms": {
                case_id: dict(envelope)
                for case_id, envelope in self.case_envelopes_ms.items()
            },
            "qualified": self.qualified,
            "reason": self.reason,
        }
        if self.schema_version == 2:
            value.update(
                {
                    "aggregate_score_envelope": dict(
                        self.aggregate_score_envelope or {}
                    ),
                    "qualification_stability": dict(
                        self.qualification_stability or {}
                    ),
                }
            )
        if include_digest:
            value["digest"] = self.digest
        return value

    @classmethod
    def from_value(
        cls, value: Mapping[str, object]
    ) -> "ScoringBaselineQualification":
        expected_keys = {
            "schema_version",
            "descriptor",
            "probe_digests",
            "case_envelopes_ms",
            "qualified",
            "reason",
            "digest",
        }
        if value.get("schema_version") == 2:
            expected_keys |= {
                "aggregate_score_envelope",
                "qualification_stability",
            }
        if set(value) != expected_keys:
            raise ValueError("scoring baseline qualification fields are not exact")
        descriptor_value = value["descriptor"]
        probe_digests = value["probe_digests"]
        envelopes = value["case_envelopes_ms"]
        if not isinstance(descriptor_value, Mapping):
            raise ValueError("qualification descriptor must be an object")
        if not isinstance(probe_digests, Sequence) or isinstance(
            probe_digests, (str, bytes)
        ):
            raise ValueError("qualification probe_digests must be an array")
        if not isinstance(envelopes, Mapping):
            raise ValueError("qualification case_envelopes_ms must be an object")
        qualification = cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            descriptor=ScoringBaselineDescriptor.from_value(descriptor_value),
            probe_digests=tuple(probe_digests),  # type: ignore[arg-type]
            case_envelopes_ms=envelopes,  # type: ignore[arg-type]
            qualified=value["qualified"],  # type: ignore[arg-type]
            reason=value["reason"],  # type: ignore[arg-type]
            aggregate_score_envelope=value.get(  # type: ignore[arg-type]
                "aggregate_score_envelope"
            ),
            qualification_stability=value.get(  # type: ignore[arg-type]
                "qualification_stability"
            ),
        )
        if canonical_json_text(qualification.to_dict()) != canonical_json_text(value):
            raise ValueError("scoring baseline qualification digest or fields differ")
        return qualification


def aggregate_scoring_baseline_probes(
    probes: Sequence[Mapping[str, Any]],
    *,
    environment_digest: str,
    evaluator_profile_digest: str,
    scoring_framework_git_commit: str,
) -> ScoringBaselineQualification:
    """Aggregate ten independent, exact-identity evaluator probes."""

    if len(probes) != SCORING_BASELINE_QUALIFICATION_RUNS:
        raise ValueError("exactly ten scoring baseline probes are required")
    expected_timing = device_event_protocol_snapshot()
    expected_measurement_contract = (
        scoring_baseline_measurement_contract_snapshot()
    )
    identity: tuple[object, ...] | None = None
    case_samples: dict[str, list[float]] = {}
    probe_case_samples: list[dict[str, float]] = []
    probe_digests: list[str] = []
    reference_digest = ""
    compiler_config: Mapping[str, object] = {}
    for index, probe in enumerate(probes):
        if probe.get("status") != "QUALIFIED":
            raise ValueError(f"scoring baseline probe {index} is not qualified")
        if (
            probe.get("schema_version") != 1
            or probe.get("command") != "score-baseline-probe"
        ):
            raise ValueError(f"scoring baseline probe {index} has invalid identity")
        if probe.get("timing_protocol") != expected_timing:
            raise ValueError(f"scoring baseline probe {index} changed timing protocol")
        if probe.get("measurement_contract") != expected_measurement_contract:
            raise ValueError(
                f"scoring baseline probe {index} changed measurement contract"
            )
        cases = probe.get("cases")
        if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
            raise ValueError(f"scoring baseline probe {index} has invalid cases")
        current_identity = (
            probe.get("protocol_id"),
            probe.get("scoring_framework_git_commit"),
            probe.get("reference_source_sha256"),
            canonical_sha256(probe.get("compiler_config")),
            canonical_sha256(probe.get("measurement_contract")),
            probe.get("environment_snapshot_digest"),
            tuple(case.get("case_id") for case in cases if isinstance(case, Mapping)),
        )
        if probe.get("scoring_framework_git_commit") != (
            scoring_framework_git_commit
        ):
            raise ValueError(
                f"scoring baseline probe {index} changed scoring framework commit"
            )
        if identity is None:
            identity = current_identity
            reference_digest = str(probe["reference_source_sha256"])
            raw_config = probe.get("compiler_config")
            if not isinstance(raw_config, Mapping):
                raise ValueError("compiler_config must be an object")
            compiler_config = dict(raw_config)
        elif current_identity != identity:
            raise ValueError("scoring baseline probe identity drift")
        probe_digests.append(canonical_sha256(probe))
        probe_samples: dict[str, float] = {}
        for case in cases:
            if not isinstance(case, Mapping) or set(case) != {
                "case_id",
                "matched_ratio",
                "eager_matched_ratio",
                "compiler_first_invocation_seconds",
                "compiled_to_eager_ratio",
                "anchor_median_ms",
                "anchor_measurement",
                "performance_measurement",
            }:
                raise ValueError("probe case fields are not exact")
            case_id = case.get("case_id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError("probe case identity is invalid")
            for ratio_name in ("matched_ratio", "eager_matched_ratio"):
                ratio = case.get(ratio_name)
                if (
                    isinstance(ratio, bool)
                    or not isinstance(ratio, (int, float))
                    or not math.isfinite(float(ratio))
                    or not MIN_SCORING_REFERENCE_MATCHED_RATIO
                    <= float(ratio)
                    <= 1.0
                ):
                    raise ValueError("probe correctness proof is invalid")
            first_invocation_seconds = case.get(
                "compiler_first_invocation_seconds"
            )
            if (
                isinstance(first_invocation_seconds, bool)
                or not isinstance(first_invocation_seconds, (int, float))
                or not math.isfinite(float(first_invocation_seconds))
                or float(first_invocation_seconds) <= 0.0
            ):
                raise ValueError("probe compilation proof is invalid")
            _, _, derived_anchor = validate_device_event_measurement(
                case.get("anchor_measurement"),
                field=f"probe[{index}].{case_id}.anchor_measurement",
            )
            performance_candidate, performance_incumbent, _ = (
                validate_device_event_measurement(
                    case.get("performance_measurement"),
                    field=f"probe[{index}].{case_id}.performance_measurement",
                )
            )
            latency = case.get("anchor_median_ms")
            if (
                isinstance(latency, bool)
                or not isinstance(latency, (int, float))
                or not math.isfinite(float(latency))
                or float(latency) <= 0.0
            ):
                raise ValueError("probe candidate latency is invalid")
            if not math.isclose(
                float(latency),
                derived_anchor,
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise ValueError("probe anchor median is inconsistent")
            compiled_to_eager = case.get("compiled_to_eager_ratio")
            if (
                isinstance(compiled_to_eager, bool)
                or not isinstance(compiled_to_eager, (int, float))
                or not math.isclose(
                    float(compiled_to_eager),
                    performance_candidate / performance_incumbent,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                )
                or not 0.0
                < float(compiled_to_eager)
                <= MAX_COMPILED_TO_EAGER_RATIO
            ):
                raise ValueError("probe compiled-to-eager ratio is inconsistent")
            case_samples.setdefault(case_id, []).append(float(latency))
            if case_id in probe_samples:
                raise ValueError("probe case identity is duplicated")
            probe_samples[case_id] = float(latency)
        probe_case_samples.append(probe_samples)
    if any(
        len(values) != SCORING_BASELINE_QUALIFICATION_RUNS
        for values in case_samples.values()
    ):
        raise ValueError("probe case set is incomplete")

    baselines: dict[str, float] = {}
    envelopes: dict[str, dict[str, float]] = {}
    case_stability_exceeded = False
    for case_id, values in sorted(case_samples.items()):
        median = statistics.median(values)
        mad = statistics.median(abs(value - median) for value in values)
        relative_mad = mad / median
        baselines[case_id] = median
        envelopes[case_id] = {
            "p01": _quantile(values, 0.01),
            "p50": median,
            "p99": _quantile(values, 0.99),
            "mad": mad,
            "relative_mad": relative_mad,
        }
        if relative_mad > MAX_CASE_RELATIVE_MAD:
            case_stability_exceeded = True
    aggregate_scores = [
        ObjectiveScoreReport.calculate(samples, baselines).objective_score
        for samples in probe_case_samples
    ]
    aggregate_median = statistics.median(aggregate_scores)
    aggregate_mad = statistics.median(
        abs(score - aggregate_median) for score in aggregate_scores
    )
    aggregate_envelope = {
        "p01": _quantile(aggregate_scores, 0.01),
        "p50": aggregate_median,
        "p99": _quantile(aggregate_scores, 0.99),
        "mad": aggregate_mad,
        "relative_mad": aggregate_mad / aggregate_median,
        "parity_bias_points": abs(aggregate_median - 50.0),
        "maximum_probe_deviation_points": max(
            abs(score - 50.0) for score in aggregate_scores
        ),
    }
    qualified = False
    if case_stability_exceeded:
        reason = "case_relative_mad_exceeded"
    elif aggregate_mad > MAX_AGGREGATE_SCORE_MAD_POINTS:
        reason = "aggregate_score_mad_exceeded"
    elif (
        aggregate_envelope["parity_bias_points"]
        > MAX_AGGREGATE_PARITY_BIAS_POINTS
    ):
        reason = "aggregate_score_parity_bias_exceeded"
    elif (
        aggregate_envelope["maximum_probe_deviation_points"]
        > MAX_AGGREGATE_PROBE_DEVIATION_POINTS
    ):
        reason = "aggregate_score_probe_deviation_exceeded"
    else:
        qualified = True
        reason = "qualified"
    descriptor = ScoringBaselineDescriptor(
        environment_digest=environment_digest,
        evaluator_profile_digest=evaluator_profile_digest,
        measurement_contract_digest=str(
            expected_measurement_contract["digest"]
        ),
        scoring_framework_git_commit=scoring_framework_git_commit,
        reference_source_sha256=reference_digest,
        compiler_backend="inductor",
        compiler_config=compiler_config,
        case_baseline_ms=baselines,
    )
    return ScoringBaselineQualification(
        descriptor=descriptor,
        probe_digests=tuple(probe_digests),
        case_envelopes_ms=envelopes,
        qualified=qualified,
        reason=reason,
        aggregate_score_envelope=aggregate_envelope,
        qualification_stability=scoring_baseline_stability_contract_snapshot(),
    )


def validate_anchor_drift(
    qualification: ScoringBaselineQualification,
    observed_case_ms: Mapping[str, float],
) -> None:
    """Reject an evaluator whose compiled anchor left its frozen envelope."""

    if not qualification.qualified:
        raise ValueError("scoring baseline qualification is not active")
    if set(observed_case_ms) != set(qualification.case_envelopes_ms):
        raise ValueError("scoring anchor case set changed")
    for case_id, observed in observed_case_ms.items():
        value = float(observed)
        envelope = qualification.case_envelopes_ms[case_id]
        if (
            not math.isfinite(value)
            or value <= 0.0
            or value < envelope["p01"]
            or value > envelope["p99"]
        ):
            raise ValueError(f"SCORING_ENVIRONMENT_DRIFT:{case_id}")


__all__ = [
    "MAX_RELATIVE_MAD",
    "SCORING_BASELINE_QUALIFICATION_RUNS",
    "ScoringBaselineQualification",
    "aggregate_scoring_baseline_probes",
    "validate_anchor_drift",
]
