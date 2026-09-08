"""Trusted-host aggregation for independent scoring-baseline probes."""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .device_timing import device_event_protocol_snapshot
from .objective_scoring import ScoringBaselineDescriptor
from .platform.canonical import (
    canonical_json_text,
    canonical_sha256,
    require_sha256_digest,
)


SCORING_BASELINE_QUALIFICATION_RUNS = 10
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
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
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
        )
        if canonical_json_text(qualification.to_dict()) != canonical_json_text(value):
            raise ValueError("scoring baseline qualification digest or fields differ")
        return qualification


def aggregate_scoring_baseline_probes(
    probes: Sequence[Mapping[str, Any]],
    *,
    environment_digest: str,
    evaluator_profile_digest: str,
) -> ScoringBaselineQualification:
    """Aggregate ten independent, exact-identity evaluator probes."""

    if len(probes) != SCORING_BASELINE_QUALIFICATION_RUNS:
        raise ValueError("exactly ten scoring baseline probes are required")
    expected_timing = device_event_protocol_snapshot()
    identity: tuple[object, ...] | None = None
    case_samples: dict[str, list[float]] = {}
    probe_digests: list[str] = []
    reference_digest = ""
    compiler_config: Mapping[str, object] = {}
    for index, probe in enumerate(probes):
        if probe.get("status") != "QUALIFIED":
            raise ValueError(f"scoring baseline probe {index} is not qualified")
        if probe.get("schema_version") != 1 or probe.get("command") != "score-baseline-probe":
            raise ValueError(f"scoring baseline probe {index} has invalid identity")
        if probe.get("timing_protocol") != expected_timing:
            raise ValueError(f"scoring baseline probe {index} changed timing protocol")
        cases = probe.get("cases")
        if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
            raise ValueError(f"scoring baseline probe {index} has invalid cases")
        current_identity = (
            probe.get("protocol_id"),
            probe.get("reference_source_sha256"),
            canonical_sha256(probe.get("compiler_config")),
            probe.get("environment_snapshot_digest"),
            tuple(case.get("case_id") for case in cases if isinstance(case, Mapping)),
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
        for case in cases:
            if not isinstance(case, Mapping):
                raise ValueError("probe case must be an object")
            case_id = case.get("case_id")
            measurement = case.get("measurement")
            if not isinstance(case_id, str) or not isinstance(measurement, Mapping):
                raise ValueError("probe case identity is invalid")
            latency = measurement.get("candidate_median_ms")
            if (
                isinstance(latency, bool)
                or not isinstance(latency, (int, float))
                or not math.isfinite(float(latency))
                or float(latency) <= 0.0
            ):
                raise ValueError("probe candidate latency is invalid")
            case_samples.setdefault(case_id, []).append(float(latency))
    if any(len(values) != SCORING_BASELINE_QUALIFICATION_RUNS for values in case_samples.values()):
        raise ValueError("probe case set is incomplete")

    baselines: dict[str, float] = {}
    envelopes: dict[str, dict[str, float]] = {}
    qualified = True
    reason = "qualified"
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
        if relative_mad > MAX_RELATIVE_MAD:
            qualified = False
            reason = "relative_mad_exceeded"
    descriptor = ScoringBaselineDescriptor(
        environment_digest=environment_digest,
        evaluator_profile_digest=evaluator_profile_digest,
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
