"""Versioned XPU-OJ-aligned objective scoring without promotion authority.

The first revision deliberately uses a zero hardware anchor.  It is therefore
named a proxy rather than a SOL score.  Correctness, the scoring baseline, and
the deployment incumbent are separate identities: this module only computes
an objective score and a noise-aware decision from already trusted evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import re
from types import MappingProxyType
from typing import Mapping, Sequence

from .platform.canonical import (
    canonical_json_text,
    canonical_sha256,
    require_sha256_digest,
)
from .scoring import case_speedups, geometric_mean_speedup


OBJECTIVE_SCORE_SCHEMA_VERSION = 1
XPUOJ_TH0_PROXY_PROTOCOL_ID = "xpuoj-th0-proxy-v1"
XPUOJ_TH0_PROXY_MODE = "PROXY_CURRENT_MACA"
MIN_VISIBLE_SCORE_DELTA = 0.25
MAX_QUALIFIED_NOISE_THRESHOLD = 0.50
MAX_CASE_REGRESSION = 0.03


def _finite(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _finite_positive(value: object, *, field: str) -> float:
    number = _finite(value, field=field)
    if number <= 0.0:
        raise ValueError(f"{field} must be positive")
    return number


def _finite_nonnegative(value: object, *, field: str) -> float:
    number = _finite(value, field=field)
    if number < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return number


def _case_names(*mappings: Mapping[str, object]) -> tuple[str, ...]:
    if not mappings or not mappings[0]:
        raise ValueError("case mappings must not be empty")
    if any(not isinstance(key, str) for key in mappings[0]):
        raise TypeError("case IDs must be strings")
    expected = set(mappings[0])
    for mapping in mappings[1:]:
        if any(not isinstance(key, str) for key in mapping):
            raise TypeError("case IDs must be strings")
        names = set(mapping)
        if names != expected:
            raise ValueError("case mappings must contain identical case IDs")
    return tuple(sorted(expected))


def _canonical_copy(value: object, *, field: str) -> object:
    try:
        return json.loads(canonical_json_text(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be canonical JSON data") from exc


def anchored_case_score(
    candidate_latency_ms: object,
    scoring_baseline_latency_ms: object,
    hardware_anchor_latency_ms: object = 0.0,
) -> float:
    """Return the continuous anchored score for one correct case.

    ``50`` represents parity with the scoring baseline and ``100`` represents
    the hardware anchor.  Values are intentionally not rounded or clipped.
    """

    candidate = _finite_positive(candidate_latency_ms, field="candidate latency")
    baseline = _finite_positive(
        scoring_baseline_latency_ms, field="scoring baseline latency"
    )
    anchor = _finite_nonnegative(
        hardware_anchor_latency_ms, field="hardware anchor latency"
    )
    if baseline <= anchor:
        raise ValueError("scoring baseline latency must exceed hardware anchor")
    denominator = 1.0 + (candidate - anchor) / (baseline - anchor)
    if denominator <= 0.0:
        raise ValueError("anchored score denominator must be positive")
    return 100.0 / denominator


@dataclass(frozen=True, slots=True)
class CaseObjectiveScore:
    case_id: str
    candidate_latency_ms: float
    scoring_baseline_latency_ms: float
    hardware_anchor_latency_ms: float
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise ValueError("case_id must be a non-empty string")
        candidate = _finite_positive(
            self.candidate_latency_ms, field="candidate_latency_ms"
        )
        baseline = _finite_positive(
            self.scoring_baseline_latency_ms,
            field="scoring_baseline_latency_ms",
        )
        anchor = _finite_nonnegative(
            self.hardware_anchor_latency_ms,
            field="hardware_anchor_latency_ms",
        )
        score = _finite(self.score, field="score")
        expected = anchored_case_score(candidate, baseline, anchor)
        if not math.isclose(score, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("case score does not match its frozen latencies")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObjectiveScoreReport:
    """A continuous objective report for one correctness-qualified result."""

    protocol_id: str
    mode: str
    case_scores: tuple[CaseObjectiveScore, ...]
    objective_score: float
    status: str = "AVAILABLE"
    schema_version: int = OBJECTIVE_SCORE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OBJECTIVE_SCORE_SCHEMA_VERSION:
            raise ValueError("unsupported objective score schema version")
        if self.protocol_id != XPUOJ_TH0_PROXY_PROTOCOL_ID:
            raise ValueError("unsupported objective score protocol")
        if self.mode != XPUOJ_TH0_PROXY_MODE:
            raise ValueError("unsupported objective score mode")
        if self.status != "AVAILABLE":
            raise ValueError("calculated objective score status must be AVAILABLE")
        cases = tuple(self.case_scores)
        if not cases or len({case.case_id for case in cases}) != len(cases):
            raise ValueError("case_scores must contain unique case IDs")
        expected = math.fsum(case.score for case in cases) / len(cases)
        objective = _finite(self.objective_score, field="objective_score")
        if not math.isclose(objective, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise ValueError("objective_score is not the arithmetic case mean")
        object.__setattr__(self, "case_scores", cases)

    @classmethod
    def calculate(
        cls,
        candidate_latencies_ms: Mapping[str, object],
        scoring_baseline_latencies_ms: Mapping[str, object],
        *,
        hardware_anchor_latencies_ms: Mapping[str, object] | None = None,
    ) -> "ObjectiveScoreReport":
        anchors: Mapping[str, object] = (
            {str(key): 0.0 for key in candidate_latencies_ms}
            if hardware_anchor_latencies_ms is None
            else hardware_anchor_latencies_ms
        )
        names = _case_names(
            candidate_latencies_ms,
            scoring_baseline_latencies_ms,
            anchors,
        )
        candidate_by_name = {
            str(key): value for key, value in candidate_latencies_ms.items()
        }
        baseline_by_name = {
            str(key): value for key, value in scoring_baseline_latencies_ms.items()
        }
        anchors_by_name = {str(key): value for key, value in anchors.items()}
        cases: list[CaseObjectiveScore] = []
        for name in names:
            candidate = _finite_positive(
                candidate_by_name[name], field=f"candidate[{name!r}]"
            )
            baseline = _finite_positive(
                baseline_by_name[name], field=f"baseline[{name!r}]"
            )
            anchor = _finite_nonnegative(
                anchors_by_name[name], field=f"anchor[{name!r}]"
            )
            cases.append(
                CaseObjectiveScore(
                    case_id=name,
                    candidate_latency_ms=candidate,
                    scoring_baseline_latency_ms=baseline,
                    hardware_anchor_latency_ms=anchor,
                    score=anchored_case_score(candidate, baseline, anchor),
                )
            )
        return cls(
            protocol_id=XPUOJ_TH0_PROXY_PROTOCOL_ID,
            mode=XPUOJ_TH0_PROXY_MODE,
            case_scores=tuple(cases),
            objective_score=math.fsum(case.score for case in cases) / len(cases),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "mode": self.mode,
            "status": self.status,
            "objective_score": self.objective_score,
            "case_scores": [case.to_dict() for case in self.case_scores],
        }

    @classmethod
    def from_value(cls, value: Mapping[str, object]) -> "ObjectiveScoreReport":
        expected_keys = {
            "schema_version",
            "protocol_id",
            "mode",
            "status",
            "objective_score",
            "case_scores",
        }
        if set(value) != expected_keys:
            raise ValueError("objective score report fields are not exact")
        raw_cases = value["case_scores"]
        if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
            raise ValueError("case_scores must be an array")
        cases: list[CaseObjectiveScore] = []
        for raw in raw_cases:
            if not isinstance(raw, Mapping) or set(raw) != {
                "case_id",
                "candidate_latency_ms",
                "scoring_baseline_latency_ms",
                "hardware_anchor_latency_ms",
                "score",
            }:
                raise ValueError("case score fields are not exact")
            cases.append(
                CaseObjectiveScore(
                    case_id=raw["case_id"],  # type: ignore[arg-type]
                    candidate_latency_ms=raw["candidate_latency_ms"],  # type: ignore[arg-type]
                    scoring_baseline_latency_ms=raw[
                        "scoring_baseline_latency_ms"
                    ],  # type: ignore[arg-type]
                    hardware_anchor_latency_ms=raw[
                        "hardware_anchor_latency_ms"
                    ],  # type: ignore[arg-type]
                    score=raw["score"],  # type: ignore[arg-type]
                )
            )
        report = cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            protocol_id=value["protocol_id"],  # type: ignore[arg-type]
            mode=value["mode"],  # type: ignore[arg-type]
            status=value["status"],  # type: ignore[arg-type]
            objective_score=value["objective_score"],  # type: ignore[arg-type]
            case_scores=tuple(cases),
        )
        if canonical_json_text(report.to_dict()) != canonical_json_text(value):
            raise ValueError("objective score report is not canonical")
        return report


@dataclass(frozen=True, slots=True)
class ScoringBaselineDescriptor:
    """Immutable identity of a compiled scoring baseline."""

    environment_digest: str
    evaluator_profile_digest: str
    measurement_contract_digest: str
    scoring_framework_git_commit: str
    reference_source_sha256: str
    compiler_backend: str
    compiler_config: Mapping[str, object]
    case_baseline_ms: Mapping[str, float]
    protocol_id: str = XPUOJ_TH0_PROXY_PROTOCOL_ID
    schema_version: int = OBJECTIVE_SCORE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OBJECTIVE_SCORE_SCHEMA_VERSION:
            raise ValueError("unsupported scoring baseline schema version")
        if self.protocol_id != XPUOJ_TH0_PROXY_PROTOCOL_ID:
            raise ValueError("unsupported scoring baseline protocol")
        require_sha256_digest(self.environment_digest, field="environment_digest")
        require_sha256_digest(
            self.evaluator_profile_digest, field="evaluator_profile_digest"
        )
        require_sha256_digest(
            self.measurement_contract_digest,
            field="measurement_contract_digest",
        )
        if (
            not isinstance(self.scoring_framework_git_commit, str)
            or not re.fullmatch(r"[0-9a-f]{40}", self.scoring_framework_git_commit)
        ):
            raise ValueError(
                "scoring_framework_git_commit must be 40 lowercase hex digits"
            )
        require_sha256_digest(
            self.reference_source_sha256, field="reference_source_sha256"
        )
        if self.compiler_backend != "inductor":
            raise ValueError("compiler_backend must be inductor")
        compiler_config = _canonical_copy(
            self.compiler_config, field="compiler_config"
        )
        if type(compiler_config) is not dict:
            raise ValueError("compiler_config must be an object")
        names = _case_names(self.case_baseline_ms)
        normalized_cases: dict[str, float] = {}
        for name in names:
            normalized_cases[name] = _finite_positive(
                self.case_baseline_ms[name], field=f"case_baseline_ms[{name!r}]"
            )
        object.__setattr__(
            self, "compiler_config", MappingProxyType(compiler_config)
        )
        object.__setattr__(
            self, "case_baseline_ms", MappingProxyType(normalized_cases)
        )

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "environment_digest": self.environment_digest,
            "evaluator_profile_digest": self.evaluator_profile_digest,
            "measurement_contract_digest": self.measurement_contract_digest,
            "scoring_framework_git_commit": self.scoring_framework_git_commit,
            "reference_source_sha256": self.reference_source_sha256,
            "compiler_backend": self.compiler_backend,
            "compiler_config": _canonical_copy(
                self.compiler_config, field="compiler_config"
            ),
            "case_baseline_ms": {
                str(key): float(value)
                for key, value in sorted(self.case_baseline_ms.items())
            },
        }
        if include_digest:
            value["digest"] = self.digest
        return value

    @classmethod
    def from_value(cls, value: Mapping[str, object]) -> "ScoringBaselineDescriptor":
        expected_keys = {
            "schema_version",
            "protocol_id",
            "environment_digest",
            "evaluator_profile_digest",
            "measurement_contract_digest",
            "scoring_framework_git_commit",
            "reference_source_sha256",
            "compiler_backend",
            "compiler_config",
            "case_baseline_ms",
            "digest",
        }
        if set(value) != expected_keys:
            raise ValueError("scoring baseline descriptor fields are not exact")
        if not isinstance(value["compiler_config"], Mapping):
            raise ValueError("compiler_config must be an object")
        if not isinstance(value["case_baseline_ms"], Mapping):
            raise ValueError("case_baseline_ms must be an object")
        descriptor = cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            protocol_id=value["protocol_id"],  # type: ignore[arg-type]
            environment_digest=value["environment_digest"],  # type: ignore[arg-type]
            evaluator_profile_digest=value[
                "evaluator_profile_digest"
            ],  # type: ignore[arg-type]
            measurement_contract_digest=value[
                "measurement_contract_digest"
            ],  # type: ignore[arg-type]
            scoring_framework_git_commit=value[
                "scoring_framework_git_commit"
            ],  # type: ignore[arg-type]
            reference_source_sha256=value[
                "reference_source_sha256"
            ],  # type: ignore[arg-type]
            compiler_backend=value["compiler_backend"],  # type: ignore[arg-type]
            compiler_config=value["compiler_config"],
            case_baseline_ms=value["case_baseline_ms"],  # type: ignore[arg-type]
        )
        if value["digest"] != descriptor.digest:
            raise ValueError("scoring baseline descriptor digest mismatch")
        if canonical_json_text(descriptor.to_dict()) != canonical_json_text(value):
            raise ValueError("scoring baseline descriptor is not canonical")
        return descriptor


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass(frozen=True, slots=True)
class ScoreNoiseCalibration:
    """Independent same-artifact null distribution for score deltas."""

    null_deltas: tuple[float, ...]
    scoring_baseline_digest: str
    minimum_visible_delta: float = MIN_VISIBLE_SCORE_DELTA
    schema_version: int = OBJECTIVE_SCORE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OBJECTIVE_SCORE_SCHEMA_VERSION:
            raise ValueError("unsupported score calibration schema version")
        require_sha256_digest(
            self.scoring_baseline_digest, field="scoring_baseline_digest"
        )
        normalized = tuple(
            _finite(value, field=f"null_deltas[{index}]")
            for index, value in enumerate(self.null_deltas)
        )
        if len(normalized) < 20:
            raise ValueError("at least 20 independent null deltas are required")
        _finite_nonnegative(
            self.minimum_visible_delta, field="minimum_visible_delta"
        )
        object.__setattr__(self, "null_deltas", normalized)

    @property
    def positive_null_delta_p99(self) -> float:
        return _quantile(
            tuple(max(0.0, float(value)) for value in self.null_deltas), 0.99
        )

    @property
    def score_threshold(self) -> float:
        return max(self.minimum_visible_delta, self.positive_null_delta_p99)

    @property
    def qualified(self) -> bool:
        return self.score_threshold <= MAX_QUALIFIED_NOISE_THRESHOLD

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "scoring_baseline_digest": self.scoring_baseline_digest,
            "null_deltas": list(self.null_deltas),
            "sample_count": len(self.null_deltas),
            "minimum_visible_delta": self.minimum_visible_delta,
            "positive_null_delta_p99": self.positive_null_delta_p99,
            "score_threshold": self.score_threshold,
            "qualified": self.qualified,
        }
        if include_digest:
            value["digest"] = self.digest
        return value

    @classmethod
    def from_value(cls, value: Mapping[str, object]) -> "ScoreNoiseCalibration":
        expected_keys = {
            "schema_version",
            "scoring_baseline_digest",
            "null_deltas",
            "sample_count",
            "minimum_visible_delta",
            "positive_null_delta_p99",
            "score_threshold",
            "qualified",
            "digest",
        }
        if set(value) != expected_keys:
            raise ValueError("score calibration fields are not exact")
        raw_deltas = value["null_deltas"]
        if not isinstance(raw_deltas, Sequence) or isinstance(
            raw_deltas, (str, bytes)
        ):
            raise ValueError("null_deltas must be an array")
        calibration = cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            scoring_baseline_digest=value[
                "scoring_baseline_digest"
            ],  # type: ignore[arg-type]
            null_deltas=tuple(raw_deltas),  # type: ignore[arg-type]
            minimum_visible_delta=value["minimum_visible_delta"],  # type: ignore[arg-type]
        )
        if canonical_json_text(calibration.to_dict()) != canonical_json_text(value):
            raise ValueError("score calibration derived fields or digest mismatch")
        return calibration


@dataclass(frozen=True, slots=True)
class ObjectivePromotionDecision:
    promoted: bool
    needs_confirmation: bool
    reason: str
    primary_objective_delta: float
    confirmation_objective_delta: float | None
    primary_paired_speedup: float
    confirmation_paired_speedup: float | None
    primary_worst_case_regression: float
    confirmation_worst_case_regression: float | None
    score_threshold: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _worst_case_regression(
    baseline: Mapping[str, float], candidate: Mapping[str, float]
) -> float:
    speedups = case_speedups(baseline, candidate)
    return max(max(0.0, (1.0 / speedup) - 1.0) for speedup in speedups.values())


def evaluate_objective_promotion(
    *,
    primary_candidate_score: float,
    primary_incumbent_score: float,
    primary_incumbent_latencies_ms: Mapping[str, float],
    primary_candidate_latencies_ms: Mapping[str, float],
    calibration: ScoreNoiseCalibration,
    confirmation_candidate_score: float | None = None,
    confirmation_incumbent_score: float | None = None,
    confirmation_incumbent_latencies_ms: Mapping[str, float] | None = None,
    confirmation_candidate_latencies_ms: Mapping[str, float] | None = None,
    max_case_regression: float = MAX_CASE_REGRESSION,
) -> ObjectivePromotionDecision:
    """Apply the XPU-OJ objective and incumbent safety gates.

    The caller remains responsible for correctness, identity, environment,
    same-hash confirmation, and resource-fencing checks.
    """

    max_regression = _finite_nonnegative(
        max_case_regression, field="max_case_regression"
    )
    if not calibration.qualified:
        raise ValueError("score noise calibration is not qualified")
    threshold = calibration.score_threshold
    primary_delta = _finite(
        primary_candidate_score, field="primary_candidate_score"
    ) - _finite(primary_incumbent_score, field="primary_incumbent_score")
    primary_speedup = geometric_mean_speedup(
        primary_incumbent_latencies_ms, primary_candidate_latencies_ms
    )
    primary_worst = _worst_case_regression(
        primary_incumbent_latencies_ms, primary_candidate_latencies_ms
    )
    primary_qualifies = (
        primary_delta + 1.0e-12 >= threshold
        and primary_speedup + 1.0e-12 >= 1.0
        and primary_worst <= max_regression + 1.0e-12
    )
    if not primary_qualifies:
        if primary_delta + 1.0e-12 < threshold:
            reason = "objective_delta_below_threshold"
        elif primary_speedup + 1.0e-12 < 1.0:
            reason = "paired_aggregate_regression"
        else:
            reason = "per_case_regression_exceeded"
        return ObjectivePromotionDecision(
            promoted=False,
            needs_confirmation=False,
            reason=reason,
            primary_objective_delta=primary_delta,
            confirmation_objective_delta=None,
            primary_paired_speedup=primary_speedup,
            confirmation_paired_speedup=None,
            primary_worst_case_regression=primary_worst,
            confirmation_worst_case_regression=None,
            score_threshold=threshold,
        )

    confirmation_values = (
        confirmation_candidate_score,
        confirmation_incumbent_score,
        confirmation_incumbent_latencies_ms,
        confirmation_candidate_latencies_ms,
    )
    if all(value is None for value in confirmation_values):
        return ObjectivePromotionDecision(
            promoted=False,
            needs_confirmation=True,
            reason="confirmation_required",
            primary_objective_delta=primary_delta,
            confirmation_objective_delta=None,
            primary_paired_speedup=primary_speedup,
            confirmation_paired_speedup=None,
            primary_worst_case_regression=primary_worst,
            confirmation_worst_case_regression=None,
            score_threshold=threshold,
        )
    if any(value is None for value in confirmation_values):
        raise ValueError("confirmation evidence must be supplied as one complete set")
    assert confirmation_candidate_score is not None
    assert confirmation_incumbent_score is not None
    assert confirmation_incumbent_latencies_ms is not None
    assert confirmation_candidate_latencies_ms is not None
    confirmation_delta = _finite(
        confirmation_candidate_score, field="confirmation_candidate_score"
    ) - _finite(
        confirmation_incumbent_score, field="confirmation_incumbent_score"
    )
    confirmation_speedup = geometric_mean_speedup(
        confirmation_incumbent_latencies_ms,
        confirmation_candidate_latencies_ms,
    )
    confirmation_worst = _worst_case_regression(
        confirmation_incumbent_latencies_ms,
        confirmation_candidate_latencies_ms,
    )
    confirmed = (
        confirmation_delta + 1.0e-12 >= threshold
        and confirmation_speedup + 1.0e-12 >= 1.0
        and confirmation_worst <= max_regression + 1.0e-12
    )
    return ObjectivePromotionDecision(
        promoted=confirmed,
        needs_confirmation=False,
        reason="promoted" if confirmed else "confirmation_failed",
        primary_objective_delta=primary_delta,
        confirmation_objective_delta=confirmation_delta,
        primary_paired_speedup=primary_speedup,
        confirmation_paired_speedup=confirmation_speedup,
        primary_worst_case_regression=primary_worst,
        confirmation_worst_case_regression=confirmation_worst,
        score_threshold=threshold,
    )


__all__ = [
    "CaseObjectiveScore",
    "MAX_CASE_REGRESSION",
    "MAX_QUALIFIED_NOISE_THRESHOLD",
    "MIN_VISIBLE_SCORE_DELTA",
    "OBJECTIVE_SCORE_SCHEMA_VERSION",
    "ObjectivePromotionDecision",
    "ObjectiveScoreReport",
    "ScoreNoiseCalibration",
    "ScoringBaselineDescriptor",
    "XPUOJ_TH0_PROXY_MODE",
    "XPUOJ_TH0_PROXY_PROTOCOL_ID",
    "anchored_case_score",
    "evaluate_objective_promotion",
]
