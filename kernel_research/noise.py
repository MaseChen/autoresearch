"""Scientifically isolated same-artifact noise evidence and reports.

V2 evidence is selected by an immutable :class:`NoiseEvidenceScope`, never by
an untagged source hash.  The independent full evaluation is the statistical
unit.  Raw per-case samples remain available for order diagnostics, but are
not falsely treated as independent observations.

The historical hash-based report remains available for one compatibility
cycle.  It is explicitly ``LEGACY_READ_ONLY`` and can never activate the
CURRENT protocol's automatic-promotion gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

from .constants import CURRENT_C500_EVALUATION_PROTOCOL_ID
from .history import ExperimentRecord, HistoryStore, LEGACY_NAMESPACE_ID
from .platform.artifacts import ArtifactId
from .platform.canonical import canonical_sha256, require_sha256_digest
from .platform.identity import (
    BaselineRef,
    ExecutionEnvironmentDigest,
    ExperimentIdentity,
)
from .platform.profiles import (
    CURRENT_RESEARCH_NAMESPACE,
    LEGACY_RESEARCH_NAMESPACE,
    ProfileRef,
)
from .platform.proposal import CandidateBundle, TRITON_PYTHON_BUNDLE_LIMITS


ENVIRONMENT_KEYS = (
    "device",
    "device_name",
    "driver_version",
    "system_maca_version",
    "sdk_version",
    "torch_version",
    "triton_version",
    "evaluator_image",
    "framework_commit",
)
_SOURCE_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
SQLITE_MAX_INT = (1 << 63) - 1


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(_value: str) -> Any:
        raise ValueError(f"{label} contains a non-finite JSON number")

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{label} must contain one JSON object")
    return value


def validate_current_noise_baseline(
    baseline: ExperimentRecord,
    *,
    expected_experiment_uid: str | None = None,
    expected_artifact_id: ArtifactId | None = None,
    expected_environment: ExecutionEnvironmentDigest | None = None,
) -> ExperimentIdentity:
    """Validate the exact CURRENT baseline authority for NOISE evidence."""

    if not isinstance(baseline, ExperimentRecord):
        raise TypeError("noise baseline must be an ExperimentRecord")
    try:
        identity = ExperimentIdentity.from_value(dict(baseline.identity))
        artifact_id = ArtifactId.parse(baseline.artifact_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("noise baseline has no valid V2 scientific identity") from exc
    if (
        baseline.namespace_id != CURRENT_RESEARCH_NAMESPACE.namespace_id
        or identity.namespace != CURRENT_RESEARCH_NAMESPACE
        or identity.evaluation_protocol
        != CURRENT_RESEARCH_NAMESPACE.evaluation_protocol
        or identity.evaluation_protocol.id
        != CURRENT_C500_EVALUATION_PROTOCOL_ID
        or baseline.backend != "c500"
        or baseline.suite != "full"
        or identity.suite != "full"
        or baseline.status != "SUCCESS"
        or not baseline.promotable
        or identity.experiment_uid != baseline.experiment_uid
        or identity.namespace_id != baseline.namespace_id
        or identity.condition_digest != baseline.condition_digest
        or identity.candidate_artifact_id != artifact_id
        or identity.replicate_kind != baseline.replicate_kind
        or identity.replicate_index != baseline.replicate_index
        or not identity.execution_environment.is_resolved
        or baseline.result.get("status") != "SUCCESS"
        or baseline.result.get("evaluation_protocol_id")
        != CURRENT_C500_EVALUATION_PROTOCOL_ID
        or baseline.result.get("request_identity") != identity.to_dict()
    ):
        raise ValueError(
            "noise baseline is not exact CURRENT promotable c500/full evidence"
        )
    if (
        expected_experiment_uid is not None
        and baseline.experiment_uid != expected_experiment_uid
    ):
        raise ValueError("noise evidence references another baseline UID")
    if expected_artifact_id is not None and artifact_id != expected_artifact_id:
        raise ValueError("noise evidence references another baseline artifact")
    if expected_environment is not None:
        identity.execution_environment.require_match(
            expected_environment, context="noise baseline"
        )
    return identity


def trusted_noise_baseline_source(
    history: HistoryStore,
    baseline: ExperimentRecord,
) -> tuple[ExperimentIdentity, bytes]:
    """Verify CURRENT namespace, bundle/source CAS, and entrypoint bytes."""

    namespace = history.get_namespace(CURRENT_RESEARCH_NAMESPACE.namespace_id)
    if (
        namespace is None
        or dict(namespace.identity) != CURRENT_RESEARCH_NAMESPACE.to_dict()
    ):
        raise ValueError("History has no exact CURRENT research namespace snapshot")
    identity = validate_current_noise_baseline(baseline)
    artifact_id = ArtifactId.parse(baseline.artifact_id)
    artifact = history.get_candidate_artifact(str(artifact_id))
    if artifact is None:
        raise ValueError("frozen baseline tagged artifact is missing")
    try:
        artifact_bytes = history.read_candidate_artifact(str(artifact_id))
    except (KeyError, RuntimeError) as exc:
        raise ValueError("frozen baseline tagged artifact is corrupted") from exc
    if artifact.byte_size != len(artifact_bytes):
        raise ValueError("frozen baseline artifact size metadata is inconsistent")

    if artifact_id.is_legacy_source:
        if (
            artifact.artifact_kind != "source_text_v1"
            or artifact_id.digest != baseline.candidate_hash
        ):
            raise ValueError("baseline source artifact identity is inconsistent")
        source_bytes = artifact_bytes
    else:
        if artifact.artifact_kind != "source_bundle_v1":
            raise ValueError("baseline bundle uses an unexpected artifact kind")
        try:
            bundle = CandidateBundle.from_value(
                _strict_json_object(artifact_bytes, label="baseline bundle"),
                limits=TRITON_PYTHON_BUNDLE_LIMITS,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("baseline bundle is invalid") from exc
        if (
            bundle.bundle_bytes != artifact_bytes
            or bundle.artifact_id != artifact_id
            or dict(artifact.manifest) != bundle.manifest
        ):
            raise ValueError("baseline bundle ID, manifest, and bytes disagree")
        entrypoint = next(
            file for file in bundle.files if file.path == bundle.entrypoint
        )
        source_bytes = entrypoint.content_bytes

    source_artifact_id = ArtifactId.source_sha256(baseline.candidate_hash)
    source_artifact = history.get_candidate_artifact(str(source_artifact_id))
    if source_artifact is None or source_artifact.artifact_kind != "source_text_v1":
        raise ValueError("baseline has no authoritative entrypoint source artifact")
    try:
        authoritative_source = history.read_candidate_artifact(
            str(source_artifact_id)
        )
    except (KeyError, RuntimeError) as exc:
        raise ValueError("baseline entrypoint source artifact is corrupted") from exc
    if (
        source_artifact.byte_size != len(authoritative_source)
        or authoritative_source != source_bytes
        or hashlib.sha256(authoritative_source).hexdigest()
        != baseline.candidate_hash
    ):
        raise ValueError("baseline tagged artifact and entrypoint bytes disagree")
    return identity, authoritative_source


def _finite_positive(values: Iterable[object], *, label: str) -> tuple[float, ...]:
    result: list[float] = []
    for index, value in enumerate(values):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}[{index}] must be numeric") from exc
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{label}[{index}] must be finite and positive")
        result.append(number)
    if not result:
        raise ValueError(f"{label} must not be empty")
    return tuple(result)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("quantile input must not be empty")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("geometric mean input must not be empty")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _environment_snapshot(record: ExperimentRecord) -> dict[str, Any]:
    environment = dict(record.environment)
    return {key: environment.get(key) for key in ENVIRONMENT_KEYS}


def _trusted_result_environment_snapshot(
    record: ExperimentRecord, *, label: str
) -> dict[str, Any]:
    raw = record.result.get("environment")
    if not isinstance(raw, Mapping):
        raise ValueError(f"{label} has no trusted evaluator environment snapshot")
    result = {key: raw.get(key) for key in ENVIRONMENT_KEYS}
    if result != _environment_snapshot(record):
        raise ValueError(
            f"{label} evaluator environment disagrees with History"
        )
    return result


def _environment_digest(snapshot: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(snapshot),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def noise_condition_family_digest(identity: ExperimentIdentity) -> str:
    """Digest all scientific conditions except the intentional replicate index."""

    if not isinstance(identity, ExperimentIdentity):
        raise TypeError("identity must be an ExperimentIdentity")
    material = identity.condition_dict()
    material.pop("replicate_index", None)
    return canonical_sha256(
        {
            "schema_version": 1,
            "kind": "same-artifact-noise-condition-family",
            "condition": material,
        }
    )


def _strict_noise_identity(record: ExperimentRecord) -> ExperimentIdentity:
    """Validate one row as non-promotable, resolved V2 noise evidence."""

    try:
        identity = ExperimentIdentity.from_value(dict(record.identity))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"experiment {record.id} has an invalid V2 noise identity: {exc}"
        ) from exc
    column_checks = {
        "experiment_uid": (record.experiment_uid, identity.experiment_uid),
        "namespace_id": (record.namespace_id, identity.namespace_id),
        "artifact_id": (record.artifact_id, str(identity.candidate_artifact_id)),
        "condition_digest": (record.condition_digest, identity.condition_digest),
        "replicate_kind": (record.replicate_kind, identity.replicate_kind),
        "replicate_index": (record.replicate_index, identity.replicate_index),
        "suite": (record.suite, identity.suite),
    }
    mismatched = [
        name for name, (stored, trusted) in column_checks.items() if stored != trusted
    ]
    if mismatched:
        raise ValueError(
            f"experiment {record.id} identity/History mismatch: "
            + ", ".join(sorted(mismatched))
        )
    if identity.replicate_kind != "noise" or identity.stage != "NOISE":
        raise ValueError(
            f"experiment {record.id} is not a dedicated NOISE replicate"
        )
    if identity.replicate_index > SQLITE_MAX_INT:
        raise ValueError("noise replicate_index exceeds SQLite INTEGER range")
    if identity.suite != "full" or record.backend != "c500":
        raise ValueError("V2 noise evidence must use the c500/full protocol")
    if not identity.execution_environment.is_resolved:
        raise ValueError("V2 noise evidence requires a resolved execution environment")
    if (
        identity.candidate_artifact_id != identity.parent_artifact_id
        or identity.candidate_artifact_id != identity.baseline.artifact_id
    ):
        raise ValueError(
            "V2 noise evidence must remeasure the exact frozen baseline artifact"
        )
    if not record.baseline_experiment_uid:
        raise ValueError("V2 noise evidence is missing its frozen baseline experiment")
    if record.promotable or record.result.get("eligible_for_promotion") is not False:
        raise ValueError("noise evidence must be explicitly non-promotable")
    if record.result.get("promotion") is not None:
        raise ValueError("noise evidence must not contain a promotion decision")
    evidence = record.result.get("evidence")
    if not isinstance(evidence, Mapping) or evidence.get("role") != "noise":
        raise ValueError("noise evidence is missing its trusted evidence-only marker")
    if evidence.get("promotion_eligible") is not False:
        raise ValueError("noise evidence marker must deny promotion authority")
    if record.result.get("evaluation_protocol_id") != identity.evaluation_protocol.id:
        raise ValueError("noise evidence protocol does not match its namespace")
    if record.result.get("request_identity") != identity.to_dict():
        raise ValueError("noise evaluator did not exactly echo its request identity")
    return identity


@dataclass(frozen=True, slots=True)
class NoiseEvidenceScope:
    """Exact selector for one scientifically comparable noise cohort."""

    namespace_id: str
    evaluation_protocol: ProfileRef
    condition_family_digest: str
    candidate_hash: str
    candidate_artifact_id: ArtifactId
    parent_artifact_id: ArtifactId
    baseline: BaselineRef
    baseline_experiment_uid: str
    execution_environment: ExecutionEnvironmentDigest
    backend: str = "c500"
    suite: str = "full"

    def __post_init__(self) -> None:
        require_sha256_digest(self.namespace_id, field="noise namespace_id")
        if (
            not isinstance(self.evaluation_protocol, ProfileRef)
            or self.evaluation_protocol.kind != "evaluation_protocol"
        ):
            raise ValueError(
                "noise evaluation_protocol must be an evaluation_protocol ProfileRef"
            )
        require_sha256_digest(
            self.condition_family_digest, field="noise condition_family_digest"
        )
        if not isinstance(self.candidate_hash, str) or not _SOURCE_HASH_RE.fullmatch(
            self.candidate_hash
        ):
            raise ValueError("noise candidate_hash must be a lowercase SHA256")
        if not isinstance(self.candidate_artifact_id, ArtifactId):
            raise ValueError("noise candidate_artifact_id must be an ArtifactId")
        if not isinstance(self.parent_artifact_id, ArtifactId):
            raise ValueError("noise parent_artifact_id must be an ArtifactId")
        if not isinstance(self.baseline, BaselineRef):
            raise ValueError("noise baseline must be a BaselineRef")
        if self.baseline.namespace_id != self.namespace_id:
            raise ValueError("noise baseline belongs to another namespace")
        if (
            self.candidate_artifact_id != self.parent_artifact_id
            or self.candidate_artifact_id != self.baseline.artifact_id
        ):
            raise ValueError("noise scope must target one exact baseline artifact")
        if (
            not isinstance(self.baseline_experiment_uid, str)
            or not self.baseline_experiment_uid
        ):
            raise ValueError("noise scope requires a baseline experiment UID")
        if (
            not isinstance(self.execution_environment, ExecutionEnvironmentDigest)
            or not self.execution_environment.is_resolved
        ):
            raise ValueError("noise scope requires a resolved execution environment")
        self.baseline.require_environment(self.execution_environment)
        if self.backend != "c500" or self.suite != "full":
            raise ValueError("noise scope is restricted to c500/full")

    @classmethod
    def from_anchor(
        cls,
        record: ExperimentRecord,
        *,
        baseline_record: ExperimentRecord,
    ) -> "NoiseEvidenceScope":
        identity = _strict_noise_identity(record)
        if record.status != "SUCCESS":
            raise ValueError("noise scope anchor must be a successful experiment")
        validate_current_noise_baseline(
            baseline_record,
            expected_experiment_uid=record.baseline_experiment_uid,
            expected_artifact_id=identity.candidate_artifact_id,
            expected_environment=identity.execution_environment,
        )
        return cls(
            namespace_id=identity.namespace_id,
            evaluation_protocol=identity.evaluation_protocol,
            condition_family_digest=noise_condition_family_digest(identity),
            candidate_hash=record.candidate_hash,
            candidate_artifact_id=identity.candidate_artifact_id,
            parent_artifact_id=identity.parent_artifact_id,
            baseline=identity.baseline,
            baseline_experiment_uid=str(record.baseline_experiment_uid),
            execution_environment=identity.execution_environment,
            backend=record.backend,
            suite=record.suite,
        )

    @property
    def is_current_protocol(self) -> bool:
        return (
            self.namespace_id == CURRENT_RESEARCH_NAMESPACE.namespace_id
            and self.evaluation_protocol
            == CURRENT_RESEARCH_NAMESPACE.evaluation_protocol
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "namespace_id": self.namespace_id,
            "evaluation_protocol": self.evaluation_protocol.to_dict(),
            "condition_family_digest": self.condition_family_digest,
            "candidate_hash": self.candidate_hash,
            "candidate_artifact_id": str(self.candidate_artifact_id),
            "parent_artifact_id": str(self.parent_artifact_id),
            "baseline": self.baseline.to_dict(),
            "baseline_experiment_uid": self.baseline_experiment_uid,
            "execution_environment": self.execution_environment.to_dict(),
            "backend": self.backend,
            "suite": self.suite,
        }


@dataclass(frozen=True, slots=True)
class NoiseCaseSummary:
    case_id: str
    speedup: float
    candidate_first_speedup: float
    baseline_first_speedup: float
    candidate_p50_us: float
    baseline_p50_us: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "speedup": self.speedup,
            "candidate_first_speedup": self.candidate_first_speedup,
            "baseline_first_speedup": self.baseline_first_speedup,
            "candidate_p50_us": self.candidate_p50_us,
            "baseline_p50_us": self.baseline_p50_us,
        }


@dataclass(frozen=True, slots=True)
class NoiseRunSummary:
    experiment_id: int
    experiment_uid: str | None
    created_at: str
    condition_digest: str | None
    replicate_index: int | None
    run_id: str | None
    aggregate_speedup: float
    candidate_first_aggregate: float
    baseline_first_aggregate: float
    runtime_environment_digest: str
    cases: tuple[NoiseCaseSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "experiment_uid": self.experiment_uid,
            "created_at": self.created_at,
            "condition_digest": self.condition_digest,
            "replicate_index": self.replicate_index,
            "run_id": self.run_id,
            "aggregate_speedup": self.aggregate_speedup,
            "candidate_first_aggregate": self.candidate_first_aggregate,
            "baseline_first_aggregate": self.baseline_first_aggregate,
            "runtime_environment_digest": self.runtime_environment_digest,
            "cases": [case.to_dict() for case in self.cases],
        }


def _case_summary(
    case: Mapping[str, Any],
    *,
    measurement_rounds: int,
    samples_per_round: int,
) -> NoiseCaseSummary:
    case_id = str(case.get("case_id", case.get("name", "unknown")))
    candidate = _finite_positive(
        case.get("latency_samples_us", ()), label=f"{case_id}.candidate"
    )
    baseline = _finite_positive(
        case.get("baseline_latency_samples_us", ()), label=f"{case_id}.baseline"
    )
    expected = measurement_rounds * samples_per_round
    if len(candidate) != len(baseline) or len(candidate) != expected:
        raise ValueError(
            f"case {case_id!r} must contain {expected} paired candidate/baseline samples"
        )
    candidate_first_indexes = tuple(
        index
        for index in range(expected)
        if ((index // samples_per_round) + (index % samples_per_round)) % 2 == 0
    )
    baseline_first_indexes = tuple(
        index for index in range(expected) if index not in candidate_first_indexes
    )

    def speedup(indexes: Sequence[int]) -> float:
        return statistics.median(baseline[index] for index in indexes) / statistics.median(
            candidate[index] for index in indexes
        )

    candidate_p50 = statistics.median(candidate)
    baseline_p50 = statistics.median(baseline)
    return NoiseCaseSummary(
        case_id=case_id,
        speedup=baseline_p50 / candidate_p50,
        candidate_first_speedup=speedup(candidate_first_indexes),
        baseline_first_speedup=speedup(baseline_first_indexes),
        candidate_p50_us=candidate_p50,
        baseline_p50_us=baseline_p50,
    )


def _run_summary(
    record: ExperimentRecord, identity: ExperimentIdentity | None = None
) -> NoiseRunSummary:
    result = dict(record.result)
    benchmark = dict(result.get("benchmark_config") or {})
    rounds = int(benchmark.get("measurement_rounds", 3))
    per_round = int(benchmark.get("samples_per_round", 10))
    raw_cases = result.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"experiment {record.id} contains no cases")
    cases = tuple(
        _case_summary(
            case,
            measurement_rounds=rounds,
            samples_per_round=per_round,
        )
        for case in raw_cases
        if isinstance(case, Mapping)
    )
    if len(cases) != len(raw_cases):
        raise ValueError(f"experiment {record.id} contains an invalid case value")
    snapshot = (
        _environment_snapshot(record)
        if identity is None
        else _trusted_result_environment_snapshot(
            record, label=f"noise experiment {record.id}"
        )
    )
    return NoiseRunSummary(
        experiment_id=record.id,
        experiment_uid=(None if identity is None else identity.experiment_uid),
        created_at=record.created_at,
        condition_digest=(None if identity is None else identity.condition_digest),
        replicate_index=(None if identity is None else identity.replicate_index),
        run_id=(None if identity is None else identity.run_id),
        aggregate_speedup=_geometric_mean([case.speedup for case in cases]),
        candidate_first_aggregate=_geometric_mean(
            [case.candidate_first_speedup for case in cases]
        ),
        baseline_first_aggregate=_geometric_mean(
            [case.baseline_first_speedup for case in cases]
        ),
        runtime_environment_digest=_environment_digest(snapshot),
        cases=cases,
    )


def _bootstrap_median_interval(
    values: Sequence[float], *, confidence: float = 0.99, draws: int = 10_000
) -> tuple[float, float]:
    if not values:
        raise ValueError("bootstrap values must not be empty")
    seed_material = "|".join(f"{value:.17g}" for value in values).encode("ascii")
    generator = random.Random(
        int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    )
    estimates = [
        statistics.median(generator.choice(values) for _ in values)
        for _ in range(draws)
    ]
    alpha = (1.0 - confidence) / 2.0
    return _quantile(estimates, alpha), _quantile(estimates, 1.0 - alpha)


def _report_statistics(
    runs: Sequence[NoiseRunSummary], *, promotion_threshold: float
) -> dict[str, Any]:
    aggregates = tuple(run.aggregate_speedup for run in runs)
    median = statistics.median(aggregates)
    mad = statistics.median(abs(value - median) for value in aggregates)
    p99 = _quantile(aggregates, 0.99)
    bootstrap_low, bootstrap_high = _bootstrap_median_interval(aggregates)
    candidate_first = tuple(run.candidate_first_aggregate for run in runs)
    baseline_first = tuple(run.baseline_first_aggregate for run in runs)
    environment_digests = sorted(
        {run.runtime_environment_digest for run in runs}
    )
    return {
        "null_distribution": {
            "median": median,
            "mad": mad,
            "p01": _quantile(aggregates, 0.01),
            "p05": _quantile(aggregates, 0.05),
            "p95": _quantile(aggregates, 0.95),
            "p99": p99,
            "bootstrap_median_99pct": [bootstrap_low, bootstrap_high],
        },
        "order_diagnostic": {
            "candidate_first_median": statistics.median(candidate_first),
            "baseline_first_median": statistics.median(baseline_first),
            "median_difference": statistics.median(candidate_first)
            - statistics.median(baseline_first),
        },
        "runtime_environment": {
            "consistent": len(environment_digests) == 1,
            "digests": environment_digests,
        },
        "p99_below_promotion_threshold": p99 < promotion_threshold,
    }


def _legacy_noise_report(
    records: Sequence[ExperimentRecord],
    *,
    candidate_hash: str,
    backend: str,
    suite: str,
    promotion_threshold: float,
) -> dict[str, Any]:
    if not _SOURCE_HASH_RE.fullmatch(candidate_hash):
        raise ValueError("candidate_hash must be a lowercase SHA256")
    selected = [
        record
        for record in records
        if record.namespace_id == LEGACY_NAMESPACE_ID
        and record.candidate_hash == candidate_hash
        and record.backend == backend
        and record.suite == suite
        and record.status == "SUCCESS"
        and record.replicate_kind in {"legacy", "primary"}
        and record.result.get("promotion", {}).get("phase") == "remeasurement"
    ]
    if not selected:
        raise ValueError("no successful legacy same-hash remeasurement records matched")
    runs = tuple(_run_summary(record) for record in selected)
    statistics_value = _report_statistics(
        runs, promotion_threshold=promotion_threshold
    )
    sufficient = len(runs) >= 10
    consistent = statistics_value["runtime_environment"]["consistent"]
    below = statistics_value["p99_below_promotion_threshold"]
    statistical_gate_passed = sufficient and consistent and below
    return {
        "api_version": 3,
        "command": "noise-report",
        "authority": "LEGACY_READ_ONLY",
        "scope": {
            "schema_version": 0,
            "namespace_id": LEGACY_RESEARCH_NAMESPACE.namespace_id,
            "evaluation_protocol": (
                LEGACY_RESEARCH_NAMESPACE.evaluation_protocol.to_dict()
            ),
            "candidate_hash": candidate_hash,
            "candidate_artifact_id": f"source-sha256-v1:{candidate_hash}",
            "condition_family_digest": None,
            "baseline": "legacy_unknown",
            "execution_environment": "legacy_unknown",
            "backend": backend,
            "suite": suite,
        },
        "candidate_hash": candidate_hash,
        "backend": backend,
        "suite": suite,
        "independent_run_count": len(runs),
        "runs": [run.to_dict() for run in runs],
        **statistics_value,
        # Preserve the old key as an alias for table/JSON consumers while
        # making the lack of CURRENT authority unambiguous.
        "environment": statistics_value["runtime_environment"],
        "promotion_threshold": promotion_threshold,
        "suggested_positive_noise_floor": max(
            0.0, statistics_value["null_distribution"]["p99"] - 1.0
        ),
        "sufficient_independent_runs": sufficient,
        "statistical_gate_passed": statistical_gate_passed,
        "automatic_promotion_allowed": False,
        "current_protocol_activation_allowed": False,
        "automatic_promotion_block_reason": (
            "legacy evidence is read-only and cannot activate CURRENT"
        ),
    }


def build_noise_report(
    records: Iterable[ExperimentRecord],
    *,
    candidate_hash: str | None = None,
    scope: NoiseEvidenceScope | None = None,
    baseline_record: ExperimentRecord | None = None,
    backend: str = "c500",
    suite: str = "full",
    promotion_threshold: float = 1.01,
) -> dict[str, Any]:
    """Build a legacy read-only or an exact V2 same-artifact noise report.

    V2 callers must derive ``scope`` from one immutable anchor experiment and
    query History through ``HistoryStore.list_noise_experiments``.  Records in
    another namespace, condition family, environment, artifact or baseline are
    excluded rather than pooled.  Malformed rows inside the selected storage
    scope fail closed.
    """

    if not math.isfinite(promotion_threshold) or promotion_threshold <= 1.0:
        raise ValueError("promotion_threshold must be finite and greater than one")
    rows = tuple(records)
    if scope is None:
        if candidate_hash is None:
            raise ValueError("legacy noise reports require candidate_hash")
        return _legacy_noise_report(
            rows,
            candidate_hash=candidate_hash,
            backend=backend,
            suite=suite,
            promotion_threshold=promotion_threshold,
        )
    if not isinstance(scope, NoiseEvidenceScope):
        raise TypeError("scope must be a NoiseEvidenceScope")
    if baseline_record is None:
        raise ValueError("V2 noise reports require the frozen baseline record")
    validate_current_noise_baseline(
        baseline_record,
        expected_experiment_uid=scope.baseline_experiment_uid,
        expected_artifact_id=scope.candidate_artifact_id,
        expected_environment=scope.execution_environment,
    )
    if candidate_hash is not None and candidate_hash != scope.candidate_hash:
        raise ValueError("candidate_hash does not match the V2 noise scope")
    if backend != scope.backend or suite != scope.suite:
        raise ValueError("backend/suite do not match the V2 noise scope")

    selected: list[tuple[ExperimentRecord, ExperimentIdentity]] = []
    high_level_count = 0
    for record in rows:
        if (
            record.namespace_id != scope.namespace_id
            or record.artifact_id != str(scope.candidate_artifact_id)
            or record.baseline_experiment_uid != scope.baseline_experiment_uid
            or record.backend != scope.backend
            or record.suite != scope.suite
            or record.replicate_kind != "noise"
        ):
            continue
        high_level_count += 1
        identity = _strict_noise_identity(record)
        if record.status != "SUCCESS":
            continue
        if (
            identity.evaluation_protocol != scope.evaluation_protocol
            or identity.parent_artifact_id != scope.parent_artifact_id
            or identity.baseline != scope.baseline
            or identity.execution_environment != scope.execution_environment
            or identity.candidate_artifact_id != scope.candidate_artifact_id
            or record.candidate_hash != scope.candidate_hash
            or noise_condition_family_digest(identity)
            != scope.condition_family_digest
        ):
            continue
        selected.append((record, identity))
    if not selected:
        raise ValueError("no successful V2 noise records matched the exact scope")
    uids = [identity.experiment_uid for _, identity in selected]
    indexes = [identity.replicate_index for _, identity in selected]
    if len(set(uids)) != len(uids):
        raise ValueError("noise evidence repeats an experiment_uid")
    if len(set(indexes)) != len(indexes):
        raise ValueError("noise evidence repeats a replicate_index")
    selected.sort(key=lambda item: (item[1].replicate_index, item[0].id))
    indexes = [identity.replicate_index for _, identity in selected]
    runs = tuple(_run_summary(record, identity) for record, identity in selected)
    statistics_value = _report_statistics(
        runs, promotion_threshold=promotion_threshold
    )
    baseline_runtime_environment_digest = _environment_digest(
        _trusted_result_environment_snapshot(
            baseline_record, label="frozen noise baseline"
        )
    )
    run_environment_digests = statistics_value["runtime_environment"][
        "digests"
    ]
    runtime_matches_baseline = (
        len(run_environment_digests) == 1
        and run_environment_digests[0]
        == baseline_runtime_environment_digest
    )
    statistics_value["runtime_environment"].update(
        {
            "baseline_digest": baseline_runtime_environment_digest,
            "matches_frozen_baseline": runtime_matches_baseline,
        }
    )
    sufficient = len(runs) >= 10
    consistent = statistics_value["runtime_environment"]["consistent"]
    below = statistics_value["p99_below_promotion_threshold"]
    statistical_gate_passed = (
        sufficient and consistent and runtime_matches_baseline and below
    )
    activation_allowed = scope.is_current_protocol and statistical_gate_passed
    if activation_allowed:
        block_reason = None
    elif not scope.is_current_protocol:
        block_reason = "scope is not the built-in CURRENT research namespace"
    elif not below:
        block_reason = "same-hash p99 reaches the promotion threshold"
    elif not consistent:
        block_reason = "same-hash runtime environments are inconsistent"
    elif not runtime_matches_baseline:
        block_reason = (
            "same-hash runtime environment differs from the frozen baseline"
        )
    else:
        block_reason = "fewer than 10 independent same-hash full runs"
    return {
        "api_version": 3,
        "command": "noise-report",
        "authority": "CURRENT_NOISE_GATE" if scope.is_current_protocol else "V2_READ_ONLY",
        "scope": scope.to_dict(),
        "candidate_hash": scope.candidate_hash,
        "backend": scope.backend,
        "suite": scope.suite,
        "selected_attempt_count": high_level_count,
        "excluded_or_unsuccessful_attempt_count": high_level_count - len(runs),
        "independent_run_count": len(runs),
        "replicate_indexes": indexes,
        "runs": [run.to_dict() for run in runs],
        **statistics_value,
        "environment": statistics_value["runtime_environment"],
        "baseline_runtime_environment_digest": (
            baseline_runtime_environment_digest
        ),
        "promotion_threshold": promotion_threshold,
        "suggested_positive_noise_floor": max(
            0.0, statistics_value["null_distribution"]["p99"] - 1.0
        ),
        "sufficient_independent_runs": sufficient,
        "statistical_gate_passed": statistical_gate_passed,
        "automatic_promotion_allowed": activation_allowed,
        "current_protocol_activation_allowed": activation_allowed,
        "automatic_promotion_block_reason": block_reason,
    }


def format_noise_report_table(report: Mapping[str, Any]) -> str:
    distribution = report["null_distribution"]
    order = report["order_diagnostic"]
    scope = report.get("scope")
    scope = scope if isinstance(scope, Mapping) else {}
    lines = [
        "same-hash noise report",
        f"authority: {report.get('authority', 'UNKNOWN')}",
        f"namespace: {scope.get('namespace_id', 'unknown')}",
        f"candidate: {report['candidate_hash']}",
        f"artifact: {scope.get('candidate_artifact_id', 'unknown')}",
        f"condition: {scope.get('condition_family_digest', 'legacy_unknown')}",
        f"backend/suite: {report['backend']}/{report['suite']}",
        f"independent runs: {report['independent_run_count']}",
        (
            "null aggregate: "
            f"median={distribution['median']:.8f} "
            f"MAD={distribution['mad']:.8f} "
            f"p01={distribution['p01']:.8f} "
            f"p99={distribution['p99']:.8f}"
        ),
        (
            "order medians: "
            f"candidate-first={order['candidate_first_median']:.8f} "
            f"baseline-first={order['baseline_first_median']:.8f}"
        ),
        f"environment consistent: {str(report['environment']['consistent']).lower()}",
        (
            "p99 below promotion threshold: "
            f"{str(report['p99_below_promotion_threshold']).lower()}"
        ),
        (
            "10-run minimum satisfied: "
            f"{str(report['sufficient_independent_runs']).lower()}"
        ),
        (
            "automatic promotion allowed: "
            f"{str(report['automatic_promotion_allowed']).lower()}"
        ),
    ]
    return "\n".join(lines)


__all__ = [
    "NoiseEvidenceScope",
    "SQLITE_MAX_INT",
    "build_noise_report",
    "format_noise_report_table",
    "noise_condition_family_digest",
    "trusted_noise_baseline_source",
    "validate_current_noise_baseline",
]
