"""Trusted, shadow-only projection of the qualified XPU-OJ proxy score.

The qualified compiled baseline is an immutable activation identity.  It is
deliberately independent from the deployment incumbent and never grants
promotion authority.  Existing History rows are not rewritten; callers may
project only newly recorded evaluator results or transient read-only views.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from .constants import REQUIRED_MATCH_RATIO
from .objective_scoring import (
    MAX_CASE_REGRESSION,
    ObjectiveScoreReport,
    ScoringBaselineDescriptor,
    XPUOJ_TH0_PROXY_PROTOCOL_ID,
)
from .platform.canonical import canonical_sha256
from .scoring import case_speedups, geometric_mean_speedup
from .scoring_candidate_measurement import (
    SCORING_CANDIDATE_WORKER_REVISION,
    scoring_candidate_measurement_contract_snapshot,
)
from .scoring_measurement import validate_device_event_measurement


SCORING_SHADOW_PROFILE_SCHEMA_VERSION = 1
SCORING_SHADOW_REPORT_SCHEMA_VERSION = 1
SCORING_SHADOW_PROFILE_ID = "xpuoj-th0-proxy-shadow-v1"
SCORING_SHADOW_LABEL = "XPU-OJ TH0 proxy (shadow; not an official OJ score)"
_UNAVAILABLE_REASONS = frozenset(
    {
        "PROFILE_NOT_FROZEN",
        "SCORING_BASELINE_ENVIRONMENT_MISMATCH",
        "NON_CURRENT_NAMESPACE",
        "SUITE_NOT_FULL",
        "RESULT_NOT_SUCCESS",
        "CORRECTNESS_NOT_QUALIFIED",
        "CANDIDATE_MEASUREMENT_NOT_COLLECTED",
        "CANDIDATE_MEASUREMENT_UNQUALIFIED",
        "CANDIDATE_MEASUREMENT_LAUNCH_REJECTED",
    }
)

_QUALIFICATION_OPERATION_ID = "score-baseline-d7423c19f914afc877afc77b"
_QUALIFICATION_OPERATION_DIGEST = (
    "sha256:d7423c19f914afc877afc77be51a53b624a0f41bdfd302362f0e0d4fffd59da9"
)
_QUALIFICATION_DIGEST = (
    "sha256:47bf2fbd525703d1bf5a76544a1d49f7920888a8ce0e43235a82ab70d34ac3f8"
)
_QUALIFICATION_PRIVATE_OBJECT_ID = (
    "sha256:1006965fcc683f498740ce95e972cebeee1a8262830385b0ead383cc525f65bf"
)
_SCORING_FRAMEWORK_GIT_COMMIT = "60e128e1a1e4f288663facca9a9c45fa7e83c368"
_MEASUREMENT_CONTRACT_DIGEST = (
    "sha256:610b8f4d7f492919b72e166ed1b3f8a77cbf9565f13ff5cb55510f545de10352"
)
_QUALIFICATION_EVIDENCE_MANIFEST_SHA256 = (
    "sha256:4511014045b0172a40a68e156a7ce45efbb5c109199998f06148c1d6d2bced11"
)

_QUALIFIED_DESCRIPTOR_VALUE: dict[str, object] = {
    "schema_version": 1,
    "protocol_id": XPUOJ_TH0_PROXY_PROTOCOL_ID,
    "environment_digest": (
        "sha256:9b046016cbec3d4960be7ab08f394a3cc3b854326967fbb802e70c25a4725f8f"
    ),
    "evaluator_profile_digest": (
        "sha256:5c8f41360327d2ec057ff70feec0b3cbc36d4173f893d2b8839ed47f1d58671b"
    ),
    "measurement_contract_digest": _MEASUREMENT_CONTRACT_DIGEST,
    "scoring_framework_git_commit": _SCORING_FRAMEWORK_GIT_COMMIT,
    "reference_source_sha256": (
        "sha256:226aeb398de8ffc892954d7e78e325f77ca3ddc3773fabd7dbba9a4c2c6cefb9"
    ),
    "compiler_backend": "inductor",
    "compiler_config": {
        "backend": "inductor",
        "dynamic": False,
        "fullgraph": True,
        "mode": "default",
    },
    "case_baseline_ms": {
        "full_decode_down": 2.807372808456421,
        "full_decode_gate_up": 5.204220771789551,
        "full_prefill_down": 24.917884826660156,
        "full_prefill_gate_up": 42.22477722167969,
    },
    "digest": (
        "sha256:473706a30fe90a88fdf14c5e2b2ae2bbe72f1b1dc65489fe68040f7ad49d9d83"
    ),
}


def scoring_shadow_profile_snapshot() -> dict[str, object]:
    """Return the exact qualified baseline activation used by new runs."""

    descriptor = ScoringBaselineDescriptor.from_value(
        _QUALIFIED_DESCRIPTOR_VALUE
    )
    material: dict[str, object] = {
        "schema_version": SCORING_SHADOW_PROFILE_SCHEMA_VERSION,
        "profile_id": SCORING_SHADOW_PROFILE_ID,
        "status": "QUALIFIED",
        "mode": "SHADOW_ONLY",
        "label": SCORING_SHADOW_LABEL,
        "promotion_authority": False,
        "qualification_operation_id": _QUALIFICATION_OPERATION_ID,
        "qualification_operation_digest": _QUALIFICATION_OPERATION_DIGEST,
        "qualification_digest": _QUALIFICATION_DIGEST,
        "qualification_private_object_id": _QUALIFICATION_PRIVATE_OBJECT_ID,
        "qualification_evidence_manifest_sha256": (
            _QUALIFICATION_EVIDENCE_MANIFEST_SHA256
        ),
        "scoring_framework_git_commit": _SCORING_FRAMEWORK_GIT_COMMIT,
        "measurement_contract_digest": _MEASUREMENT_CONTRACT_DIGEST,
        "candidate_measurement_contract": (
            scoring_candidate_measurement_contract_snapshot()
        ),
        "scoring_baseline": descriptor.to_dict(),
    }
    return {**material, "digest": canonical_sha256(material)}


def require_scoring_shadow_profile(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Accept only the exact built-in qualified activation profile."""

    expected = scoring_shadow_profile_snapshot()
    if dict(value) != expected:
        raise ValueError("scoring shadow profile differs from its qualified activation")
    return expected


def _finite_positive(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be numeric") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return number


def _correctness_qualified(result: Mapping[str, object]) -> bool:
    cases = result.get("cases")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        return False
    for case in cases:
        if not isinstance(case, Mapping):
            return False
        matched = case.get("matched_ratio")
        if isinstance(matched, bool):
            return False
        try:
            ratio = float(matched)
        except (TypeError, ValueError):
            return False
        if (
            not math.isfinite(ratio)
            or ratio < REQUIRED_MATCH_RATIO
            or ratio > 1.0
        ):
            return False
    return bool(cases)


def _common_report(
    *,
    profile: Mapping[str, object] | None,
    candidate_hash: str | None,
    incumbent_history_experiment_id: int | None,
    incumbent_candidate_hash: str | None,
    candidate_operation_id: str | None = None,
    candidate_operation_digest: str | None = None,
    candidate_operation_result_digest: str | None = None,
) -> dict[str, object]:
    if candidate_hash is not None and not re.fullmatch(r"[0-9a-f]{64}", candidate_hash):
        raise ValueError("candidate_hash must be 64 lowercase hex digits")
    if incumbent_candidate_hash is not None and not re.fullmatch(
        r"[0-9a-f]{64}", incumbent_candidate_hash
    ):
        raise ValueError("incumbent_candidate_hash must be 64 lowercase hex digits")
    if incumbent_history_experiment_id is not None and (
        type(incumbent_history_experiment_id) is not int
        or incumbent_history_experiment_id <= 0
    ):
        raise ValueError("incumbent_history_experiment_id must be positive")
    operation_values = (
        candidate_operation_id,
        candidate_operation_digest,
        candidate_operation_result_digest,
    )
    if any(value is not None for value in operation_values):
        if not all(isinstance(value, str) for value in operation_values):
            raise ValueError("candidate operation identity must be complete")
        if not re.fullmatch(
            r"score-candidate-[0-9a-f]{24}", str(candidate_operation_id)
        ):
            raise ValueError("candidate operation ID is invalid")
        for value in operation_values[1:]:
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(value)):
                raise ValueError("candidate operation digest is invalid")
    validated = (
        None if profile is None else require_scoring_shadow_profile(profile)
    )
    baseline = (
        None
        if validated is None
        else ScoringBaselineDescriptor.from_value(
            validated["scoring_baseline"]  # type: ignore[arg-type]
        )
    )
    return {
        "schema_version": SCORING_SHADOW_REPORT_SCHEMA_VERSION,
        "protocol_id": XPUOJ_TH0_PROXY_PROTOCOL_ID,
        "label": SCORING_SHADOW_LABEL,
        "mode": "SHADOW_ONLY",
        "promotion_authority": False,
        "profile_digest": None if validated is None else validated["digest"],
        "qualification_digest": (
            None if validated is None else validated["qualification_digest"]
        ),
        "scoring_baseline_digest": None if baseline is None else baseline.digest,
        "scoring_framework_git_commit": (
            None
            if validated is None
            else validated["scoring_framework_git_commit"]
        ),
        "candidate_hash": candidate_hash,
        "incumbent_history_experiment_id": incumbent_history_experiment_id,
        "incumbent_candidate_hash": incumbent_candidate_hash,
        "candidate_measurement_digest": None,
        "candidate_measurement": None,
        "candidate_operation_id": candidate_operation_id,
        "candidate_operation_digest": candidate_operation_digest,
        "candidate_operation_result_digest": candidate_operation_result_digest,
    }


def unavailable_scoring_shadow_report(
    *,
    reason: str,
    profile: Mapping[str, object] | None,
    candidate_hash: str | None,
    incumbent_history_experiment_id: int | None,
    incumbent_candidate_hash: str | None,
    candidate_operation_id: str | None = None,
    candidate_operation_digest: str | None = None,
    candidate_operation_result_digest: str | None = None,
) -> dict[str, object]:
    """Return an explicit null-valued shadow result; unavailable is never zero."""

    if reason not in _UNAVAILABLE_REASONS:
        raise ValueError("unsupported scoring shadow unavailable reason")
    material = {
        **_common_report(
            profile=profile,
            candidate_hash=candidate_hash,
            incumbent_history_experiment_id=incumbent_history_experiment_id,
            incumbent_candidate_hash=incumbent_candidate_hash,
            candidate_operation_id=candidate_operation_id,
            candidate_operation_digest=candidate_operation_digest,
            candidate_operation_result_digest=candidate_operation_result_digest,
        ),
        "status": "UNAVAILABLE",
        "reason": reason,
        "objective": None,
        "paired_safety": None,
    }
    return {**material, "digest": canonical_sha256(material)}


def project_scoring_shadow_report(
    result: Mapping[str, object],
    *,
    suite: str,
    profile: Mapping[str, object],
    incumbent_history_experiment_id: int,
    incumbent_candidate_hash: str,
    candidate_measurement: Mapping[str, object] | None = None,
    candidate_framework_git_commit: str | None = None,
    candidate_operation_id: str | None = None,
    candidate_operation_digest: str | None = None,
    candidate_operation_result_digest: str | None = None,
    candidate_operation_status: str | None = None,
) -> dict[str, object]:
    """Project one evaluator result without affecting the promotion decision."""

    validated = require_scoring_shadow_profile(profile)
    candidate_hash_value = result.get("candidate_hash")
    candidate_hash = (
        candidate_hash_value if isinstance(candidate_hash_value, str) else None
    )
    common = {
        "profile": validated,
        "candidate_hash": candidate_hash,
        "incumbent_history_experiment_id": incumbent_history_experiment_id,
        "incumbent_candidate_hash": incumbent_candidate_hash,
        "candidate_operation_id": candidate_operation_id,
        "candidate_operation_digest": candidate_operation_digest,
        "candidate_operation_result_digest": candidate_operation_result_digest,
    }
    if suite != "full":
        return unavailable_scoring_shadow_report(
            reason="SUITE_NOT_FULL", **common
        )
    if result.get("status") != "SUCCESS":
        return unavailable_scoring_shadow_report(
            reason="RESULT_NOT_SUCCESS", **common
        )
    if not _correctness_qualified(result):
        return unavailable_scoring_shadow_report(
            reason="CORRECTNESS_NOT_QUALIFIED", **common
        )

    if candidate_measurement is None:
        if candidate_operation_status == "LAUNCH_REJECTED":
            return unavailable_scoring_shadow_report(
                reason="CANDIDATE_MEASUREMENT_LAUNCH_REJECTED", **common
            )
        return unavailable_scoring_shadow_report(
            reason="CANDIDATE_MEASUREMENT_NOT_COLLECTED", **common
        )
    normalized_measurement = dict(candidate_measurement)
    if (
        candidate_operation_status
        not in {"QUALIFIED", "UNQUALIFIED"}
        or candidate_operation_result_digest
        != canonical_sha256(normalized_measurement)
        or candidate_operation_status != normalized_measurement.get("status")
    ):
        raise ValueError("candidate measurement operation binding is invalid")
    container_exit = normalized_measurement.pop("container_exit_code", None)
    status = normalized_measurement.get("status")
    raw_traceback = normalized_measurement.pop("traceback", None)
    if raw_traceback is not None and (
        not isinstance(raw_traceback, str)
        or len(raw_traceback.encode("utf-8")) > 64 * 1024
    ):
        raise ValueError("candidate measurement traceback is invalid")
    if (
        not isinstance(candidate_framework_git_commit, str)
        or not re.fullmatch(r"[0-9a-f]{40}", candidate_framework_git_commit)
    ):
        raise ValueError("candidate measurement framework commit is not frozen")
    expected_common_echo = {
        "schema_version": 1,
        "command": "score-candidate-probe",
        "protocol_id": XPUOJ_TH0_PROXY_PROTOCOL_ID,
        "worker_revision": SCORING_CANDIDATE_WORKER_REVISION,
        "scoring_framework_git_commit": candidate_framework_git_commit,
        "scoring_profile_digest": validated["digest"],
        "candidate_hash": candidate_hash,
        "incumbent_hash": incumbent_candidate_hash,
        "measurement_contract": validated["candidate_measurement_contract"],
        "timing_protocol": validated["candidate_measurement_contract"][
            "timing_protocol"
        ],
    }
    if any(
        normalized_measurement.get(key) != value
        for key, value in expected_common_echo.items()
    ):
        raise ValueError("candidate measurement identity echo mismatch")
    if status == "UNQUALIFIED":
        if container_exit not in {None, 2}:
            raise ValueError("unqualified candidate measurement has wrong exit")
        if set(normalized_measurement) != {
            *expected_common_echo,
            "status",
            "gpu_state",
            "completion_trusted",
            "phase",
            "error",
            "cases",
        }:
            raise ValueError("unqualified candidate measurement fields are not exact")
        error = normalized_measurement.get("error")
        if (
            not isinstance(error, str)
            or not error
            or len(error.encode("utf-8")) > 4096
            or normalized_measurement.get("cases") != []
            or normalized_measurement.get("gpu_state")
            not in {"NOT_STARTED", "COMPLETED"}
            or normalized_measurement.get("completion_trusted") is not True
            or not isinstance(normalized_measurement.get("phase"), str)
        ):
            raise ValueError("unqualified candidate measurement is invalid")
        report = unavailable_scoring_shadow_report(
            reason="CANDIDATE_MEASUREMENT_UNQUALIFIED", **common
        )
        material = dict(report)
        material.pop("digest")
        material["candidate_measurement_digest"] = canonical_sha256(
            normalized_measurement
        )
        material["candidate_measurement"] = normalized_measurement
        return {**material, "digest": canonical_sha256(material)}
    if status != "QUALIFIED" or container_exit not in {None, 0}:
        raise ValueError("candidate measurement has no trusted terminal status")

    descriptor = ScoringBaselineDescriptor.from_value(
        validated["scoring_baseline"]  # type: ignore[arg-type]
    )
    case_ids = frozenset(descriptor.case_baseline_ms)
    raw_baseline_hash = result.get("baseline_candidate_hash")
    if raw_baseline_hash != incumbent_candidate_hash:
        raise ValueError("evaluator baseline hash differs from the frozen incumbent")
    expected_measurement_fields = {
        "schema_version",
        "command",
        "status",
        "gpu_state",
        "completion_trusted",
        "phase",
        "protocol_id",
        "worker_revision",
        "scoring_framework_git_commit",
        "scoring_profile_digest",
        "candidate_hash",
        "incumbent_hash",
        "measurement_contract",
        "timing_protocol",
        "environment",
        "environment_snapshot_digest",
        "cases",
    }
    if set(normalized_measurement) != expected_measurement_fields:
        raise ValueError("candidate measurement fields are not exact")
    expected_echo = {
        "status": "QUALIFIED",
        "gpu_state": "COMPLETED",
        "completion_trusted": True,
        "phase": "complete",
        **expected_common_echo,
    }
    if any(
        normalized_measurement.get(key) != value
        for key, value in expected_echo.items()
    ):
        raise ValueError("candidate measurement identity echo mismatch")
    environment = normalized_measurement.get("environment")
    if not isinstance(environment, Mapping) or (
        normalized_measurement.get("environment_snapshot_digest")
        != canonical_sha256(environment)
        or normalized_measurement.get("environment_snapshot_digest")
        != validated["candidate_measurement_contract"][
            "qualified_probe_environment_snapshot_digest"
        ]
    ):
        raise ValueError("candidate measurement environment digest mismatch")
    raw_cases = normalized_measurement.get("cases")
    if not isinstance(raw_cases, Sequence) or isinstance(raw_cases, (str, bytes)):
        raise ValueError("candidate measurement cases are not an array")
    candidate_ms: dict[str, float] = {}
    incumbent_ms: dict[str, float] = {}
    expected_case_fields = {
        "case_id",
        "candidate_matched_ratio",
        "candidate_fresh_output_ratio",
        "candidate_anchor_median_ms",
        "candidate_anchor_measurement",
        "paired_candidate_matched_ratio",
        "paired_incumbent_matched_ratio",
        "paired_candidate_fresh_output_ratio",
        "paired_incumbent_fresh_output_ratio",
        "paired_candidate_median_ms",
        "paired_incumbent_median_ms",
        "paired_safety_measurement",
    }
    minimum = float(
        validated["candidate_measurement_contract"]["minimum_correctness_ratio"]
    )
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping) or set(raw_case) != expected_case_fields:
            raise ValueError("candidate measurement case fields are not exact")
        case_id = raw_case.get("case_id")
        if not isinstance(case_id, str) or case_id in candidate_ms:
            raise ValueError("candidate measurement case ID is invalid")
        for ratio_field in (
            "candidate_matched_ratio",
            "candidate_fresh_output_ratio",
            "paired_candidate_matched_ratio",
            "paired_incumbent_matched_ratio",
            "paired_candidate_fresh_output_ratio",
            "paired_incumbent_fresh_output_ratio",
        ):
            ratio = _finite_positive(
                raw_case.get(ratio_field), field=f"{case_id}.{ratio_field}"
            )
            if ratio < minimum or ratio > 1.0:
                raise ValueError("candidate measurement correctness is not qualified")
        _, _, anchor_median = validate_device_event_measurement(
            raw_case.get("candidate_anchor_measurement"),
            field=f"{case_id}.candidate_anchor_measurement",
        )
        paired_candidate, paired_incumbent, _ = validate_device_event_measurement(
            raw_case.get("paired_safety_measurement"),
            field=f"{case_id}.paired_safety_measurement",
        )
        claimed_anchor = _finite_positive(
            raw_case.get("candidate_anchor_median_ms"),
            field=f"{case_id}.candidate_anchor_median_ms",
        )
        claimed_candidate = _finite_positive(
            raw_case.get("paired_candidate_median_ms"),
            field=f"{case_id}.paired_candidate_median_ms",
        )
        claimed_incumbent = _finite_positive(
            raw_case.get("paired_incumbent_median_ms"),
            field=f"{case_id}.paired_incumbent_median_ms",
        )
        if not math.isclose(claimed_anchor, anchor_median, rel_tol=1e-12):
            raise ValueError("candidate anchor median differs from its rounds")
        if not math.isclose(claimed_candidate, paired_candidate, rel_tol=1e-12):
            raise ValueError("paired candidate median differs from its rounds")
        if not math.isclose(claimed_incumbent, paired_incumbent, rel_tol=1e-12):
            raise ValueError("paired incumbent median differs from its rounds")
        candidate_ms[case_id] = claimed_anchor
        incumbent_ms[case_id] = claimed_incumbent
    if frozenset(candidate_ms) != case_ids:
        raise ValueError("candidate measurement case IDs differ from scoring baseline")
    objective = ObjectiveScoreReport.calculate(
        candidate_ms, descriptor.case_baseline_ms
    )
    paired_candidate_ms = {
        str(case["case_id"]): float(case["paired_candidate_median_ms"])
        for case in raw_cases
    }
    speedups = case_speedups(incumbent_ms, paired_candidate_ms)
    aggregate_speedup = geometric_mean_speedup(
        incumbent_ms, paired_candidate_ms
    )
    worst_case_regression = max(
        max(0.0, (1.0 / speedup) - 1.0) for speedup in speedups.values()
    )
    common_report = _common_report(
        profile=validated,
        candidate_hash=candidate_hash,
        incumbent_history_experiment_id=incumbent_history_experiment_id,
        incumbent_candidate_hash=incumbent_candidate_hash,
        candidate_operation_id=candidate_operation_id,
        candidate_operation_digest=candidate_operation_digest,
        candidate_operation_result_digest=candidate_operation_result_digest,
    )
    common_report["candidate_measurement_digest"] = canonical_sha256(
        normalized_measurement
    )
    common_report["candidate_measurement"] = normalized_measurement
    material = {
        **common_report,
        "status": "AVAILABLE",
        "reason": None,
        "objective": objective.to_dict(),
        "paired_safety": {
            "status": "AVAILABLE",
            "incumbent_history_experiment_id": incumbent_history_experiment_id,
            "incumbent_candidate_hash": incumbent_candidate_hash,
            "aggregate_speedup": aggregate_speedup,
            "worst_case_regression": worst_case_regression,
            "maximum_allowed_case_regression": MAX_CASE_REGRESSION,
            "per_case_speedups": dict(sorted(speedups.items())),
        },
    }
    return {**material, "digest": canonical_sha256(material)}


def public_scoring_shadow_summary(
    value: object,
) -> dict[str, object]:
    """Return a bounded, case-free projection for prompts and Console."""

    unavailable = {
        "label": SCORING_SHADOW_LABEL,
        "status": "UNAVAILABLE",
        "reason": "NOT_RECORDED",
        "objective_score": None,
        "objective_delta_from_parity": None,
        "paired_aggregate_speedup": None,
        "paired_worst_case_regression": None,
        "promotion_authority": False,
        "profile_digest": None,
        "scoring_baseline_digest": None,
    }
    if not isinstance(value, Mapping):
        return unavailable
    expected_keys = {
        "schema_version",
        "protocol_id",
        "label",
        "mode",
        "promotion_authority",
        "profile_digest",
        "qualification_digest",
        "scoring_baseline_digest",
        "scoring_framework_git_commit",
        "candidate_hash",
        "incumbent_history_experiment_id",
        "incumbent_candidate_hash",
        "candidate_measurement_digest",
        "candidate_measurement",
        "candidate_operation_id",
        "candidate_operation_digest",
        "candidate_operation_result_digest",
        "status",
        "reason",
        "objective",
        "paired_safety",
        "digest",
    }
    if set(value) != expected_keys:
        raise ValueError("scoring shadow report fields are not exact")
    material = dict(value)
    claimed_digest = material.pop("digest")
    if claimed_digest != canonical_sha256(material):
        raise ValueError("scoring shadow report digest mismatch")
    if (
        value.get("schema_version") != SCORING_SHADOW_REPORT_SCHEMA_VERSION
        or value.get("protocol_id") != XPUOJ_TH0_PROXY_PROTOCOL_ID
        or value.get("label") != SCORING_SHADOW_LABEL
        or value.get("mode") != "SHADOW_ONLY"
        or value.get("promotion_authority") is not False
    ):
        raise ValueError("scoring shadow report contract is invalid")
    status = value.get("status")
    reason = value.get("reason")
    objective = value.get("objective")
    safety = value.get("paired_safety")
    measurement_digest = value.get("candidate_measurement_digest")
    measurement = value.get("candidate_measurement")
    operation_id = value.get("candidate_operation_id")
    operation_digest = value.get("candidate_operation_digest")
    operation_result_digest = value.get("candidate_operation_result_digest")
    expected_profile = scoring_shadow_profile_snapshot()
    report_profile_digest = value.get("profile_digest")
    if report_profile_digest is None:
        if any(
            value.get(field) is not None
            for field in (
                "qualification_digest",
                "scoring_baseline_digest",
                "scoring_framework_git_commit",
            )
        ):
            raise ValueError("unfrozen scoring shadow has activation identity")
    elif (
        report_profile_digest != expected_profile["digest"]
        or value.get("qualification_digest")
        != expected_profile["qualification_digest"]
        or value.get("scoring_baseline_digest")
        != expected_profile["scoring_baseline"]["digest"]
        or value.get("scoring_framework_git_commit")
        != expected_profile["scoring_framework_git_commit"]
    ):
        raise ValueError("scoring shadow activation identity is not trusted")
    operation_present = operation_id is not None
    if operation_present:
        if (
            not isinstance(operation_id, str)
            or not re.fullmatch(r"score-candidate-[0-9a-f]{24}", operation_id)
            or not isinstance(operation_digest, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", operation_digest)
            or not isinstance(operation_result_digest, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", operation_result_digest)
            or operation_id
            != "score-candidate-"
            + operation_digest.removeprefix("sha256:")[:24]
        ):
            raise ValueError("scoring shadow operation identity is invalid")
    elif operation_digest is not None or operation_result_digest is not None:
        raise ValueError("scoring shadow operation identity is incomplete")
    if status == "AVAILABLE":
        if reason is not None or not isinstance(objective, Mapping):
            raise ValueError("available scoring shadow has no objective score")
        if (
            not isinstance(measurement, Mapping)
            or measurement_digest != canonical_sha256(measurement)
            or not operation_present
        ):
            raise ValueError("available scoring shadow measurement is invalid")
        candidate_hash = value.get("candidate_hash")
        incumbent_hash = value.get("incumbent_candidate_hash")
        incumbent_experiment_id = value.get("incumbent_history_experiment_id")
        candidate_framework_commit = measurement.get(
            "scoring_framework_git_commit"
        )
        if (
            not isinstance(candidate_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", candidate_hash)
            or not isinstance(incumbent_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", incumbent_hash)
            or type(incumbent_experiment_id) is not int
            or incumbent_experiment_id <= 0
            or not isinstance(candidate_framework_commit, str)
        ):
            raise ValueError("available scoring shadow identity is invalid")
        # Reuse the complete trusted measurement validator. The scientific
        # evaluator result is represented only by the minimum facts needed to
        # prove candidate/incumbent identity and correctness admission; every
        # timing, environment and worker field comes from the stored
        # measurement and is checked by project_scoring_shadow_report().
        revalidated = project_scoring_shadow_report(
            {
                "status": "SUCCESS",
                "candidate_hash": candidate_hash,
                "baseline_candidate_hash": incumbent_hash,
                "cases": [{"matched_ratio": 1.0}],
            },
            suite="full",
            profile=expected_profile,
            incumbent_history_experiment_id=incumbent_experiment_id,
            incumbent_candidate_hash=incumbent_hash,
            candidate_measurement=measurement,
            candidate_framework_git_commit=candidate_framework_commit,
            candidate_operation_id=operation_id,
            candidate_operation_digest=operation_digest,
            candidate_operation_result_digest=canonical_sha256(measurement),
            candidate_operation_status="QUALIFIED",
        )
        if (
            revalidated["objective"] != objective
            or revalidated["paired_safety"] != safety
            or revalidated["candidate_measurement_digest"] != measurement_digest
        ):
            raise ValueError("scoring shadow differs from trusted measurement")
        objective_report = ObjectiveScoreReport.from_value(objective)
        measurement_cases = measurement.get("cases")
        if not isinstance(measurement_cases, Sequence) or isinstance(
            measurement_cases, (str, bytes)
        ):
            raise ValueError("available scoring shadow has no measurement cases")
        anchors: dict[str, float] = {}
        paired_candidate: dict[str, float] = {}
        paired_incumbent: dict[str, float] = {}
        for raw_case in measurement_cases:
            if not isinstance(raw_case, Mapping):
                raise ValueError("available measurement case is invalid")
            case_id = raw_case.get("case_id")
            if not isinstance(case_id, str) or case_id in anchors:
                raise ValueError("available measurement case ID is invalid")
            anchors[case_id] = _finite_positive(
                raw_case.get("candidate_anchor_median_ms"),
                field=f"{case_id}.candidate_anchor_median_ms",
            )
            paired_candidate[case_id] = _finite_positive(
                raw_case.get("paired_candidate_median_ms"),
                field=f"{case_id}.paired_candidate_median_ms",
            )
            paired_incumbent[case_id] = _finite_positive(
                raw_case.get("paired_incumbent_median_ms"),
                field=f"{case_id}.paired_incumbent_median_ms",
            )
        objective_anchors = {
            case.case_id: case.candidate_latency_ms
            for case in objective_report.case_scores
        }
        if anchors != objective_anchors:
            raise ValueError("objective score differs from candidate measurement")
        objective_score: object = objective_report.objective_score
        if not isinstance(safety, Mapping) or set(safety) != {
            "status",
            "incumbent_history_experiment_id",
            "incumbent_candidate_hash",
            "aggregate_speedup",
            "worst_case_regression",
            "maximum_allowed_case_regression",
            "per_case_speedups",
        } or safety.get("status") != "AVAILABLE":
            raise ValueError("available scoring shadow has no paired safety")
        if (
            safety.get("incumbent_history_experiment_id")
            != value.get("incumbent_history_experiment_id")
            or safety.get("incumbent_candidate_hash")
            != value.get("incumbent_candidate_hash")
        ):
            raise ValueError("paired safety identity differs from its report")
        aggregate_speedup = _finite_positive(
            safety.get("aggregate_speedup"), field="paired aggregate speedup"
        )
        worst_case_regression = safety.get("worst_case_regression")
        if isinstance(worst_case_regression, bool):
            raise ValueError("paired worst-case regression is invalid")
        try:
            worst_case_number = float(worst_case_regression)
        except (TypeError, ValueError) as exc:
            raise ValueError("paired worst-case regression is invalid") from exc
        if not math.isfinite(worst_case_number) or worst_case_number < 0.0:
            raise ValueError("paired worst-case regression is invalid")
        if safety.get("maximum_allowed_case_regression") != MAX_CASE_REGRESSION:
            raise ValueError("paired safety threshold differs")
        expected_speedups = case_speedups(paired_incumbent, paired_candidate)
        expected_aggregate = geometric_mean_speedup(
            paired_incumbent, paired_candidate
        )
        expected_worst = max(
            max(0.0, (1.0 / speedup) - 1.0)
            for speedup in expected_speedups.values()
        )
        if (
            safety.get("per_case_speedups") != dict(sorted(expected_speedups.items()))
            or not math.isclose(
                aggregate_speedup, expected_aggregate, rel_tol=1.0e-12
            )
            or not math.isclose(
                worst_case_number, expected_worst, rel_tol=1.0e-12
            )
        ):
            raise ValueError("paired safety differs from candidate measurement")
    elif status == "UNAVAILABLE":
        if (
            reason not in _UNAVAILABLE_REASONS
            or objective is not None
            or safety is not None
        ):
            raise ValueError("unavailable scoring shadow is inconsistent")
        if reason == "CANDIDATE_MEASUREMENT_UNQUALIFIED":
            if (
                not isinstance(measurement, Mapping)
                or measurement_digest != canonical_sha256(measurement)
                or not operation_present
            ):
                raise ValueError("unqualified measurement digest is invalid")
        elif measurement_digest is not None or measurement is not None:
            raise ValueError("unavailable report has an unexpected measurement")
        objective_score = None
        aggregate_speedup = None
        worst_case_number = None
    else:
        raise ValueError("scoring shadow status is invalid")
    return {
        "label": value.get("label", SCORING_SHADOW_LABEL),
        "status": status,
        "reason": reason,
        "objective_score": objective_score,
        "objective_delta_from_parity": (
            None if objective_score is None else float(objective_score) - 50.0
        ),
        "paired_aggregate_speedup": (
            aggregate_speedup
        ),
        "paired_worst_case_regression": (
            worst_case_number
        ),
        "promotion_authority": False,
        "profile_digest": value.get("profile_digest"),
        "scoring_baseline_digest": value.get("scoring_baseline_digest"),
    }


SCORING_SHADOW_PROFILE_DIGEST = str(scoring_shadow_profile_snapshot()["digest"])


__all__ = [
    "SCORING_SHADOW_LABEL",
    "SCORING_SHADOW_PROFILE_DIGEST",
    "SCORING_SHADOW_PROFILE_ID",
    "project_scoring_shadow_report",
    "public_scoring_shadow_summary",
    "require_scoring_shadow_profile",
    "scoring_shadow_profile_snapshot",
    "unavailable_scoring_shadow_report",
]
