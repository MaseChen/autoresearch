"""Reusable raw-evaluation recording and C500 promotion decisions."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
from typing import Any, Mapping

from . import __version__
from .contract import EVALUATION_STATUSES, validate_candidate
from .executor import evaluate_isolated
from .history import ExperimentRecord, HistoryStore, LEGACY_NAMESPACE_ID
from .platform.canonical import canonical_json_bytes
from .platform.identity import ExperimentIdentity
from .research_policy import validate_research_candidate_bounded
from .scoring import evaluate_promotion, percentiles
from .constants import (
    CURRENT_C500_CASE_IDS,
    CURRENT_C500_EVALUATION_PROTOCOL_ID,
    LEGACY_C500_CASE_IDS,
    LEGACY_C500_EVALUATION_PROTOCOL_ID,
    REQUIRED_MATCH_RATIO,
)


SCHEMA_VERSION = 1
CURRENT_EVALUATION_PROTOCOL_ID = CURRENT_C500_EVALUATION_PROTOCOL_ID
LEGACY_EVALUATION_PROTOCOL_ID = LEGACY_C500_EVALUATION_PROTOCOL_ID
LEGACY_EXPECTED_C500_CASES = LEGACY_C500_CASE_IDS
EXPECTED_C500_CASES = CURRENT_C500_CASE_IDS
REQUEST_IDENTITY_MAX_BYTES = 64 * 1024


def load_request_identity(
    path: str | Path,
    *,
    max_bytes: int = REQUEST_IDENTITY_MAX_BYTES,
) -> dict[str, Any]:
    """Read one bounded strict JSON object supplied by the trusted host."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("request identity size limit must be a positive integer")
    source = Path(path)
    with source.open("rb") as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(
            f"request identity exceeds the {max_bytes}-byte limit"
        )

    def reject_constant(_value: str) -> Any:
        raise ValueError("non-finite JSON numbers are not allowed")

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ValueError("duplicate JSON object keys are not allowed")
            value[key] = child
        return value

    try:
        decoded = raw.decode("utf-8")
        value = json.loads(
            decoded,
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ValueError(
            "request identity must contain one strict UTF-8 JSON object"
        ) from exc
    if type(value) is not dict:
        raise ValueError("request identity must be a JSON object")
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "request identity contains a value outside the strict JSON model"
        ) from exc
    return value


def controller_environment() -> dict[str, Any]:
    return {
        "framework_version": __version__,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pid": os.getpid(),
    }


def git_commit(candidate: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=candidate.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def history_cases(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in result.get("cases", []):
        matched_ratio = case.get("matched_ratio")
        passed = case.get("passed")
        if passed is None and matched_ratio is not None:
            passed = bool(matched_ratio >= REQUIRED_MATCH_RATIO)
        rows.append(
            {
                "name": case.get("case_id", case.get("name", "unknown")),
                "matched_ratio": matched_ratio,
                "passed": None if passed is None else bool(passed),
                "raw_samples": case.get(
                    "latency_samples_us", case.get("raw_samples", [])
                ),
                "baseline_samples": case.get(
                    "baseline_latency_samples_us",
                    case.get("baseline_samples_us", []),
                ),
                "metrics": {
                    key: case.get(key)
                    for key in (
                        "status",
                        "p20_us",
                        "p50_us",
                        "p80_us",
                        "baseline_p20_us",
                        "baseline_p50_us",
                        "baseline_p80_us",
                        "error",
                    )
                    if case.get(key) is not None
                },
            }
        )
    return rows


def case_medians(
    result: dict[str, Any], *, baseline: bool = False
) -> dict[str, float]:
    medians: dict[str, float] = {}
    prefix = "baseline_" if baseline else ""
    for case in result.get("cases", []):
        name = str(case.get("case_id", case.get("name", "unknown")))
        median = case.get(f"{prefix}p50_us")
        if median is None:
            sample_key = (
                "baseline_latency_samples_us"
                if baseline
                else "latency_samples_us"
            )
            samples = case.get(sample_key, [])
            if samples:
                median = statistics.median(float(value) for value in samples)
        if median is None:
            raise ValueError(f"case {name!r} has no {prefix}latency samples")
        medians[name] = float(median)
    if not medians:
        raise ValueError("successful C500 result contains no measured cases")
    return medians


def find_primary_confirmation_source(
    history: HistoryStore,
    *,
    candidate_hash: str,
    baseline_id: int,
    backend: str,
    suite: str,
    namespace_id: str | None = None,
) -> ExperimentRecord | None:
    records = history.find_by_candidate_hash(
        candidate_hash,
        **({} if namespace_id is None else {"namespace_id": namespace_id}),
    )
    consumed_primary_ids = {
        promotion.get("primary_experiment_id")
        for record in records
        for promotion in (record.result.get("promotion", {}),)
        if (
            record.backend == backend
            and record.suite == suite
            and promotion.get("phase") == "confirmation"
            and promotion.get("baseline_experiment_id") == baseline_id
        )
    }
    for record in reversed(records):
        promotion = record.result.get("promotion", {})
        if (
            record.backend == backend
            and record.suite == suite
            and promotion.get("phase") == "primary"
            and promotion.get("baseline_experiment_id") == baseline_id
            and record.id not in consumed_primary_ids
        ):
            return record
    return None


def apply_c500_promotion(
    result: dict[str, Any],
    *,
    history: HistoryStore,
    best: ExperimentRecord | None,
    candidate_hash: str,
    suite: str,
    namespace_id: str | None = None,
) -> None:
    """Bootstrap a baseline, then require an independent confirmation run."""

    previous_primary = None
    if suite == "full" and best is not None and candidate_hash != best.candidate_hash:
        previous_primary = find_primary_confirmation_source(
            history,
            candidate_hash=candidate_hash,
            baseline_id=best.id,
            backend="c500",
            suite=suite,
            namespace_id=namespace_id,
        )

    if result.get("status") != "SUCCESS":
        if previous_primary is not None:
            result["eligible_for_promotion"] = False
            result["aggregate_score"] = None
            result["promotion"] = {
                "phase": "confirmation",
                "reason": "confirmation_status_failed",
                "baseline_experiment_id": best.id if best is not None else None,
                "baseline_candidate_hash": (
                    best.candidate_hash if best is not None else None
                ),
                "primary_experiment_id": previous_primary.id,
                "confirmed": False,
            }
        return

    if suite != "full":
        result["eligible_for_promotion"] = False
        result["aggregate_score"] = None
        result["promotion"] = {
            "phase": "validation",
            "reason": "only_full_suite_is_promotable",
            "confirmed": False,
        }
        return

    if best is None:
        result["eligible_for_promotion"] = True
        result["aggregate_score"] = 1.0
        result["promotion"] = {
            "phase": "baseline",
            "reason": "initial_correct_baseline",
            "baseline_experiment_id": None,
            "baseline_candidate_hash": None,
            "confirmed": True,
            "global_score": 1.0,
        }
        return

    if candidate_hash == best.candidate_hash:
        result["eligible_for_promotion"] = False
        result["aggregate_score"] = best.aggregate_score
        result["promotion"] = {
            "phase": "remeasurement",
            "reason": "candidate_matches_current_best",
            "baseline_experiment_id": best.id,
            "baseline_candidate_hash": best.candidate_hash,
            "confirmed": True,
            "global_score": best.aggregate_score,
        }
        return

    baseline_medians = case_medians(result, baseline=True)
    candidate_medians = case_medians(result, baseline=False)
    primary = evaluate_promotion(baseline_medians, candidate_medians)
    base_global_score = float(
        1.0 if best.aggregate_score is None else best.aggregate_score
    )
    provisional_global_score = base_global_score * primary.aggregate_speedup
    previous_primary = find_primary_confirmation_source(
        history,
        candidate_hash=candidate_hash,
        baseline_id=best.id,
        backend="c500",
        suite=suite,
        namespace_id=namespace_id,
    )
    if previous_primary is not None:
        previous_baseline = case_medians(
            dict(previous_primary.result), baseline=True
        )
        previous_candidate = case_medians(
            dict(previous_primary.result), baseline=False
        )
        confirmation = evaluate_promotion(
            previous_baseline,
            previous_candidate,
            candidate_medians,
            confirmation_baseline=baseline_medians,
        )
        conservative_speedup = min(
            confirmation.aggregate_speedup,
            confirmation.confirmation_speedup or confirmation.aggregate_speedup,
        )
        global_score = base_global_score * conservative_speedup
        result["eligible_for_promotion"] = confirmation.promoted
        result["aggregate_score"] = global_score
        result["promotion"] = {
            "phase": "confirmation",
            "reason": confirmation.reason,
            "baseline_experiment_id": best.id,
            "baseline_candidate_hash": best.candidate_hash,
            "primary_experiment_id": previous_primary.id,
            "confirmed": confirmation.promoted,
            "global_score": global_score,
            "decision": confirmation.to_dict(),
            "primary_normalized_speedup": (
                confirmation.aggregate_speedup
            ),
            "confirmation_normalized_speedup": (
                confirmation.confirmation_speedup
            ),
            "conservative_normalized_speedup": conservative_speedup,
        }
        return

    result["eligible_for_promotion"] = False
    result["aggregate_score"] = provisional_global_score
    result["promotion"] = {
        "phase": "primary" if primary.needs_confirmation else "rejected",
        "reason": primary.reason,
        "baseline_experiment_id": best.id,
        "baseline_candidate_hash": best.candidate_hash,
        "confirmed": False,
        "global_score": provisional_global_score,
        "decision": primary.to_dict(),
    }


def raw_evaluate(
    candidate_path: str | Path,
    *,
    backend: str,
    suite: str,
    baseline_path: str | Path | None = None,
    timeout_sec: float | None = None,
    request_identity_path: str | Path | None = None,
    expected_evaluation_protocol_id: str | None = None,
) -> dict[str, Any]:
    """Run an isolated evaluation without opening history or applying promotion."""

    request_identity = (
        None
        if request_identity_path is None
        else load_request_identity(request_identity_path)
    )
    # A raw V1 request carries no scientific namespace.  Keep it on the
    # historical protocol so a later V1 recording step cannot place current
    # shadow evidence in the legacy namespace.  V2 always supplies an exact
    # request identity and selects its registered protocol below.
    evaluation_protocol_id = LEGACY_EVALUATION_PROTOCOL_ID
    trusted_target = None
    resolved_backend = backend
    if request_identity is not None:
        namespace = request_identity.get("namespace")
        protocol = (
            namespace.get("evaluation_protocol")
            if isinstance(namespace, Mapping)
            else None
        )
        protocol_id = protocol.get("id") if isinstance(protocol, Mapping) else None
        if isinstance(namespace, Mapping) and "namespace_id" in namespace:
            from .platform.legacy import resolve_builtin_target
            from .platform.profiles import ResearchNamespace

            try:
                resolved_namespace = ResearchNamespace.from_value(namespace)
                trusted_target = resolve_builtin_target(resolved_namespace)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(
                    "request identity has no exact trusted target binding"
                ) from exc
            resolved_backend = trusted_target.device.evaluator_backend()
            evaluation_protocol_id = trusted_target.protocol.protocol_id
            if backend != resolved_backend:
                raise ValueError(
                    "requested evaluator backend does not match trusted target"
                )
            trusted_target.protocol.expected_cases(suite)
        else:
            # One-cycle V1 request-echo compatibility.  It may identify only
            # the legacy protocol and never enters the composable V2 route.
            if protocol_id != LEGACY_EVALUATION_PROTOCOL_ID:
                raise ValueError(
                    "request identity references an untrusted evaluation protocol"
                )
            evaluation_protocol_id = str(protocol_id)
    if expected_evaluation_protocol_id is not None:
        if request_identity is None:
            raise ValueError(
                "an explicit evaluation protocol requires a request identity"
            )
        if expected_evaluation_protocol_id != evaluation_protocol_id:
            raise ValueError(
                "requested evaluation protocol does not match trusted target"
            )
    candidate = Path(candidate_path).resolve()
    validation = validate_candidate(candidate)
    baseline = None if baseline_path is None else Path(baseline_path).resolve()
    policy = (
        validate_research_candidate_bounded(validation.source)
        if trusted_target is None
        else trusted_target.validate_candidate(validation.source)
    )
    if not policy.valid:
        result: dict[str, Any] = {
            "status": "CONTRACT_ERROR",
            "eligible_for_promotion": False,
            "aggregate_score": None,
            "cases": [],
            "environment": {},
            "error": "; ".join(
                f"{item.code}: {item.message}" for item in policy.errors
            ),
            "policy": policy.to_dict(),
        }
    else:
        result = evaluate_isolated(
            candidate,
            backend=resolved_backend,
            suite=suite,
            timeout_sec=timeout_sec,
            candidate_source=validation.source,
            baseline_path=baseline,
            evaluation_protocol_id=evaluation_protocol_id,
        )
    returned_protocol = result.get("evaluation_protocol_id")
    if returned_protocol is not None and returned_protocol != evaluation_protocol_id:
        raise ValueError("isolated evaluator returned another protocol identity")
    environment = controller_environment()
    environment.update(result.get("environment") or {})
    result["environment"] = environment
    payload = dict(result)
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "command": "evaluate-raw",
            "backend": resolved_backend,
            "suite": suite,
            "candidate_hash": validation.sha256,
            "baseline_candidate_hash": (
                validate_candidate(baseline).sha256 if baseline is not None else None
            ),
            "evaluation_protocol_id": evaluation_protocol_id,
        }
    )
    if request_identity is not None:
        payload["request_identity"] = request_identity
    return payload


def record_external_result(
    *,
    candidate_source: str,
    result: dict[str, Any],
    backend: str,
    suite: str,
    state_dir: str | Path,
    note: str,
    git_revision: str | None = None,
    identity: Any | None = None,
    baseline_experiment_id: int | None = None,
) -> ExperimentRecord:
    """Validate, promote and persist a result produced by ``evaluate-raw``."""

    validation = validate_candidate(source=candidate_source)
    if not validation.is_valid:
        raise ValueError("cannot record a candidate that fails the public contract")
    if result.get("candidate_hash") not in {None, validation.sha256}:
        raise ValueError("raw result candidate hash does not match source")
    if result.get("backend") not in {None, backend}:
        raise ValueError("raw result backend does not match requested backend")
    if result.get("suite") not in {None, suite}:
        raise ValueError("raw result suite does not match requested suite")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("raw result schema_version must be 1")
    if result.get("command") != "evaluate-raw":
        raise ValueError("raw result command must be evaluate-raw")
    if result.get("status") not in EVALUATION_STATUSES:
        raise ValueError("raw result contains an unknown status")
    if backend == "c500" and result.get("status") == "MOCK_VALIDATED":
        raise ValueError("C500 raw result may not use MOCK_VALIDATED")

    state = Path(state_dir).resolve()
    with HistoryStore(state / "history.sqlite3", state_dir=state) as history:
        identity_value: Mapping[str, Any] | None = None
        evidence_only_identity: ExperimentIdentity | None = None
        if identity is not None:
            to_dict = getattr(identity, "to_dict", None)
            identity_value = (
                to_dict() if callable(to_dict) else identity
            )
            if not isinstance(identity_value, Mapping):
                raise TypeError("identity must be a mapping or expose to_dict()")
            namespace_value = identity_value.get("namespace")
            if not isinstance(namespace_value, Mapping):
                raise ValueError("V3 identity must contain a namespace snapshot")
            namespace_id = namespace_value.get("namespace_id")
            if not isinstance(namespace_id, str):
                raise ValueError("V3 identity namespace_id is missing")
            protocol_value = namespace_value.get("evaluation_protocol")
            protocol_id = (
                protocol_value.get("id")
                if isinstance(protocol_value, Mapping)
                else None
            )
            if result.get("evaluation_protocol_id") != protocol_id:
                raise ValueError(
                    "raw result protocol does not match its research namespace"
                )
            if identity_value.get("replicate_kind") in {"noise", "retest"}:
                try:
                    evidence_only_identity = ExperimentIdentity.from_value(
                        dict(identity_value)
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"evidence-only evaluation identity is invalid: {exc}"
                    ) from exc
                expected_stage = evidence_only_identity.replicate_kind.upper()
                if evidence_only_identity.stage != expected_stage:
                    raise ValueError(
                        "evidence-only replicate stage must match its replicate kind"
                    )
                if backend != "c500" or suite != "full":
                    raise ValueError(
                        "noise/retest collection is restricted to c500/full"
                    )
                if not evidence_only_identity.execution_environment.is_resolved:
                    raise ValueError(
                        "noise/retest collection requires a resolved execution environment"
                    )
                if evidence_only_identity.replicate_kind == "noise" and (
                    evidence_only_identity.candidate_artifact_id
                    != evidence_only_identity.parent_artifact_id
                    or evidence_only_identity.candidate_artifact_id
                    != evidence_only_identity.baseline.artifact_id
                ):
                    raise ValueError(
                        "noise collection must remeasure the exact frozen baseline artifact"
                    )
            if result.get("request_identity") != dict(identity_value):
                raise ValueError("evaluator did not exactly echo request identity")
            history.ensure_namespace(namespace_id, namespace_value)
        else:
            namespace_id = LEGACY_NAMESPACE_ID
            if result.get("evaluation_protocol_id") not in {
                None,
                LEGACY_EVALUATION_PROTOCOL_ID,
            }:
                raise ValueError(
                    "a V1 raw result may only use the legacy evaluation protocol"
                )

        if (
            identity_value is not None
            and backend == "c500"
            and suite == "full"
            and baseline_experiment_id is None
        ):
            raise ValueError(
                "a V3 full result requires an explicit frozen baseline experiment"
            )
        if baseline_experiment_id is not None:
            best = history.get_experiment(baseline_experiment_id)
            if best is None:
                raise ValueError("frozen baseline experiment does not exist")
            if namespace_id is not None and best.namespace_id != namespace_id:
                raise ValueError("frozen baseline belongs to another namespace")
            if evidence_only_identity is not None:
                if (
                    best.artifact_id
                    != str(evidence_only_identity.baseline.artifact_id)
                ):
                    raise ValueError(
                        "evidence-only identity does not reference the frozen "
                        "baseline artifact"
                    )
                try:
                    baseline_identity = ExperimentIdentity.from_value(
                        dict(best.identity)
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "evidence-only collection requires a resolved V2 "
                        "baseline experiment"
                    ) from exc
                if (
                    baseline_identity.experiment_uid != best.experiment_uid
                    or baseline_identity.namespace_id != best.namespace_id
                    or str(baseline_identity.candidate_artifact_id)
                    != best.artifact_id
                    or baseline_identity.condition_digest
                    != best.condition_digest
                ):
                    raise ValueError(
                        "frozen baseline History columns do not match its identity"
                    )
                baseline_identity.execution_environment.require_match(
                    evidence_only_identity.execution_environment,
                    context="noise/retest baseline",
                )
                evidence_only_identity.baseline.require_environment(
                    evidence_only_identity.execution_environment
                )
        else:
            # Namespace-local best lookup exists only for the public V1
            # compatibility workflow.  V2 full evaluations are required to
            # take the explicit branch above.
            best = (
                history.get_best_for_namespace(
                    LEGACY_NAMESPACE_ID,
                    backend="c500",
                    suite=suite,
                )
                if backend == "c500" and suite == "full"
                else None
            )
        if backend == "c500" and suite == "full" and best is not None:
            if result.get("baseline_candidate_hash") != best.candidate_hash:
                raise ValueError(
                    "raw result baseline hash does not match current accepted baseline"
                )
        recorded_result = copy.deepcopy(result)
        recorded_result["eligible_for_promotion"] = False
        recorded_result["aggregate_score"] = None
        recorded_result.pop("promotion", None)
        if backend == "c500" and recorded_result["status"] == "SUCCESS":
            if recorded_result.get("container_exit_code") not in {None, 0}:
                raise ValueError("SUCCESS result has a non-zero container exit code")
            cases = recorded_result.get("cases")
            if not isinstance(cases, list):
                raise ValueError("successful raw C500 result must contain cases")
            names = tuple(str(case.get("case_id")) for case in cases)
            expected_cases = (
                EXPECTED_C500_CASES[suite]
                if result.get("evaluation_protocol_id")
                == CURRENT_EVALUATION_PROTOCOL_ID
                else LEGACY_EXPECTED_C500_CASES[suite]
            )
            if names != expected_cases:
                raise ValueError("raw C500 result case order/identity is invalid")
            for case in cases:
                if case.get("status") != "SUCCESS":
                    raise ValueError("raw SUCCESS result contains a failed case")
                matched = case.get("matched_ratio")
                if (
                    matched is None
                    or not REQUIRED_MATCH_RATIO <= float(matched) <= 1.0
                ):
                    raise ValueError("raw SUCCESS case fails correctness threshold")
                samples = case.get("latency_samples_us")
                if not isinstance(samples, list) or len(samples) != 30:
                    raise ValueError(
                        "raw SUCCESS case must contain 30 candidate samples"
                    )
                candidate_percentiles = percentiles(samples)
                case["p20_us"] = candidate_percentiles["p20"]
                case["p50_us"] = candidate_percentiles["p50"]
                case["p80_us"] = candidate_percentiles["p80"]
                if best is not None and suite == "full":
                    baseline_samples = case.get(
                        "baseline_latency_samples_us"
                    )
                    if (
                        not isinstance(baseline_samples, list)
                        or len(baseline_samples) != 30
                    ):
                        raise ValueError(
                            "raw full comparison must contain 30 baseline samples"
                        )
                    baseline_percentiles = percentiles(baseline_samples)
                    case["baseline_p20_us"] = baseline_percentiles["p20"]
                    case["baseline_p50_us"] = baseline_percentiles["p50"]
                    case["baseline_p80_us"] = baseline_percentiles["p80"]
        if backend == "c500" and evidence_only_identity is None:
            apply_c500_promotion(
                recorded_result,
                history=history,
                best=best,
                candidate_hash=validation.sha256,
                suite=suite,
                namespace_id=namespace_id,
            )
        elif evidence_only_identity is not None:
            # Noise and retest are external scientific evidence.  They can
            # explain or gate future decisions but can never become primary,
            # confirmation or locally promotable evidence themselves.
            from .noise import noise_condition_family_digest

            recorded_result["eligible_for_promotion"] = False
            recorded_result["aggregate_score"] = None
            recorded_result.pop("promotion", None)
            recorded_result["evidence"] = {
                "schema_version": 1,
                "role": evidence_only_identity.replicate_kind,
                "promotion_eligible": False,
                "namespace_id": evidence_only_identity.namespace_id,
                "evaluation_protocol": (
                    evidence_only_identity.evaluation_protocol.to_dict()
                ),
                "condition_digest": evidence_only_identity.condition_digest,
                "condition_family_digest": noise_condition_family_digest(
                    evidence_only_identity
                ),
                "candidate_artifact_id": str(
                    evidence_only_identity.candidate_artifact_id
                ),
                "baseline": evidence_only_identity.baseline.to_dict(),
                "baseline_experiment_uid": (
                    None if best is None else best.experiment_uid
                ),
                "execution_environment": (
                    evidence_only_identity.execution_environment.to_dict()
                ),
            }
        environment = controller_environment()
        environment.update(recorded_result.get("environment") or {})
        recorded_result["environment"] = environment
        record_values = {
            "candidate_source": validation.source,
            "candidate_hash": validation.sha256,
            "git_commit": git_revision,
            "backend": backend,
            "suite": suite,
            "status": str(recorded_result["status"]),
            "promotable": bool(
                recorded_result.get("eligible_for_promotion", False)
            ),
            "aggregate_score": recorded_result.get("aggregate_score"),
            "note": note,
            "environment": environment,
            "error_summary": recorded_result.get("error"),
            "case_measurements": history_cases(recorded_result),
            "result": recorded_result,
        }
        record = (
            history.record_legacy_experiment(
                namespace_id=LEGACY_NAMESPACE_ID,
                **record_values,
            )
            if identity_value is None
            else history.record_experiment(
                **record_values,
                identity=identity_value,
                baseline_experiment_uid=(
                    None if best is None else best.experiment_uid
                ),
            )
        )
        promotion = recorded_result.get("promotion", {})
        primary_id = promotion.get("primary_experiment_id")
        if identity_value is not None and type(primary_id) is int:
            primary = history.get_experiment(primary_id)
            if primary is not None:
                history.add_experiment_relation(
                    source_experiment_uid=primary.experiment_uid,
                    target_experiment_uid=record.experiment_uid,
                    relation_type="confirmation_of",
                    metadata={"baseline_experiment_id": best.id if best else None},
                )
    return record


def record_noise_external_result(
    *,
    candidate_source: str,
    result: dict[str, Any],
    state_dir: str | Path,
    note: str,
    identity: ExperimentIdentity,
    baseline_experiment_id: int,
    git_revision: str | None = None,
) -> ExperimentRecord:
    """Trusted recording entry for one non-promotable C500/full noise run.

    Evaluation remains a separate ``evaluate-raw`` action with the exact
    request identity.  This host-side entry validates its echo, frozen
    baseline, resolved environment and same-artifact contract before History
    insertion.  It deliberately exposes no promotion switch.
    """

    if not isinstance(identity, ExperimentIdentity):
        raise TypeError("identity must be an ExperimentIdentity")
    if identity.replicate_kind != "noise" or identity.stage != "NOISE":
        raise ValueError("noise recording requires a dedicated NOISE identity")
    from .noise import SQLITE_MAX_INT, trusted_noise_baseline_source
    from .platform.identity import BaselineRef
    from .platform.profiles import CURRENT_RESEARCH_NAMESPACE

    if identity.replicate_index > SQLITE_MAX_INT:
        raise ValueError("noise replicate_index exceeds SQLite INTEGER range")
    if type(baseline_experiment_id) is not int or baseline_experiment_id < 1:
        raise ValueError("noise baseline_experiment_id must be positive")
    state = Path(state_dir).resolve()
    with HistoryStore(state / "history.sqlite3", state_dir=state) as history:
        baseline = history.get_experiment(baseline_experiment_id)
        if baseline is None:
            raise ValueError("frozen noise baseline experiment does not exist")
        baseline_identity, baseline_bytes = trusted_noise_baseline_source(
            history, baseline
        )
        expected_ref = BaselineRef.create(
            namespace=CURRENT_RESEARCH_NAMESPACE,
            artifact_id=baseline_identity.candidate_artifact_id,
            source="deployment",
            revision=f"history-{baseline.id}",
            execution_environment=baseline_identity.execution_environment,
        )
        if (
            identity.namespace != CURRENT_RESEARCH_NAMESPACE
            or identity.evaluation_protocol
            != CURRENT_RESEARCH_NAMESPACE.evaluation_protocol
            or identity.candidate_artifact_id
            != baseline_identity.candidate_artifact_id
            or identity.parent_artifact_id
            != baseline_identity.candidate_artifact_id
            or identity.baseline != expected_ref
            or candidate_source.encode("utf-8") != baseline_bytes
        ):
            raise ValueError("noise identity does not freeze the selected baseline")
        baseline_identity.execution_environment.require_match(
            identity.execution_environment,
            context="noise baseline",
        )
    record = record_external_result(
        candidate_source=candidate_source,
        result=result,
        backend="c500",
        suite="full",
        state_dir=state,
        note=note,
        git_revision=git_revision,
        identity=identity,
        baseline_experiment_id=baseline_experiment_id,
    )
    if record.promotable or record.replicate_kind != "noise":  # pragma: no cover
        raise RuntimeError("trusted noise recorder produced promotable evidence")
    return record
