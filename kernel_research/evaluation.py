"""Reusable raw-evaluation recording and C500 promotion decisions."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import platform
import statistics
import subprocess
from typing import Any

from . import __version__
from .contract import EVALUATION_STATUSES, validate_candidate
from .executor import evaluate_isolated
from .history import ExperimentRecord, HistoryStore
from .research_policy import validate_research_candidate_bounded
from .scoring import evaluate_promotion, percentiles
from .constants import REQUIRED_MATCH_RATIO


SCHEMA_VERSION = 1
EXPECTED_C500_CASES = {
    "smoke": ("smoke_gate_up", "smoke_down"),
    "quick": (
        "quick_decode_gate_up",
        "quick_prefill_gate_up",
        "quick_decode_down",
        "quick_prefill_down",
    ),
    "full": (
        "full_decode_gate_up",
        "full_prefill_gate_up",
        "full_decode_down",
        "full_prefill_down",
    ),
}


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
) -> ExperimentRecord | None:
    records = history.find_by_candidate_hash(candidate_hash)
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
) -> dict[str, Any]:
    """Run an isolated evaluation without opening history or applying promotion."""

    candidate = Path(candidate_path).resolve()
    validation = validate_candidate(candidate)
    baseline = None if baseline_path is None else Path(baseline_path).resolve()
    policy = validate_research_candidate_bounded(validation.source)
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
            backend=backend,
            suite=suite,
            timeout_sec=timeout_sec,
            candidate_source=validation.source,
            baseline_path=baseline,
        )
    environment = controller_environment()
    environment.update(result.get("environment") or {})
    result["environment"] = environment
    payload = dict(result)
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "command": "evaluate-raw",
            "backend": backend,
            "suite": suite,
            "candidate_hash": validation.sha256,
            "baseline_candidate_hash": (
                validate_candidate(baseline).sha256 if baseline is not None else None
            ),
        }
    )
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
        best = (
            history.get_best(backend="c500", suite=suite)
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
            if names != EXPECTED_C500_CASES[suite]:
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
        if backend == "c500":
            apply_c500_promotion(
                recorded_result,
                history=history,
                best=best,
                candidate_hash=validation.sha256,
                suite=suite,
            )
        environment = controller_environment()
        environment.update(recorded_result.get("environment") or {})
        recorded_result["environment"] = environment
        record = history.record_experiment(
            candidate_source=validation.source,
            candidate_hash=validation.sha256,
            git_commit=git_revision,
            backend=backend,
            suite=suite,
            status=str(recorded_result["status"]),
            promotable=bool(
                recorded_result.get("eligible_for_promotion", False)
            ),
            aggregate_score=recorded_result.get("aggregate_score"),
            note=note,
            environment=environment,
            error_summary=recorded_result.get("error"),
            case_measurements=history_cases(recorded_result),
            result=recorded_result,
        )
    return record
