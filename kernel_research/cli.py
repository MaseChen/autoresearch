"""Command-line interface for fixed kernel evaluation and history."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from typing import Any, Sequence

from . import __version__
from .contract import validate_candidate
from .executor import doctor_isolated, evaluate_isolated
from .history import ExperimentRecord, HistoryStore
from .scoring import evaluate_promotion


SCHEMA_VERSION = 1
SUCCESS_STATUSES = frozenset({"MOCK_VALIDATED", "SUCCESS"})


def _controller_environment() -> dict[str, Any]:
    return {
        "framework_version": __version__,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pid": os.getpid(),
    }


def _git_commit(candidate: Path) -> str | None:
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


def _print_json(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2))


def _history_cases(result: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in result.get("cases", []):
        matched_ratio = case.get("matched_ratio")
        passed = case.get("passed")
        if passed is None and matched_ratio is not None:
            passed = bool(matched_ratio >= 0.99)
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


def _case_medians(
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


def _find_primary_confirmation_source(
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


def _apply_c500_promotion(
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
        previous_primary = _find_primary_confirmation_source(
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

    baseline_medians = _case_medians(result, baseline=True)
    candidate_medians = _case_medians(result, baseline=False)
    primary = evaluate_promotion(baseline_medians, candidate_medians)
    base_global_score = float(best.aggregate_score or 1.0)
    provisional_global_score = base_global_score * primary.aggregate_speedup
    if previous_primary is not None:
        previous_baseline = _case_medians(
            dict(previous_primary.result), baseline=True
        )
        previous_candidate = _case_medians(
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


def _doctor(args: argparse.Namespace) -> int:
    payload = doctor_isolated(args.backend, timeout_sec=args.timeout)
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "command": "doctor",
            "backend": args.backend,
            "controller_environment": _controller_environment(),
        }
    )
    _print_json(payload)
    return 0 if payload["status"] in SUCCESS_STATUSES else 2


def evaluate_and_record(
    candidate_path: str | Path,
    *,
    backend: str,
    suite: str,
    state_dir: str | Path,
    note: str = "",
    timeout_sec: float | None = None,
    test_fault: str | None = None,
) -> dict[str, Any]:
    """Run one complete controller iteration and atomically persist it.

    ``test_fault`` is an internal integration-test hook. The CLI intentionally
    exposes no corresponding option.
    """

    candidate = Path(candidate_path).resolve()
    validation = validate_candidate(candidate)
    git_commit = _git_commit(candidate)
    state_dir = Path(state_dir).resolve()
    with HistoryStore(state_dir / "history.sqlite3", state_dir=state_dir) as history:
        best = (
            history.get_best(backend="c500", suite=suite)
            if backend == "c500" and suite == "full"
            else None
        )
        baseline_path = (
            state_dir / best.artifact_path if best is not None else None
        )
        baseline_source = (
            baseline_path.read_text(encoding="utf-8")
            if baseline_path is not None
            else None
        )
        result = evaluate_isolated(
            candidate,
            backend=backend,
            suite=suite,
            timeout_sec=timeout_sec,
            candidate_source=validation.source,
            baseline_path=baseline_path,
            baseline_source=baseline_source,
            test_fault=test_fault,
        )
        if backend == "c500":
            _apply_c500_promotion(
                result,
                history=history,
                best=best,
                candidate_hash=validation.sha256,
                suite=suite,
            )

        environment = _controller_environment()
        environment.update(result.get("environment") or {})
        result["environment"] = environment
        record = history.record_experiment(
            candidate_source=validation.source,
            candidate_hash=validation.sha256,
            git_commit=git_commit,
            backend=backend,
            suite=suite,
            status=result["status"],
            promotable=bool(result.get("eligible_for_promotion", False)),
            aggregate_score=result.get("aggregate_score"),
            note=note,
            environment=environment,
            error_summary=result.get("error"),
            case_measurements=_history_cases(result),
            result=result,
        )

    payload = dict(result)
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "command": "evaluate",
            "backend": backend,
            "suite": suite,
            "git_commit": git_commit,
            "experiment_id": record.id,
            "duplicate": record.duplicate,
            "duplicate_of_id": record.duplicate_of_id,
            "artifact_path": record.artifact_path,
            "note": note,
        }
    )
    return payload


def _evaluate(args: argparse.Namespace) -> int:
    payload = evaluate_and_record(
        args.candidate,
        backend=args.backend,
        suite=args.suite,
        state_dir=args.state_dir,
        note=args.note,
        timeout_sec=args.timeout,
    )
    _print_json(payload)

    if payload["status"] in SUCCESS_STATUSES:
        return 0
    if payload["status"] == "TIMEOUT":
        return 4
    if payload["status"] in {"CONTRACT_ERROR", "PRECISION_FAILED"}:
        return 2
    return 3


def _history(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir).resolve()
    with HistoryStore(state_dir / "history.sqlite3", state_dir=state_dir) as history:
        if args.format == "json":
            print(history.export_json())
        elif args.format == "tsv":
            print(history.export_tsv(), end="")
        else:
            print(history.format_table())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kernel-research",
        description="Evaluate and track Fused MoE Triton candidates.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="inspect backend capabilities")
    doctor.add_argument("--backend", choices=("mock", "c500"), required=True)
    doctor.add_argument("--timeout", type=float, default=None)
    doctor.set_defaults(handler=_doctor)

    evaluate = subparsers.add_parser("evaluate", help="evaluate kernel.py")
    evaluate.add_argument("--backend", choices=("mock", "c500"), required=True)
    evaluate.add_argument(
        "--suite", choices=("smoke", "quick", "full"), default="smoke"
    )
    evaluate.add_argument("--candidate", default="kernel.py")
    evaluate.add_argument("--state-dir", default=".autoresearch")
    evaluate.add_argument("--note", default="")
    evaluate.add_argument("--timeout", type=float, default=None)
    evaluate.set_defaults(handler=_evaluate)

    history = subparsers.add_parser("history", help="show experiment history")
    history.add_argument("--format", choices=("table", "json", "tsv"), default="table")
    history.add_argument("--state-dir", default=".autoresearch")
    history.set_defaults(handler=_history)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
