"""Command-line interface for fixed kernel evaluation and history."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from . import __version__
from .contract import validate_candidate
from .executor import doctor_isolated, evaluate_isolated
from .evaluation import (
    apply_c500_promotion,
    controller_environment,
    git_commit,
    history_cases,
    raw_evaluate,
)
from .history import HistoryStore


SCHEMA_VERSION = 1
SUCCESS_STATUSES = frozenset({"MOCK_VALIDATED", "SUCCESS"})

# Backward-compatible private aliases for the original evaluator test surface.
_apply_c500_promotion = apply_c500_promotion
_history_cases = history_cases


def _print_json(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2))


def _doctor(args: argparse.Namespace) -> int:
    payload = doctor_isolated(args.backend, timeout_sec=args.timeout)
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "command": "doctor",
            "backend": args.backend,
            "controller_environment": controller_environment(),
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
    candidate_git_commit = git_commit(candidate)
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
            apply_c500_promotion(
                result,
                history=history,
                best=best,
                candidate_hash=validation.sha256,
                suite=suite,
            )

        environment = controller_environment()
        environment.update(result.get("environment") or {})
        result["environment"] = environment
        record = history.record_experiment(
            candidate_source=validation.source,
            candidate_hash=validation.sha256,
            git_commit=candidate_git_commit,
            backend=backend,
            suite=suite,
            status=result["status"],
            promotable=bool(result.get("eligible_for_promotion", False)),
            aggregate_score=result.get("aggregate_score"),
            note=note,
            environment=environment,
            error_summary=result.get("error"),
            case_measurements=history_cases(result),
            result=result,
        )

    payload = dict(result)
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "command": "evaluate",
            "backend": backend,
            "suite": suite,
            "git_commit": candidate_git_commit,
            "experiment_id": record.id,
            "duplicate": record.duplicate,
            "duplicate_of_id": record.duplicate_of_id,
            "artifact_path": record.artifact_path,
            "note": note,
        }
    )
    return payload


def _evaluate_raw(args: argparse.Namespace) -> int:
    payload = raw_evaluate(
        args.candidate,
        backend=args.backend,
        suite=args.suite,
        baseline_path=args.baseline,
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

    evaluate_raw = subparsers.add_parser(
        "evaluate-raw",
        help="evaluate without opening history or applying promotion",
    )
    evaluate_raw.add_argument(
        "--backend", choices=("mock", "c500"), required=True
    )
    evaluate_raw.add_argument(
        "--suite", choices=("smoke", "quick", "full"), default="smoke"
    )
    evaluate_raw.add_argument("--candidate", required=True)
    evaluate_raw.add_argument("--baseline", default=None)
    evaluate_raw.add_argument("--timeout", type=float, default=None)
    evaluate_raw.set_defaults(handler=_evaluate_raw)

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
