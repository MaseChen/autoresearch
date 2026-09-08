"""Legacy-compatible CLI plus bounded V2 evidence operations.

The ``evaluate`` command intentionally remains bound to the legacy research
namespace and protocol for one compatibility cycle.  Its ``promotable`` field
is legacy History evidence only; it never updates a V2 campaign lineage or the
authoritative deployment baseline pin.

``noise-collect`` is the only V2 operation here that may append History; the
recorder strips all promotion authority.  ``profile collect`` writes only an
advisory raw trace/private-CAS object and scientific summary/CAS object.  Neither
operation can change a baseline or Campaign lineage.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from . import __version__
from .constants import LEGACY_C500_EVALUATION_PROTOCOL_ID
from .contract import validate_candidate
from .executor import doctor_isolated, evaluate_isolated
from .evaluation import (
    apply_c500_promotion,
    controller_environment,
    git_commit,
    history_cases,
    raw_evaluate,
)
from .history import HistoryStore, LEGACY_NAMESPACE_ID
from .noise import (
    NoiseEvidenceScope,
    build_noise_report,
    format_noise_report_table,
)
from .autorun.controller import ResearchController
from .autorun.errors import ControlledRuntimeError
from .autorun.models import ControllerConfig
from .profiling import (
    PROFILE_RECIPE_IDS,
    abandon_profile_image_canary,
    finalize_known_profile_image_canary,
    run_bounded_profile,
    run_profile_image_doctor,
    run_profiling_doctor,
)


SCHEMA_VERSION = 1
SUCCESS_STATUSES = frozenset({"MOCK_VALIDATED", "SUCCESS"})
SCORING_FRAMEWORK_COMMIT_ENV = "KERNEL_RESEARCH_SCORING_FRAMEWORK_COMMIT"

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


def _score_baseline_probe(_args: argparse.Namespace) -> int:
    # Lazy import is mandatory: the trusted host Python intentionally has no
    # Torch/NumPy, while this fixed command runs inside the evaluator image.
    from .scoring_baseline_worker import run_scoring_baseline_probe

    payload = run_scoring_baseline_probe(
        scoring_framework_git_commit=os.environ.get(
            SCORING_FRAMEWORK_COMMIT_ENV, ""
        )
    )
    _print_json(payload)
    return 0 if payload.get("status") == "QUALIFIED" else 2


def _score_baseline_qualify(args: argparse.Namespace) -> int:
    controller = ResearchController(ControllerConfig.load(args.config))
    payload = controller.qualify_scoring_baseline()
    _print_json(payload)
    return 0 if payload.get("status") == "QUALIFIED" else 2


def _score_baseline_finalize_pre_gpu(args: argparse.Namespace) -> int:
    controller = ResearchController(ControllerConfig.load(args.config))
    payload = controller.finalize_scoring_pre_gpu_failure()
    _print_json(payload)
    return 0


def _score_baseline_abandon_unknown_oom(args: argparse.Namespace) -> int:
    controller = ResearchController(ControllerConfig.load(args.config))
    payload = controller.finalize_scoring_unknown_oom()
    _print_json(payload)
    return 0


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
    """Run and persist one legacy-compatible evaluation.

    Baseline lookup below is the historical, namespace-local compatibility
    ranking.  It is not a V2 ``BaselineRef`` and has no deployment authority.

    ``test_fault`` is an internal integration-test hook. The CLI intentionally
    exposes no corresponding option.
    """

    candidate = Path(candidate_path).resolve()
    validation = validate_candidate(candidate)
    candidate_git_commit = git_commit(candidate)
    state_dir = Path(state_dir).resolve()
    with HistoryStore(state_dir / "history.sqlite3", state_dir=state_dir) as history:
        best = (
            history.get_best_for_namespace(
                LEGACY_NAMESPACE_ID,
                backend="c500",
                suite=suite,
            )
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
            evaluation_protocol_id=LEGACY_C500_EVALUATION_PROTOCOL_ID,
            test_fault=test_fault,
        )
        if (
            result.get("evaluation_protocol_id")
            != LEGACY_C500_EVALUATION_PROTOCOL_ID
        ):
            raise ValueError(
                "legacy evaluator returned another evaluation protocol"
            )
        if backend == "c500":
            apply_c500_promotion(
                result,
                history=history,
                best=best,
                candidate_hash=validation.sha256,
                suite=suite,
                namespace_id=LEGACY_NAMESPACE_ID,
            )

        environment = controller_environment()
        environment.update(result.get("environment") or {})
        result["environment"] = environment
        record = history.record_legacy_experiment(
            namespace_id=LEGACY_NAMESPACE_ID,
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
        request_identity_path=args.request_identity,
        expected_evaluation_protocol_id=args.evaluation_protocol,
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


def _noise_report(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir).resolve()
    with HistoryStore(
        state_dir / "history.sqlite3", state_dir=state_dir
    ) as history:
        strict_requested = bool(args.namespace_id or args.anchor_experiment_uid)
        if strict_requested:
            if not args.namespace_id or not args.anchor_experiment_uid:
                raise ValueError(
                    "V2 noise reports require both --namespace-id and "
                    "--anchor-experiment-uid"
                )
            anchor = history.get_experiment_by_uid(args.anchor_experiment_uid)
            if anchor is None:
                raise ValueError("noise anchor experiment does not exist")
            if anchor.namespace_id != args.namespace_id:
                raise ValueError("noise anchor belongs to another namespace")
            if not anchor.baseline_experiment_uid:
                raise ValueError("noise anchor has no frozen baseline UID")
            baseline = history.get_experiment_by_uid(
                anchor.baseline_experiment_uid
            )
            if baseline is None:
                raise ValueError("noise anchor baseline no longer exists")
            scope = NoiseEvidenceScope.from_anchor(
                anchor, baseline_record=baseline
            )
            assertions = {
                "candidate hash": (args.candidate_hash, scope.candidate_hash),
                "protocol digest": (
                    args.protocol_digest,
                    scope.evaluation_protocol.digest,
                ),
                "condition family digest": (
                    args.condition_family_digest,
                    scope.condition_family_digest,
                ),
                "environment digest": (
                    args.environment_digest,
                    scope.execution_environment.digest,
                ),
                "artifact ID": (
                    args.artifact_id,
                    str(scope.candidate_artifact_id),
                ),
                "baseline experiment UID": (
                    args.baseline_experiment_uid,
                    scope.baseline_experiment_uid,
                ),
            }
            mismatched = [
                name
                for name, (requested, frozen) in assertions.items()
                if requested is not None and requested != frozen
            ]
            if mismatched:
                raise ValueError(
                    "noise scope assertion mismatch: "
                    + ", ".join(sorted(mismatched))
                )
            records = history.list_noise_experiments(
                namespace_id=scope.namespace_id,
                artifact_id=str(scope.candidate_artifact_id),
                baseline_experiment_uid=scope.baseline_experiment_uid,
                backend=scope.backend,
                suite=scope.suite,
            )
            report = build_noise_report(
                records, scope=scope, baseline_record=baseline
            )
        else:
            if not args.candidate_hash:
                raise ValueError(
                    "legacy noise reports require --candidate-hash"
                )
            legacy_only_options = {
                "--protocol-digest": args.protocol_digest,
                "--condition-family-digest": args.condition_family_digest,
                "--environment-digest": args.environment_digest,
                "--artifact-id": args.artifact_id,
                "--baseline-experiment-uid": args.baseline_experiment_uid,
            }
            unexpected = [
                name for name, value in legacy_only_options.items() if value is not None
            ]
            if unexpected:
                raise ValueError(
                    "V2 scope assertions require --namespace-id and "
                    "--anchor-experiment-uid: "
                    + ", ".join(unexpected)
                )
            report = build_noise_report(
                history.list_experiments(candidate_hash=args.candidate_hash),
                candidate_hash=args.candidate_hash,
                backend=args.backend,
                suite=args.suite,
            )
    if args.format == "json":
        _print_json(report)
    else:
        print(format_noise_report_table(report))
    return 0


def _noise_collect(args: argparse.Namespace) -> int:
    controller = ResearchController(ControllerConfig.load(args.config))
    record = controller.collect_noise(
        candidate_path=args.candidate,
        namespace_id=args.namespace_id,
        baseline_experiment_uid=args.baseline_experiment_uid,
        replicate_index=args.replicate_index,
    )
    payload = dict(record.result)
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "command": "noise-collect",
            "experiment_id": record.id,
            "experiment_uid": record.experiment_uid,
            "namespace_id": record.namespace_id,
            "artifact_id": record.artifact_id,
            "baseline_experiment_uid": record.baseline_experiment_uid,
            "replicate_kind": record.replicate_kind,
            "replicate_index": record.replicate_index,
            "promotable": record.promotable,
        }
    )
    _print_json(payload)
    if record.status == "SUCCESS":
        return 0
    if record.status == "TIMEOUT":
        return 4
    if record.status in {"CONTRACT_ERROR", "PRECISION_FAILED"}:
        return 2
    return 3


def _profile_doctor(args: argparse.Namespace) -> int:
    report = run_profiling_doctor(timeout_sec=args.timeout)
    _print_json(report)
    return 0 if report["status"] == "READY" else 2


def _profile_collect(args: argparse.Namespace) -> int:
    report = run_bounded_profile(
        ControllerConfig.load(args.config),
        campaign_database=args.database,
        campaign_id=args.campaign_id,
        gate_id=args.gate_id,
        experiment_uid=args.experiment_uid,
        namespace_id=args.namespace_id,
        execution_environment_digest=args.environment_digest,
        recipe_id=args.recipe,
    )
    _print_json(report)
    return 0


def _profile_image_doctor(args: argparse.Namespace) -> int:
    report = run_profile_image_doctor(
        ControllerConfig.load(args.config),
        campaign_database=args.database,
        campaign_id=args.campaign_id,
    )
    _print_json(report)
    return 0 if report["status"] in {"READY", "ALREADY_COMPLETED"} else 2


def _profile_image_doctor_abandon(args: argparse.Namespace) -> int:
    report = abandon_profile_image_canary(
        ControllerConfig.load(args.config),
        campaign_database=args.database,
        campaign_id=args.campaign_id,
    )
    _print_json(report)
    return 0


def _profile_image_doctor_finalize_known(args: argparse.Namespace) -> int:
    report = finalize_known_profile_image_canary(
        ControllerConfig.load(args.config),
        campaign_database=args.database,
        campaign_id=args.campaign_id,
    )
    _print_json(report)
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

    score_baseline_probe = subparsers.add_parser(
        "score-baseline-probe",
        help=argparse.SUPPRESS,
    )
    score_baseline_probe.set_defaults(handler=_score_baseline_probe)

    score = subparsers.add_parser(
        "score", help="qualify and inspect versioned objective scoring evidence"
    )
    score_commands = score.add_subparsers(dest="score_command", required=True)
    score_baseline_qualify = score_commands.add_parser(
        "baseline-qualify",
        help="run the fixed ten-probe compiled scoring baseline qualification",
    )
    score_baseline_qualify.add_argument("--config", required=True)
    score_baseline_qualify.set_defaults(handler=_score_baseline_qualify)
    score_baseline_finalize = score_commands.add_parser(
        "baseline-finalize-pre-gpu",
        help="finalize the one proven pre-GPU scoring CLI rejection",
    )
    score_baseline_finalize.add_argument("--config", required=True)
    score_baseline_finalize.set_defaults(
        handler=_score_baseline_finalize_pre_gpu
    )
    score_baseline_abandon_oom = score_commands.add_parser(
        "baseline-abandon-unknown-oom",
        help="abandon the exact archived scoring OOM after a trusted doctor",
    )
    score_baseline_abandon_oom.add_argument("--config", required=True)
    score_baseline_abandon_oom.set_defaults(
        handler=_score_baseline_abandon_unknown_oom
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate kernel.py in the legacy compatibility namespace",
    )
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
    evaluate_raw.add_argument("--request-identity", default=None)
    evaluate_raw.add_argument("--evaluation-protocol", default=None)
    evaluate_raw.add_argument("--timeout", type=float, default=None)
    evaluate_raw.set_defaults(handler=_evaluate_raw)

    history = subparsers.add_parser("history", help="show experiment history")
    history.add_argument("--format", choices=("table", "json", "tsv"), default="table")
    history.add_argument("--state-dir", default=".autoresearch")
    history.set_defaults(handler=_history)

    noise = subparsers.add_parser(
        "noise-report",
        help="summarize isolated same-artifact A/B remeasurement noise",
    )
    noise.add_argument("--state-dir", default=".autoresearch")
    noise.add_argument(
        "--candidate-hash",
        help=(
            "legacy hash selector, or an optional assertion for a V2 anchor"
        ),
    )
    noise.add_argument(
        "--namespace-id",
        help="exact V2 ResearchNamespace digest (requires an anchor UID)",
    )
    noise.add_argument(
        "--anchor-experiment-uid",
        help=(
            "immutable NOISE experiment whose protocol, condition, environment, "
            "artifact and BaselineRef define the report scope"
        ),
    )
    noise.add_argument("--protocol-digest")
    noise.add_argument("--condition-family-digest")
    noise.add_argument("--environment-digest")
    noise.add_argument("--artifact-id")
    noise.add_argument("--baseline-experiment-uid")
    noise.add_argument("--backend", default="c500")
    noise.add_argument("--suite", default="full")
    noise.add_argument("--format", choices=("json", "table"), default="table")
    noise.set_defaults(handler=_noise_report)

    noise_collect = subparsers.add_parser(
        "noise-collect",
        help=(
            "collect one non-promotable CURRENT c500/full same-baseline "
            "replicate"
        ),
    )
    noise_collect.add_argument(
        "--config",
        required=True,
        help="trusted ControllerConfig; its bounded evaluator timeout is used",
    )
    noise_collect.add_argument("--candidate", default="kernel.py")
    noise_collect.add_argument("--namespace-id", required=True)
    noise_collect.add_argument("--baseline-experiment-uid", required=True)
    noise_collect.add_argument("--replicate-index", required=True, type=int)
    noise_collect.set_defaults(handler=_noise_collect)

    profile = subparsers.add_parser(
        "profile", help="inspect or collect advisory profiling evidence"
    )
    profile_commands = profile.add_subparsers(
        dest="profile_command", required=True
    )
    profile_doctor = profile_commands.add_parser(
        "doctor", help="inspect the fixed MetaX profiling toolchain"
    )
    profile_doctor.add_argument("--timeout", type=float, default=5.0)
    profile_doctor.set_defaults(handler=_profile_doctor)
    profile_image_doctor = profile_commands.add_parser(
        "image-doctor",
        help="qualify the fixed active profiler image before Campaign soak",
    )
    profile_image_doctor.add_argument("--config", required=True)
    profile_image_doctor.add_argument(
        "--database",
        required=True,
        help="canonical <runtime_root>/campaign/campaign.sqlite3",
    )
    profile_image_doctor.add_argument("--campaign-id", required=True)
    profile_image_doctor.set_defaults(handler=_profile_image_doctor)
    profile_image_doctor_abandon = profile_commands.add_parser(
        "image-doctor-abandon",
        help=(
            "run a fresh trusted doctor and terminalize one quarantined "
            "canary without replay"
        ),
    )
    profile_image_doctor_abandon.add_argument("--config", required=True)
    profile_image_doctor_abandon.add_argument(
        "--database",
        required=True,
        help="canonical <runtime_root>/campaign/campaign.sqlite3",
    )
    profile_image_doctor_abandon.add_argument("--campaign-id", required=True)
    profile_image_doctor_abandon.set_defaults(
        handler=_profile_image_doctor_abandon
    )
    profile_image_doctor_finalize_known = profile_commands.add_parser(
        "image-doctor-finalize-known",
        help=(
            "terminalize the exact settled A6 known failure without doctor "
            "or replay"
        ),
    )
    profile_image_doctor_finalize_known.add_argument("--config", required=True)
    profile_image_doctor_finalize_known.add_argument(
        "--database",
        required=True,
        help="canonical <runtime_root>/campaign/campaign.sqlite3",
    )
    profile_image_doctor_finalize_known.add_argument(
        "--campaign-id", required=True
    )
    profile_image_doctor_finalize_known.set_defaults(
        handler=_profile_image_doctor_finalize_known
    )
    profile_collect = profile_commands.add_parser(
        "collect",
        help=(
            "collect one advisory trace from a soak-qualified V2 experiment"
        ),
    )
    profile_collect.add_argument("--config", required=True)
    profile_collect.add_argument(
        "--database",
        required=True,
        help="canonical <runtime_root>/campaign/campaign.sqlite3",
    )
    profile_collect.add_argument("--campaign-id", required=True)
    profile_collect.add_argument("--gate-id", required=True)
    profile_collect.add_argument("--experiment-uid", required=True)
    profile_collect.add_argument("--namespace-id", required=True)
    profile_collect.add_argument("--environment-digest", required=True)
    profile_collect.add_argument(
        "--recipe", choices=PROFILE_RECIPE_IDS, required=True
    )
    profile_collect.set_defaults(handler=_profile_collect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except (OSError, ValueError, ControlledRuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
