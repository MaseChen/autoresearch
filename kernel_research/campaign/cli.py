"""Dependency-light management CLI for the local CampaignStore.

The CLI changes only local campaign state.  Its OJ surface can export an
offline package and record human-supplied feedback; it has no submit command.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Iterator, Sequence

from ..autorun.controller import gpu_lock
from ..autorun.models import ControllerConfig
from ..autorun.store import ControllerStore
from ..history import HistoryStore
from ..platform.canonical import canonical_sha256
from ..platform.identity import BaselineRef
from ..platform.profiles import ResearchNamespace
from .benchmark import (
    BenchmarkArm,
    BenchmarkCohortPlan,
    benchmark_plan_from_campaign_snapshot,
    bind_benchmark_campaign_snapshot,
    build_trusted_benchmark_report,
    derive_authoritative_benchmark_plan,
    execute_benchmark_campaign,
    require_executable_benchmark_plan,
)
from .controller_runner import (
    ResearchControllerRunner,
    legacy_campaign_snapshot,
    trusted_resume_doctor,
    trusted_child_reservation,
)
from .models import (
    BudgetAmount,
    CampaignMode,
    CampaignStatus,
)
from .oj import (
    export_oj_submission_package,
    nominate_exploratory_from_child,
    nominate_primary_from_child,
)
from .paths import (
    campaign_maintenance_fence,
    validate_production_campaign_database,
)
from .soak import SOAK_STAGES, SoakGate
from .soak_collector import SoakObservationCollector
from .store import CampaignStore, OJ_TYPES, OJ_VERDICTS
from .supervisor import CampaignSupervisor, ChildRunPlan


CLI_SCHEMA_VERSION = 1
MAX_JSON_INPUT_BYTES = 2 * 1024 * 1024

_PAUSED_STATUSES = (
    CampaignStatus.PAUSED_OPERATOR.value,
    CampaignStatus.PAUSED_BUDGET.value,
    CampaignStatus.PAUSED_UNKNOWN_OUTCOME.value,
    CampaignStatus.PAUSED_HARD_FAILURE.value,
    CampaignStatus.PAUSED_DATA_INTEGRITY.value,
)


def _print(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2))


def _success(command: str, result: Any) -> int:
    _print(
        {
            "schema_version": CLI_SCHEMA_VERSION,
            "command": command,
            "status": "SUCCESS",
            "result": result,
        }
    )
    return 0


def _print_error(exc: BaseException) -> None:
    print(
        json.dumps(
            {
                "schema_version": CLI_SCHEMA_VERSION,
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            sort_keys=True,
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )


def _json_object(path: str, field: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"{field} must identify a JSON file")
    data = source.read_bytes()
    if len(data) > MAX_JSON_INPUT_BYTES:
        raise ValueError(f"{field} exceeds the CLI JSON input limit")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} must contain strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return value


def _store(args: argparse.Namespace) -> CampaignStore:
    return CampaignStore(validate_production_campaign_database(args.database))


@contextmanager
def _lifecycle_store(args: argparse.Namespace) -> Iterator[CampaignStore]:
    """Open the production store while holding the runtime lifecycle fence.

    ``CampaignStore`` remains deliberately path agnostic.  The production CLI
    has already crossed the canonical database boundary, so it is the layer
    that derives the runtime root and serializes activation with deployment
    publication.
    """

    database = validate_production_campaign_database(args.database)
    runtime_root = database.parent.parent
    with campaign_maintenance_fence(runtime_root):
        with CampaignStore(database) as store:
            yield store


def _create(args: argparse.Namespace) -> int:
    if args.mode == CampaignMode.BENCHMARK.value:
        raise ValueError(
            "BENCHMARK Campaigns must be created through benchmark init"
        )
    baseline_ref = (
        None
        if args.initial_baseline_ref is None
        else _json_object(args.initial_baseline_ref, "initial_baseline_ref")
    )
    with _lifecycle_store(args) as store:
        result = store.create_campaign(
            campaign_id=args.campaign_id,
            namespace_id=args.namespace_id,
            mode=args.mode,
            snapshot=_json_object(args.snapshot, "snapshot"),
            budget_limit=BudgetAmount(
                candidates=args.max_candidates,
                wall_ms=args.max_wall_ms,
                gpu_ms=args.max_gpu_ms,
                tokens=args.max_tokens,
                cost_microusd=args.max_cost_microusd,
            ),
            initial_artifact_id=args.initial_artifact_id,
            initial_baseline_ref=baseline_ref,
            initial_policy_snapshot=_json_object(
                args.initial_policy_snapshot, "initial_policy_snapshot"
            ),
            allow_staged_lineage=args.allow_staged_lineage,
        )
    return _success("campaign.create", result)


def _start(args: argparse.Namespace) -> int:
    with _lifecycle_store(args) as store:
        result = store.start_campaign(args.campaign_id)
    return _success("campaign.start", result)


def _status(args: argparse.Namespace) -> int:
    with _store(args) as store:
        if args.campaign_id is None:
            result: Any = {
                "campaigns": store.list_campaigns(active_only=args.active_only),
                "integrity": store.integrity_check(),
            }
        else:
            result = {
                "campaign": store.get_campaign(args.campaign_id),
                "children": store.list_child_runs(args.campaign_id),
                "lineage": store.list_baseline_revisions(args.campaign_id),
                "integrity": store.integrity_check(),
            }
    return _success("campaign.status", result)


def _pause(args: argparse.Namespace) -> int:
    with _store(args) as store:
        result = store.pause_campaign(
            args.campaign_id, status=args.status, reason=args.reason
        )
    return _success("campaign.pause", result)


def _resume(args: argparse.Namespace) -> int:
    with _lifecycle_store(args) as store:
        campaign = store.get_campaign(args.campaign_id)
        evidence_digest = None
        if campaign["status"] in {
            CampaignStatus.PAUSED_HARD_FAILURE.value,
            CampaignStatus.PAUSED_UNKNOWN_OUTCOME.value,
        }:
            if args.config is None:
                raise ValueError(
                    "hard/unknown Campaign resume requires --config for the trusted doctor"
                )
            config = ControllerConfig.load(args.config)
            runtime_root = Path(args.database).parent.parent
            if config.controller_dir.parent != runtime_root:
                raise ValueError(
                    "resume config does not belong to the Campaign runtime"
                )
            # All paths that need both host locks use maintenance -> GPU.
            with gpu_lock(config.controller_dir / "gpu1.lock"):
                evidence = trusted_resume_doctor(
                    config,
                    campaign_store=store,
                    campaign=campaign,
                    resource_id=args.resource_id,
                )
            recorded = store.record_doctor_evidence(
                args.campaign_id, evidence=evidence
            )
            evidence_digest = recorded["digest"]
        result = store.resume_campaign(
            args.campaign_id,
            doctor_evidence_digest=evidence_digest,
        )
    return _success("campaign.resume", result)


def _child_list(args: argparse.Namespace) -> int:
    with _store(args) as store:
        result = store.list_child_runs(args.campaign_id)
    return _success("campaign.child.list", result)


def _campaign_supervisor(
    args: argparse.Namespace, store: CampaignStore
) -> CampaignSupervisor:
    config = ControllerConfig.load(args.config)
    runner = ResearchControllerRunner(
        config,
        campaign_store=store,
        fence_lookup=store.get_active_resource_lease,
    )
    return CampaignSupervisor(
        store,
        runner,
        resource_id=args.resource_id,
        lease_ttl_seconds=args.lease_ttl_seconds,
    )


def _child_execute(args: argparse.Namespace) -> int:
    proposer_profile = _json_object(
        args.proposer_profile, "proposer_profile"
    )
    with _store(args) as store:
        if store.get_campaign(args.campaign_id)["mode"] == CampaignMode.BENCHMARK.value:
            raise ValueError(
                "BENCHMARK children must be executed through benchmark execute"
            )
        remaining = BudgetAmount.from_mapping(
            store.budget_status(args.campaign_id)["remaining"]
        )
        plan = ChildRunPlan(
            controller_run_id=args.controller_run_id,
            proposer_profile=proposer_profile,
            reservation=trusted_child_reservation(
                proposer_profile=proposer_profile,
                campaign_remaining=remaining,
                max_candidates=args.max_candidates,
                max_wall_seconds=args.max_wall_seconds,
                max_consecutive_failures=args.max_consecutive_failures,
            ),
            max_candidates=args.max_candidates,
            max_wall_seconds=args.max_wall_seconds,
            max_consecutive_failures=args.max_consecutive_failures,
        )
        result = _campaign_supervisor(args, store).execute_child(
            args.campaign_id, plan
        )
    return _success("campaign.child.execute", result)


def _lineage_list(args: argparse.Namespace) -> int:
    with _store(args) as store:
        result = store.list_baseline_revisions(args.campaign_id)
    return _success("campaign.lineage.list", result)


def _lineage_advance(args: argparse.Namespace) -> int:
    with _store(args) as store:
        result = _campaign_supervisor(args, store).advance_staged_lineage(
            args.campaign_id,
            child_id=args.child_id,
            idempotency_key=args.idempotency_key,
        )
    return _success("campaign.lineage.advance", result)


def _benchmark_campaign_snapshot(
    plan: BenchmarkCohortPlan,
    feedback_snapshot: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    base = legacy_campaign_snapshot(
        mode=CampaignMode.BENCHMARK,
        cohort_id=plan.cohort_id,
        history_cutoff=plan.history_cutoff,
        prompt_protocol_digest=plan.prompt_protocol_digest,
        feedback_snapshot_digest=plan.feedback_snapshot_digest,
        namespace=plan.namespace,
    )
    return bind_benchmark_campaign_snapshot(
        base, plan, feedback_snapshot=feedback_snapshot
    )


def _benchmark_create(args: argparse.Namespace) -> int:
    arms_value = _json_object(args.arms, "arms")
    if set(arms_value) != {"arms"} or not isinstance(
        arms_value["arms"], list
    ):
        raise ValueError("arms must contain exactly one arms list")
    plan = BenchmarkCohortPlan(
        cohort_id=args.cohort_id,
        namespace=ResearchNamespace.from_value(
            _json_object(args.namespace, "namespace")
        ),
        baseline=BaselineRef.from_value(
            _json_object(args.baseline, "baseline")
        ),
        history_cutoff=args.history_cutoff,
        prompt_protocol_digest=args.prompt_protocol_digest,
        feedback_snapshot_digest=args.feedback_snapshot_digest,
        arms=tuple(
            BenchmarkArm.from_value(value) for value in arms_value["arms"]
        ),
        repetitions=args.repetitions,
        max_candidates=args.max_candidates,
        max_wall_seconds=args.max_wall_seconds,
        max_consecutive_failures=args.max_consecutive_failures,
    )
    return _success(
        "campaign.benchmark.create",
        {
            "authority": "DRAFT",
            "plan": plan.snapshot,
            "schedule": [item.to_dict() for item in plan.schedule()],
        },
    )


def _benchmark_runtime_databases(
    args: argparse.Namespace,
) -> tuple[ControllerConfig, Path, Path]:
    config = ControllerConfig.load(args.config)
    runtime_root = Path(args.database).parent.parent
    if (
        config.controller_dir.parent != runtime_root
        or config.state_dir.parent != runtime_root
    ):
        raise ValueError(
            "benchmark config does not belong to the Campaign runtime"
        )
    controller_database = config.controller_dir / "controller.sqlite3"
    if not controller_database.is_file():
        raise ValueError("Controller database does not exist")
    history_database = config.state_dir / "history.sqlite3"
    if not history_database.is_file():
        raise ValueError("History database does not exist")
    return config, controller_database, history_database


def _benchmark_init(args: argparse.Namespace) -> int:
    draft = BenchmarkCohortPlan.from_snapshot(
        _json_object(args.plan, "benchmark plan")
    )
    require_executable_benchmark_plan(draft)
    config, controller_database, history_database = (
        _benchmark_runtime_databases(args)
    )
    policy = _json_object(args.initial_policy_snapshot, "initial_policy_snapshot")
    budget = BudgetAmount(
        candidates=args.max_candidates,
        wall_ms=args.max_wall_ms,
        gpu_ms=args.max_gpu_ms,
        tokens=args.max_tokens,
        cost_microusd=args.max_cost_microusd,
    )
    scheduled = len(draft.schedule())
    token_ceiling = sum(
        trusted_child_reservation(
            proposer_profile=assignment.proposer_profile.to_dict(),
            campaign_remaining=BudgetAmount(
                candidates=draft.max_candidates,
                wall_ms=draft.max_wall_seconds * 1000,
                gpu_ms=draft.max_wall_seconds * 1000,
                tokens=10**18,
                cost_microusd=1,
            ),
            max_candidates=draft.max_candidates,
            max_wall_seconds=draft.max_wall_seconds,
            max_consecutive_failures=draft.max_consecutive_failures,
        ).tokens
        for assignment in draft.schedule()
    )
    predictable_minimums = {
        "candidates": scheduled * draft.max_candidates,
        "wall_ms": scheduled * draft.max_wall_seconds * 1000,
        "gpu_ms": scheduled * draft.max_wall_seconds * 1000,
        "tokens": token_ceiling,
    }
    if any(
        getattr(budget, name) < minimum
        for name, minimum in predictable_minimums.items()
    ) or budget.cost_microusd <= 0:
        raise ValueError(
            "benchmark Campaign budget cannot fund its frozen child schedule"
        )
    with (
        _lifecycle_store(args) as store,
        ControllerStore(controller_database) as controller,
        HistoryStore(history_database, state_dir=config.state_dir) as history,
    ):
        try:
            campaign = store.get_campaign(args.campaign_id)
        except KeyError:
            plan, feedback_snapshot = derive_authoritative_benchmark_plan(
                draft, controller, history
            )
            snapshot = _benchmark_campaign_snapshot(plan, feedback_snapshot)
            campaign = store.create_campaign(
                campaign_id=args.campaign_id,
                namespace_id=plan.namespace.namespace_id,
                mode=CampaignMode.BENCHMARK.value,
                snapshot=snapshot,
                budget_limit=budget,
                initial_baseline_ref=plan.baseline,
                initial_policy_snapshot=policy,
                allow_staged_lineage=False,
            )
        else:
            snapshot_value = campaign.get("snapshot")
            if not isinstance(snapshot_value, dict):
                raise ValueError("benchmark Campaign has no frozen snapshot")
            plan = benchmark_plan_from_campaign_snapshot(snapshot_value)
            require_executable_benchmark_plan(plan)
            expected = replace(
                draft,
                history_cutoff=plan.history_cutoff,
                feedback_snapshot_digest=plan.feedback_snapshot_digest,
            )
            if expected != plan:
                raise ValueError(
                    "campaign_id belongs to another benchmark cohort draft"
                )
            snapshot = snapshot_value
        expected = {
            "namespace_id": plan.namespace.namespace_id,
            "mode": CampaignMode.BENCHMARK.value,
            "snapshot": snapshot,
            "allow_staged_lineage": False,
        }
        if any(campaign.get(name) != value for name, value in expected.items()):
            raise ValueError("campaign_id belongs to another frozen Campaign")
        if campaign["snapshot_digest"] != canonical_sha256(snapshot):
            raise ValueError("benchmark Campaign snapshot digest is inconsistent")
        if campaign["budget"]["limit"] != budget.to_dict():
            raise ValueError("benchmark Campaign budget differs from init request")
        revisions = store.list_baseline_revisions(args.campaign_id)
        if (
            len(revisions) != 1
            or revisions[0].get("baseline_ref") != plan.baseline.to_dict()
            or revisions[0].get("policy_snapshot") != policy
        ):
            raise ValueError("benchmark Campaign seed differs from init request")
        if campaign["status"] == CampaignStatus.CREATED.value:
            campaign = store.start_campaign(args.campaign_id)
        result = {
            "campaign": campaign,
            "authority": "AUTHORITATIVE",
            "benchmark_plan": plan.snapshot,
            "campaign_snapshot": snapshot,
            "benchmark_plan_digest": plan.snapshot["snapshot_digest"],
            "scheduled_children": scheduled,
            "lineage_advanced": False,
        }
    return _success("campaign.benchmark.init", result)


def _benchmark_execute(args: argparse.Namespace) -> int:
    with _store(args) as store:
        result = execute_benchmark_campaign(
            store,
            _campaign_supervisor(args, store),
            campaign_id=args.campaign_id,
            reservation_builder=trusted_child_reservation,
        )
    return _success("campaign.benchmark.execute", result)


def _benchmark_report(args: argparse.Namespace) -> int:
    config, controller_database, history_database = (
        _benchmark_runtime_databases(args)
    )
    with (
        _store(args) as store,
        ControllerStore(controller_database) as controller,
        HistoryStore(history_database, state_dir=config.state_dir) as history,
    ):
        campaign = store.get_campaign(args.campaign_id)
        snapshot = campaign.get("snapshot")
        if not isinstance(snapshot, dict):
            raise ValueError("benchmark Campaign has no frozen snapshot")
        benchmark_plan_from_campaign_snapshot(snapshot)
        report = build_trusted_benchmark_report(
            store, controller, history, campaign_id=args.campaign_id
        )
    return _success("campaign.benchmark.report", report)


def _soak_start(args: argparse.Namespace) -> int:
    config = ControllerConfig.load(args.config)
    with _store(args) as store:
        collector = SoakObservationCollector(
            config, campaign_database=store.path
        )
        result = SoakGate(
            store, gate_id=args.gate_id, collector=collector
        ).start(stage=args.stage)
    return _success("campaign.soak.start", result)


def _soak_heartbeat(args: argparse.Namespace) -> int:
    config = ControllerConfig.load(args.config)
    with _store(args) as store:
        collector = SoakObservationCollector(
            config, campaign_database=store.path
        )
        result = SoakGate(
            store, gate_id=args.gate_id, collector=collector
        ).heartbeat()
    return _success("campaign.soak.heartbeat", result)


def _soak_status(args: argparse.Namespace) -> int:
    config = ControllerConfig.load(args.config)
    with _store(args) as store:
        collector = SoakObservationCollector(
            config, campaign_database=store.path
        )
        result = SoakGate(
            store, gate_id=args.gate_id, collector=collector
        ).status()
    return _success("campaign.soak.status", result)


def _oj_nominate(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir).resolve()
    history_database = state_dir / "history.sqlite3"
    if not history_database.is_file():
        raise ValueError("state_dir does not contain history.sqlite3")
    with _store(args) as store, HistoryStore(
        history_database, state_dir=state_dir
    ) as history:
        if args.nomination_type == "PRIMARY_SUBMISSION":
            if args.artifact_id is not None:
                raise ValueError(
                    "PRIMARY_SUBMISSION derives artifact_id from promotion evidence"
                )
            result = nominate_primary_from_child(
                store,
                history,
                campaign_id=args.campaign_id,
                child_id=args.child_id,
            )
        else:
            if args.artifact_id is None:
                raise ValueError(
                    "EXPLORATORY_SUBMISSION requires --artifact-id"
                )
            result = nominate_exploratory_from_child(
                store,
                history,
                campaign_id=args.campaign_id,
                child_id=args.child_id,
                artifact_id=args.artifact_id,
            )
    return _success("campaign.oj.nominate", result)


def _oj_export(args: argparse.Namespace) -> int:
    state_dir = Path(args.state_dir).resolve()
    history_database = state_dir / "history.sqlite3"
    if not history_database.is_file():
        raise ValueError("state_dir does not contain history.sqlite3")
    with _store(args) as store, HistoryStore(
        history_database, state_dir=state_dir
    ) as history:
        result = export_oj_submission_package(
            store,
            args.nomination_id,
            destination_root=args.output_root,
            artifact_reader=history.read_candidate_artifact,
            package_name=args.package_name,
        )
    return _success("campaign.oj.export", result)


def _oj_show(args: argparse.Namespace) -> int:
    with _store(args) as store:
        result = store.get_oj_nomination(args.nomination_id)
    return _success("campaign.oj.show", result)


def _oj_feedback(args: argparse.Namespace) -> int:
    with _store(args) as store:
        result = store.record_oj_feedback(
            args.nomination_id,
            submission_id=args.submission_id,
            verdict=args.verdict,
            score=args.score,
            note=args.note,
        )
    return _success("campaign.oj.feedback", result)


def _oj_agent_feedback(args: argparse.Namespace) -> int:
    with _store(args) as store:
        result = store.oj_agent_feedback(args.nomination_id)
    return _success("campaign.oj.agent-feedback", result)


def _database_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--database", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kernel-autoresearch-campaign",
        description="Manage local bounded campaigns and manual OJ evidence.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create")
    _database_argument(create)
    create.add_argument("--campaign-id", required=True)
    create.add_argument("--namespace-id", required=True)
    create.add_argument(
        "--mode", choices=tuple(mode.value for mode in CampaignMode), required=True
    )
    create.add_argument("--snapshot", required=True)
    seed = create.add_mutually_exclusive_group(required=True)
    seed.add_argument("--initial-baseline-ref")
    # One compatibility cycle for V1 non-advancing callers.  Store policy
    # prevents this LEGACY_UNKNOWN form from enabling staged lineage.
    seed.add_argument("--initial-artifact-id", help=argparse.SUPPRESS)
    create.add_argument("--initial-policy-snapshot", required=True)
    create.add_argument("--max-candidates", type=int, default=0)
    create.add_argument("--max-wall-ms", type=int, default=0)
    create.add_argument("--max-gpu-ms", type=int, default=0)
    create.add_argument("--max-tokens", type=int, default=0)
    create.add_argument("--max-cost-microusd", type=int, default=0)
    create.add_argument("--allow-staged-lineage", action="store_true")
    create.set_defaults(handler=_create)

    start = commands.add_parser("start")
    _database_argument(start)
    start.add_argument("--campaign-id", required=True)
    start.set_defaults(handler=_start)

    status = commands.add_parser("status")
    _database_argument(status)
    status.add_argument("--campaign-id")
    status.add_argument("--active-only", action="store_true")
    status.set_defaults(handler=_status)

    pause = commands.add_parser("pause")
    _database_argument(pause)
    pause.add_argument("--campaign-id", required=True)
    pause.add_argument("--status", choices=_PAUSED_STATUSES, default="PAUSED_OPERATOR")
    pause.add_argument("--reason", required=True)
    pause.set_defaults(handler=_pause)

    resume = commands.add_parser("resume")
    _database_argument(resume)
    resume.add_argument("--campaign-id", required=True)
    resume.add_argument("--config")
    resume.add_argument("--resource-id", default="gpu1")
    resume.set_defaults(handler=_resume)

    child = commands.add_parser("child")
    child_commands = child.add_subparsers(dest="child_command", required=True)
    child_list = child_commands.add_parser("list")
    _database_argument(child_list)
    child_list.add_argument("--campaign-id", required=True)
    child_list.set_defaults(handler=_child_list)

    child_execute = child_commands.add_parser("execute")
    _database_argument(child_execute)
    child_execute.add_argument("--config", required=True)
    child_execute.add_argument("--campaign-id", required=True)
    child_execute.add_argument("--controller-run-id", required=True)
    child_execute.add_argument("--proposer-profile", required=True)
    child_execute.add_argument("--resource-id", default="gpu1")
    child_execute.add_argument("--lease-ttl-seconds", type=float, default=21900)
    child_execute.add_argument("--max-candidates", type=int, default=5)
    child_execute.add_argument("--max-wall-seconds", type=int, default=21600)
    child_execute.add_argument(
        "--max-consecutive-failures", type=int, default=3
    )
    child_execute.set_defaults(handler=_child_execute)

    lineage = commands.add_parser("lineage")
    lineage_commands = lineage.add_subparsers(
        dest="lineage_command", required=True
    )
    lineage_list = lineage_commands.add_parser("list")
    _database_argument(lineage_list)
    lineage_list.add_argument("--campaign-id", required=True)
    lineage_list.set_defaults(handler=_lineage_list)

    lineage_advance = lineage_commands.add_parser("advance")
    _database_argument(lineage_advance)
    lineage_advance.add_argument("--config", required=True)
    lineage_advance.add_argument("--campaign-id", required=True)
    lineage_advance.add_argument("--child-id", type=int, required=True)
    lineage_advance.add_argument("--resource-id", default="gpu1")
    lineage_advance.add_argument(
        "--lease-ttl-seconds", type=float, default=21900
    )
    lineage_advance.add_argument("--idempotency-key", required=True)
    lineage_advance.set_defaults(handler=_lineage_advance)

    benchmark = commands.add_parser("benchmark")
    benchmark_commands = benchmark.add_subparsers(
        dest="benchmark_command", required=True
    )
    benchmark_create = benchmark_commands.add_parser("create")
    benchmark_create.add_argument("--cohort-id", required=True)
    benchmark_create.add_argument("--namespace", required=True)
    benchmark_create.add_argument("--baseline", required=True)
    benchmark_create.add_argument("--history-cutoff", type=int, required=True)
    benchmark_create.add_argument("--prompt-protocol-digest", required=True)
    benchmark_create.add_argument("--feedback-snapshot-digest", required=True)
    benchmark_create.add_argument("--arms", required=True)
    benchmark_create.add_argument("--repetitions", type=int, required=True)
    benchmark_create.add_argument("--max-candidates", type=int, default=5)
    benchmark_create.add_argument("--max-wall-seconds", type=int, default=21600)
    benchmark_create.add_argument(
        "--max-consecutive-failures", type=int, default=3
    )
    benchmark_create.set_defaults(handler=_benchmark_create)

    benchmark_init = benchmark_commands.add_parser("init")
    _database_argument(benchmark_init)
    benchmark_init.add_argument("--config", required=True)
    benchmark_init.add_argument("--campaign-id", required=True)
    benchmark_init.add_argument("--plan", required=True)
    benchmark_init.add_argument("--initial-policy-snapshot", required=True)
    benchmark_init.add_argument("--max-candidates", type=int, required=True)
    benchmark_init.add_argument("--max-wall-ms", type=int, required=True)
    benchmark_init.add_argument("--max-gpu-ms", type=int, required=True)
    benchmark_init.add_argument("--max-tokens", type=int, required=True)
    benchmark_init.add_argument("--max-cost-microusd", type=int, required=True)
    benchmark_init.set_defaults(handler=_benchmark_init)

    benchmark_execute = benchmark_commands.add_parser("execute")
    _database_argument(benchmark_execute)
    benchmark_execute.add_argument("--config", required=True)
    benchmark_execute.add_argument("--campaign-id", required=True)
    benchmark_execute.add_argument("--resource-id", default="gpu1")
    benchmark_execute.add_argument(
        "--lease-ttl-seconds", type=float, default=21900
    )
    benchmark_execute.set_defaults(handler=_benchmark_execute)

    benchmark_report = benchmark_commands.add_parser("report")
    _database_argument(benchmark_report)
    benchmark_report.add_argument("--config", required=True)
    benchmark_report.add_argument("--campaign-id", required=True)
    benchmark_report.set_defaults(handler=_benchmark_report)

    soak = commands.add_parser("soak")
    soak_commands = soak.add_subparsers(dest="soak_command", required=True)
    soak_start = soak_commands.add_parser("start")
    _database_argument(soak_start)
    soak_start.add_argument("--config", required=True)
    soak_start.add_argument("--gate-id", required=True)
    soak_start.add_argument(
        "--stage",
        choices=tuple(stage.stage_id for stage in SOAK_STAGES),
    )
    soak_start.set_defaults(handler=_soak_start)

    soak_heartbeat = soak_commands.add_parser("heartbeat")
    _database_argument(soak_heartbeat)
    soak_heartbeat.add_argument("--config", required=True)
    soak_heartbeat.add_argument("--gate-id", required=True)
    soak_heartbeat.set_defaults(handler=_soak_heartbeat)

    soak_status = soak_commands.add_parser("status")
    _database_argument(soak_status)
    soak_status.add_argument("--config", required=True)
    soak_status.add_argument("--gate-id", required=True)
    soak_status.set_defaults(handler=_soak_status)

    oj = commands.add_parser("oj")
    oj_commands = oj.add_subparsers(dest="oj_command", required=True)
    nominate = oj_commands.add_parser("nominate")
    _database_argument(nominate)
    nominate.add_argument("--campaign-id", required=True)
    nominate.add_argument(
        "--nomination-type",
        choices=tuple(sorted(OJ_TYPES)),
        required=True,
    )
    nominate.add_argument("--child-id", type=int, required=True)
    nominate.add_argument("--state-dir", required=True)
    nominate.add_argument("--artifact-id")
    nominate.set_defaults(handler=_oj_nominate)

    export = oj_commands.add_parser("export")
    _database_argument(export)
    export.add_argument("--nomination-id", required=True)
    export.add_argument("--state-dir", required=True)
    export.add_argument("--output-root", required=True)
    export.add_argument("--package-name")
    export.set_defaults(handler=_oj_export)

    show = oj_commands.add_parser("show")
    _database_argument(show)
    show.add_argument("--nomination-id", required=True)
    show.set_defaults(handler=_oj_show)

    feedback = oj_commands.add_parser("feedback")
    _database_argument(feedback)
    feedback.add_argument("--nomination-id", required=True)
    feedback.add_argument("--submission-id", required=True)
    feedback.add_argument(
        "--verdict", choices=tuple(sorted(OJ_VERDICTS)), required=True
    )
    feedback.add_argument("--score", type=float)
    feedback.add_argument("--note", default="")
    feedback.set_defaults(handler=_oj_feedback)

    agent_feedback = oj_commands.add_parser("agent-feedback")
    _database_argument(agent_feedback)
    agent_feedback.add_argument("--nomination-id", required=True)
    agent_feedback.set_defaults(handler=_oj_agent_feedback)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        # Every production command that touches Campaign state crosses the
        # same path boundary before reading config, History, or input files.
        # ``benchmark create`` is the sole planning-only command; init,
        # execute and report cross this same production database boundary.
        if hasattr(args, "database"):
            args.database = str(
                validate_production_campaign_database(args.database)
            )
        return int(args.handler(args))
    except KeyboardInterrupt:
        _print_error(KeyboardInterrupt("operator interrupt"))
        return 130
    except (OSError, KeyError, TypeError, ValueError, sqlite3.DatabaseError) as exc:
        _print_error(exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CLI_SCHEMA_VERSION", "build_parser", "main"]
