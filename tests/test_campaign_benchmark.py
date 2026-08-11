from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from kernel_research.campaign import (
    BenchmarkArm,
    BenchmarkCohortPlan,
    BenchmarkObservation,
    BudgetAmount,
    CampaignStore,
    CampaignSupervisor,
    CURRENT_PROMPT_PROTOCOL_DIGEST,
    BoundedRunResult,
    RunnerOutcome,
    benchmark_controller_run_id,
    bind_benchmark_campaign_snapshot,
    build_benchmark_report,
    build_trusted_benchmark_report,
    child_budget_idempotency_key,
    execute_benchmark_campaign,
    filter_feedback_at_cutoff,
)
from kernel_research.autorun.store import ControllerStore
from kernel_research.campaign.cli import main as campaign_main
from kernel_research.campaign.controller_runner import legacy_campaign_snapshot
from kernel_research.history import HistoryStore
from kernel_research.platform.canonical import canonical_sha256
from kernel_research.platform.identity import (
    BaselineRef,
    ExecutionEnvironmentDigest,
)
from kernel_research.platform.profiles import (
    BUILTIN_PROFILE_REGISTRY,
    CURRENT_RESEARCH_NAMESPACE,
    LEGACY_RESEARCH_NAMESPACE,
)


def proposer(profile_id: str):
    return BUILTIN_PROFILE_REGISTRY.get(
        kind="proposer", profile_id=profile_id, revision="v1"
    ).ref


def write_runtime_config(root: Path) -> Path:
    for name in ("repo", "state", "controller", "checkpoints", "cache"):
        (root / name).mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1,
        "repository_dir": str(root / "repo"),
        "state_dir": str(root / "state"),
        "controller_dir": str(root / "controller"),
        "checkpoint_dir": str(root / "checkpoints"),
        "docker_binary": "/bin/true",
        "proposer_image": "local/proposer@sha256:" + "1" * 64,
        "evaluator_image": "local/evaluator@sha256:" + "2" * 64,
        "deepseek_key_file": str(root / "secret"),
        "gpu_devices": [
            "/dev/mxcd",
            "/dev/dri/card2",
            "/dev/dri/renderD129",
        ],
        "evaluator_cache_dir": str(root / "cache"),
        "expected_git_commit": "8" * 40,
        "expected_kernel_hash": "a" * 64,
        "acknowledge_gpu_passthrough_risk": True,
    }
    path = root / "autorun.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


class CampaignBenchmarkTests(unittest.TestCase):
    def plan(self) -> BenchmarkCohortPlan:
        namespace = LEGACY_RESEARCH_NAMESPACE
        return BenchmarkCohortPlan(
            cohort_id="cohort-1",
            namespace=namespace,
            baseline=BaselineRef.create(
                namespace=namespace,
                artifact_id="source-sha256-v1:" + "a" * 64,
                source="deployment",
                revision="deployment-1",
            ),
            history_cutoff=42,
            prompt_protocol_digest=CURRENT_PROMPT_PROTOCOL_DIGEST,
            feedback_snapshot_digest=canonical_sha256([]),
            arms=(
                BenchmarkArm("opencode", proposer("opencode-deepseek-v4-pro")),
                BenchmarkArm("direct", proposer("direct-api-deepseek-v4-pro")),
                BenchmarkArm("pi", proposer("pi-deepseek-v4-pro")),
            ),
            repetitions=4,
        )

    def active_plan(self) -> BenchmarkCohortPlan:
        namespace = LEGACY_RESEARCH_NAMESPACE
        return BenchmarkCohortPlan(
            cohort_id="trusted-cohort",
            namespace=namespace,
            baseline=BaselineRef.create(
                namespace=namespace,
                artifact_id="source-sha256-v1:" + "a" * 64,
                source="campaign",
                revision="trusted-benchmark-seed",
            ),
            history_cutoff=42,
            prompt_protocol_digest=CURRENT_PROMPT_PROTOCOL_DIGEST,
            feedback_snapshot_digest=canonical_sha256([]),
            arms=(
                BenchmarkArm("pro", proposer("opencode-deepseek-v4-pro")),
                BenchmarkArm("flash", proposer("opencode-deepseek-v4-flash")),
            ),
            repetitions=1,
            max_candidates=1,
            max_wall_seconds=1,
            max_consecutive_failures=1,
        )

    @staticmethod
    def campaign_snapshot(plan: BenchmarkCohortPlan) -> dict:
        return bind_benchmark_campaign_snapshot(
            legacy_campaign_snapshot(
                mode="BENCHMARK",
                cohort_id=plan.cohort_id,
                history_cutoff=plan.history_cutoff,
                prompt_protocol_digest=plan.prompt_protocol_digest,
                feedback_snapshot_digest=plan.feedback_snapshot_digest,
                namespace=plan.namespace,
            ),
            plan,
            feedback_snapshot=[],
        )

    def test_schedule_is_deterministic_interleaved_and_frozen(self) -> None:
        plan = self.plan()
        schedule = plan.schedule()
        self.assertEqual(schedule, plan.schedule())
        self.assertEqual(len(schedule), 12)
        for round_index in range(4):
            arms = {
                item.arm_id
                for item in schedule
                if item.round_index == round_index
            }
            self.assertEqual(arms, {"opencode", "direct", "pi"})
        self.assertEqual(plan.snapshot["mode"], "BENCHMARK")
        self.assertTrue(plan.snapshot["snapshot_digest"].startswith("sha256:"))

    def test_report_ranks_aggregate_behavior_not_a_single_candidate(self) -> None:
        plan = self.plan()
        observations = [
            BenchmarkObservation(
                arm_id=arm,
                protocol_success=True,
                valid_candidates=1,
                full_reached=1 if arm != "pi" else 0,
                gpu_ms=100,
                tokens=200,
                cost_microusd=300,
                promotions=1 if arm == "direct" else 0,
            )
            for arm in ("opencode", "direct", "pi")
        ]
        report = build_benchmark_report(plan, observations)
        self.assertFalse(report["single_best_candidate_ranking"])
        self.assertEqual(report["ranking_basis"], "aggregate_protocol_metrics")
        self.assertEqual(len(report["arm_metrics"]), 3)
        direct = next(
            row for row in report["arm_metrics"] if row["arm_id"] == "direct"
        )
        self.assertEqual(direct["promotion_rate_per_valid_candidate"], 1.0)

    def test_cutoff_and_cross_namespace_baseline_fail_closed(self) -> None:
        self.assertEqual(
            [row["id"] for row in filter_feedback_at_cutoff(
                [{"id": 1}, {"id": 43}, {"id": 42}], history_cutoff=42
            )],
            [1, 42],
        )
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, "another namespace"):
            BenchmarkCohortPlan(
                cohort_id="bad",
                namespace=plan.namespace,
                baseline=BaselineRef.create(
                    namespace="sha256:" + "d" * 64,
                    artifact_id="source-sha256-v1:" + "a" * 64,
                    source="deployment",
                    revision="bad",
                ),
                history_cutoff=1,
                prompt_protocol_digest="sha256:" + "b" * 64,
                feedback_snapshot_digest="sha256:" + "c" * 64,
                arms=plan.arms,
                repetitions=1,
            )

    def test_snapshot_round_trip_rejects_tampering(self) -> None:
        plan = self.plan()
        self.assertEqual(
            BenchmarkCohortPlan.from_snapshot(plan.snapshot), plan
        )
        tampered = dict(plan.snapshot)
        tampered["cohort_id"] = "different"
        with self.assertRaisesRegex(ValueError, "digest"):
            BenchmarkCohortPlan.from_snapshot(tampered)

    def test_benchmark_cli_creates_exact_snapshot_without_raw_report(
        self,
    ) -> None:
        plan = self.plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write(name: str, value: object) -> Path:
                path = root / name
                path.write_text(json.dumps(value), encoding="utf-8")
                return path

            namespace = write("namespace.json", plan.namespace.to_dict())
            baseline = write("baseline.json", plan.baseline.to_dict())
            arms = write(
                "arms.json", {"arms": [arm.to_dict() for arm in plan.arms]}
            )
            output = io.StringIO()
            with redirect_stdout(output):
                code = campaign_main(
                    [
                        "benchmark",
                        "create",
                        "--cohort-id",
                        plan.cohort_id,
                        "--namespace",
                        str(namespace),
                        "--baseline",
                        str(baseline),
                        "--history-cutoff",
                        str(plan.history_cutoff),
                        "--prompt-protocol-digest",
                        plan.prompt_protocol_digest,
                        "--feedback-snapshot-digest",
                        plan.feedback_snapshot_digest,
                        "--arms",
                        str(arms),
                        "--repetitions",
                        str(plan.repetitions),
                    ]
                )
            self.assertEqual(code, 0)
            created = json.loads(output.getvalue())["result"]
            self.assertEqual(len(created["schedule"]), 12)
            self.assertEqual(created["plan"], plan.snapshot)
            self.assertEqual(created["authority"], "DRAFT")
            self.assertNotIn("campaign_snapshot", created)

            stderr = io.StringIO()
            with redirect_stderr(stderr), self.assertRaises(SystemExit):
                campaign_main(
                    [
                        "benchmark",
                        "report",
                        "--observations",
                        str(write("observations.json", {"observations": []})),
                    ]
                )

    def test_benchmark_init_is_idempotent_and_rejects_inactive_harnesses(self) -> None:
        active = self.active_plan()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            database = root / "campaign" / "campaign.sqlite3"
            config_path = write_runtime_config(root)
            with ControllerStore(
                root / "controller" / "controller.sqlite3"
            ), HistoryStore(
                root / "state" / "history.sqlite3",
                state_dir=root / "state",
            ):
                pass

            def write(name: str, value: object) -> Path:
                path = root / name
                path.write_text(json.dumps(value), encoding="utf-8")
                return path

            active_path = write("active.json", active.snapshot)
            inactive = BenchmarkCohortPlan(
                cohort_id="inactive-cohort",
                namespace=active.namespace,
                baseline=active.baseline,
                history_cutoff=active.history_cutoff,
                prompt_protocol_digest=active.prompt_protocol_digest,
                feedback_snapshot_digest=active.feedback_snapshot_digest,
                arms=(
                    active.arms[0],
                    BenchmarkArm(
                        "direct", proposer("direct-api-deepseek-v4-pro")
                    ),
                ),
                repetitions=1,
                max_candidates=1,
                max_wall_seconds=1,
                max_consecutive_failures=1,
            )
            inactive_path = write("inactive.json", inactive.snapshot)
            policy_path = write("policy.json", {"source": "trusted-test"})
            base = [
                "benchmark",
                "init",
                "--database",
                str(database),
                "--config",
                str(config_path),
                "--campaign-id",
                "benchmark-1",
                "--initial-policy-snapshot",
                str(policy_path),
                "--max-candidates",
                "2",
                "--max-wall-ms",
                "2000",
                "--max-gpu-ms",
                "2000",
                "--max-tokens",
                "10000000",
                "--max-cost-microusd",
                "200",
            ]
            for _ in range(2):
                output = io.StringIO()
                with redirect_stdout(output):
                    code = campaign_main([*base, "--plan", str(active_path)])
                self.assertEqual(code, 0)
                result = json.loads(output.getvalue())["result"]
                self.assertEqual(result["campaign"]["status"], "RUNNING")
                self.assertFalse(result["lineage_advanced"])
                self.assertEqual(result["authority"], "AUTHORITATIVE")
                self.assertEqual(result["benchmark_plan"]["history_cutoff"], 0)
                self.assertEqual(
                    result["benchmark_plan"]["feedback_snapshot_digest"],
                    canonical_sha256([]),
                )

            stderr = io.StringIO()
            with redirect_stderr(stderr):
                code = campaign_main(
                    [
                        *base[:-1],
                        "201",
                        "--campaign-id",
                        "benchmark-inactive",
                        "--plan",
                        str(inactive_path),
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("active OpenCode", stderr.getvalue())

    def test_init_derives_real_cutoff_and_complete_holdout_safe_feedback(self) -> None:
        namespace = CURRENT_RESEARCH_NAMESPACE
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            database = root / "campaign" / "campaign.sqlite3"
            config_path = write_runtime_config(root)
            result_value = {
                "status": "SUCCESS",
                "aggregate_score": 1.1,
                "cases": [
                    {
                        "case_id": "quick_shadow_tiles_127_n2",
                        "status": "SUCCESS",
                        "matched_ratio": 1.0,
                        "p50_us": 9.0,
                    },
                    {
                        "case_id": "quick_shadow_tiles_128_n1",
                        "status": "SUCCESS",
                        "matched_ratio": 1.0,
                        "p50_us": 8.0,
                    },
                ],
                "promotion": {
                    "phase": "quick",
                    "decision": {
                        "per_case_speedups": {
                            "quick_shadow_tiles_127_n2": 1.1,
                            "quick_shadow_tiles_128_n1": 1.2,
                        },
                        "confirmation_case_speedups": {
                            "quick_shadow_tiles_127_n2": 1.05,
                            "quick_shadow_tiles_128_n1": 1.15,
                        },
                    },
                },
            }
            source = "def candidate():\n    return 1\n"
            with HistoryStore(
                root / "state" / "history.sqlite3",
                state_dir=root / "state",
            ) as history:
                history.ensure_namespace(
                    namespace.namespace_id, namespace.to_dict()
                )
                record = history.record_experiment(
                    candidate_source=source,
                    status="SUCCESS",
                    backend="c500",
                    suite="quick",
                    namespace_id=namespace.namespace_id,
                    experiment_uid="feedback-experiment-1",
                    promotable=True,
                    aggregate_score=1.1,
                    result=result_value,
                )
            environment = ExecutionEnvironmentDigest.resolved(
                evaluator_image_digest="sha256:" + "1" * 64,
                toolchain_digest="sha256:" + "2" * 64,
                framework_digest="sha256:" + "3" * 64,
                operator_abi_digest="sha256:" + "4" * 64,
                build_flags_digest="sha256:" + "5" * 64,
            )
            baseline = BaselineRef.create(
                namespace=namespace,
                artifact_id=record.artifact_id,
                source="campaign",
                revision="source-staged-revision",
                execution_environment=environment,
            )
            with ControllerStore(
                root / "controller" / "controller.sqlite3"
            ) as controller:
                controller.create_run(
                    run_id="prior-scientific-run",
                    deadline_epoch=100.0,
                    config={},
                    initial_best_hash=record.candidate_hash,
                    namespace_id=namespace.namespace_id,
                    resolved_config_digest="fixture",
                    workflow_snapshot={"namespace": namespace.to_dict()},
                    baseline_ref=baseline.to_dict(),
                    history_cutoff=record.id,
                )
                iteration = controller.create_iteration(
                    "prior-scientific-run", 1, record.candidate_hash
                )
                attempt = controller.create_evaluation_attempt(
                    experiment_uid=record.experiment_uid,
                    run_id="prior-scientific-run",
                    iteration_id=int(iteration["id"]),
                    stage="QUICK",
                    suite="quick",
                    candidate_artifact_id=record.artifact_id,
                )
                controller.start_evaluation_attempt(record.experiment_uid)
                controller.finish_evaluation_attempt(
                    record.experiment_uid,
                    status="SUCCEEDED",
                    result=result_value,
                )
                controller.link_history_experiment(
                    record.experiment_uid, record.id
                )
                controller.accept_candidate(
                    int(iteration["id"]),
                    candidate_hash=record.candidate_hash,
                    hypothesis="visible hypothesis",
                    rationale="fixture",
                    candidate_path="candidate.py",
                )
                controller.update_iteration(
                    int(iteration["id"]),
                    status="COMPLETED",
                    stage="DONE",
                    outcome="QUICK_REJECTED",
                    experiment_ids={"quick": record.id},
                    result=result_value,
                )
                self.assertEqual(attempt["history_experiment_id"], None)

            draft = BenchmarkCohortPlan(
                cohort_id="authoritative-feedback",
                namespace=namespace,
                baseline=baseline,
                history_cutoff=record.id + 10_000,
                prompt_protocol_digest=CURRENT_PROMPT_PROTOCOL_DIGEST,
                feedback_snapshot_digest="sha256:" + "f" * 64,
                arms=(
                    BenchmarkArm("pro", proposer("opencode-deepseek-v4-pro")),
                    BenchmarkArm(
                        "flash", proposer("opencode-deepseek-v4-flash")
                    ),
                ),
                repetitions=1,
                max_candidates=1,
                max_wall_seconds=1,
                max_consecutive_failures=1,
            )
            plan_path = root / "draft.json"
            plan_path.write_text(json.dumps(draft.snapshot), encoding="utf-8")
            policy_path = root / "policy.json"
            policy_path.write_text("{}", encoding="utf-8")
            argv = [
                "benchmark",
                "init",
                "--database",
                str(database),
                "--config",
                str(config_path),
                "--campaign-id",
                "benchmark-feedback",
                "--plan",
                str(plan_path),
                "--initial-policy-snapshot",
                str(policy_path),
                "--max-candidates",
                "2",
                "--max-wall-ms",
                "2000",
                "--max-gpu-ms",
                "2000",
                "--max-tokens",
                "10000000",
                "--max-cost-microusd",
                "200",
            ]
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(campaign_main(argv), 0)
            authoritative = json.loads(output.getvalue())["result"]
            frozen_plan = authoritative["benchmark_plan"]
            feedback = authoritative["campaign_snapshot"]["feedback_snapshot"]
            self.assertEqual(frozen_plan["history_cutoff"], record.id)
            self.assertEqual(len(feedback), 1)
            summary = feedback[0]["result_summary"]
            self.assertEqual(
                [case["case_id"] for case in summary["cases"]],
                ["quick_shadow_tiles_127_n2"],
            )
            self.assertNotIn(
                "quick_shadow_tiles_128_n1", summary["per_case_speedups"]
            )
            self.assertEqual(
                frozen_plan["feedback_snapshot_digest"],
                canonical_sha256(feedback),
            )

            # Idempotent init consumes the already-authoritative Campaign,
            # even if the live Controller summary is later changed.
            with ControllerStore(
                root / "controller" / "controller.sqlite3"
            ) as controller:
                controller.update_iteration(
                    int(iteration["id"]), result={"status": "MUTATED"}
                )
            replay_output = io.StringIO()
            with redirect_stdout(replay_output):
                self.assertEqual(campaign_main(argv), 0)
            replay = json.loads(replay_output.getvalue())["result"]
            self.assertEqual(
                replay["campaign_snapshot"], authoritative["campaign_snapshot"]
            )

    def test_imported_staged_baseline_uses_distinct_local_seed_rows(self) -> None:
        namespace = LEGACY_RESEARCH_NAMESPACE
        environment = ExecutionEnvironmentDigest.resolved(
            evaluator_image_digest="sha256:" + "1" * 64,
            toolchain_digest="sha256:" + "2" * 64,
            framework_digest="sha256:" + "3" * 64,
            operator_abi_digest="sha256:" + "4" * 64,
            build_flags_digest="sha256:" + "5" * 64,
        )
        seed = BaselineRef.create(
            namespace=namespace,
            artifact_id="source-sha256-v1:" + "a" * 64,
            source="campaign",
            revision="deployment-source-ref",
            execution_environment=environment,
        )
        with tempfile.TemporaryDirectory() as temporary, CampaignStore(
            Path(temporary) / "campaign.sqlite3"
        ) as store:
            source_campaign = store.create_campaign(
                campaign_id="source-discovery",
                namespace_id=namespace.namespace_id,
                mode="DISCOVERY",
                snapshot={"source": "staged-fixture"},
                budget_limit=BudgetAmount(candidates=1),
                initial_baseline_ref=seed,
                initial_policy_snapshot={},
                allow_staged_lineage=True,
            )
            store.start_campaign("source-discovery")
            source_seed_row = store.list_baseline_revisions(
                "source-discovery"
            )[0]
            staged = store.advance_baseline(
                "source-discovery",
                expected_parent_revision_id=source_campaign[
                    "active_baseline_revision_id"
                ],
                parent_baseline_ref=source_seed_row["baseline_ref"],
                artifact_id="source-sha256-v1:" + "b" * 64,
                primary_experiment_uid="primary-staged",
                confirmation_experiment_uid="confirmation-staged",
                evidence_namespace_id=namespace.namespace_id,
                evidence_execution_environment=environment,
                policy_snapshot={"policy": "confirmed"},
                idempotency_key="advance-source",
            )
            staged_ref = BaselineRef.from_value(staged["baseline_ref"])
            plan = BenchmarkCohortPlan(
                cohort_id="reuse-staged-ref",
                namespace=namespace,
                baseline=staged_ref,
                history_cutoff=0,
                prompt_protocol_digest=CURRENT_PROMPT_PROTOCOL_DIGEST,
                feedback_snapshot_digest=canonical_sha256([]),
                arms=(
                    BenchmarkArm("pro", proposer("opencode-deepseek-v4-pro")),
                    BenchmarkArm(
                        "flash", proposer("opencode-deepseek-v4-flash")
                    ),
                ),
                repetitions=1,
                max_candidates=1,
                max_wall_seconds=1,
                max_consecutive_failures=1,
            )
            snapshot = self.campaign_snapshot(plan)
            campaigns = []
            revisions = []
            for campaign_id in ("benchmark-a", "benchmark-b"):
                campaign = store.create_campaign(
                    campaign_id=campaign_id,
                    namespace_id=namespace.namespace_id,
                    mode="BENCHMARK",
                    snapshot=snapshot,
                    budget_limit=BudgetAmount(candidates=2),
                    initial_baseline_ref=staged_ref,
                    initial_policy_snapshot={},
                    allow_staged_lineage=False,
                )
                campaigns.append(campaign)
                revision = store.list_baseline_revisions(campaign_id)[0]
                revisions.append(revision)
                store.start_campaign(campaign_id)
                child = store.create_child_run(
                    campaign_id,
                    controller_run_id=f"{campaign_id}-child",
                    proposer_profile=plan.arms[0].proposer_profile.to_dict(),
                    max_candidates=1,
                    max_wall_seconds=1,
                    max_consecutive_failures=1,
                )
                self.assertEqual(
                    child["baseline_revision_id"], revision["id"]
                )
            self.assertNotEqual(revisions[0]["id"], revisions[1]["id"])
            self.assertNotEqual(revisions[0]["id"], staged_ref.revision)
            self.assertNotEqual(revisions[1]["id"], staged_ref.revision)
            for campaign, revision in zip(campaigns, revisions):
                self.assertEqual(
                    campaign["active_baseline_revision_id"], revision["id"]
                )
                self.assertEqual(revision["baseline_ref"], staged_ref.to_dict())

    def test_trusted_executor_recovers_schedule_and_derives_report(self) -> None:
        plan = self.active_plan()
        snapshot = self.campaign_snapshot(plan)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "state"
            with (
                CampaignStore(root / "campaign.sqlite3") as campaign,
                ControllerStore(root / "controller.sqlite3") as controller,
                HistoryStore(
                    state / "history.sqlite3", state_dir=state
                ) as history,
            ):
                campaign.create_campaign(
                    campaign_id="benchmark-1",
                    namespace_id=plan.namespace.namespace_id,
                    mode="BENCHMARK",
                    snapshot=snapshot,
                    budget_limit=BudgetAmount(
                        candidates=2,
                        wall_ms=2_000,
                        gpu_ms=2_000,
                        tokens=200,
                        cost_microusd=200,
                    ),
                    initial_baseline_ref=plan.baseline,
                    initial_policy_snapshot={"source": "trusted-test"},
                    allow_staged_lineage=False,
                )
                campaign.start_campaign("benchmark-1")
                runner = _LedgerRunner(controller, plan)
                supervisor = CampaignSupervisor(
                    campaign,
                    runner,
                    resource_id="gpu1",
                    lease_ttl_seconds=2,
                    clock=lambda: 100.0,
                )

                def reservation_builder(**_values):  # type: ignore[no-untyped-def]
                    return BudgetAmount(
                        candidates=1,
                        wall_ms=1_000,
                        gpu_ms=1_000,
                        tokens=100,
                        cost_microusd=100,
                    )

                # Simulate a crash after the first durable child/budget intent
                # but before the external Controller action began.
                first_assignment = plan.schedule()[0]
                first_run_id = benchmark_controller_run_id(
                    "benchmark-1", plan, first_assignment
                )
                first_reservation = reservation_builder()
                campaign.reserve_budget(
                    "benchmark-1",
                    idempotency_key=child_budget_idempotency_key(first_run_id),
                    action_kind="CHILD_RUN",
                    amount=first_reservation,
                )
                campaign.create_child_run(
                    "benchmark-1",
                    controller_run_id=first_run_id,
                    proposer_profile=first_assignment.proposer_profile.to_dict(),
                    max_candidates=plan.max_candidates,
                    max_wall_seconds=plan.max_wall_seconds,
                    max_consecutive_failures=plan.max_consecutive_failures,
                )

                first = execute_benchmark_campaign(
                    campaign,
                    supervisor,
                    campaign_id="benchmark-1",
                    reservation_builder=reservation_builder,
                )
                # Recreate the cross-ledger crash window where the terminal
                # child is durable but its budget action was not yet settled.
                campaign.connection.execute(
                    """
                    UPDATE budget_actions
                    SET status = 'RESERVED', actual_candidates = NULL,
                        actual_wall_ms = NULL, actual_gpu_ms = NULL,
                        actual_tokens = NULL, actual_cost_microusd = NULL,
                        settled_at = NULL
                    WHERE campaign_id = ? AND idempotency_key = ?
                    """,
                    (
                        "benchmark-1",
                        child_budget_idempotency_key(first_run_id),
                    ),
                )
                campaign.connection.commit()
                replay = execute_benchmark_campaign(
                    campaign,
                    supervisor,
                    campaign_id="benchmark-1",
                    reservation_builder=reservation_builder,
                )
                self.assertEqual(first, replay)
                self.assertEqual(first["campaign_status"], "COMPLETED")
                self.assertEqual(first["terminal_children"], 2)
                self.assertEqual(len(runner.run_requests), 2)
                self.assertEqual(
                    campaign.get_budget_action(
                        "benchmark-1",
                        idempotency_key=child_budget_idempotency_key(
                            first_run_id
                        ),
                    )["status"],
                    "SETTLED",
                )
                self.assertEqual(
                    campaign.list_baseline_revisions("benchmark-1")[0][
                        "revision_index"
                    ],
                    0,
                )

                report = build_trusted_benchmark_report(
                    campaign,
                    controller,
                    history,
                    campaign_id="benchmark-1",
                )
                self.assertEqual(
                    report["evidence_source"],
                    "CAMPAIGN_AND_CONTROLLER_LEDGERS",
                )
                self.assertEqual(report["assignment_count"], 2)
                self.assertFalse(report["single_best_candidate_ranking"])
                self.assertEqual(
                    {row["child_run_count"] for row in report["arm_metrics"]},
                    {1},
                )


class _LedgerRunner:
    def __init__(self, store: ControllerStore, plan: BenchmarkCohortPlan) -> None:
        self.store = store
        self.plan = plan
        self.run_requests = []
        self.results = {}

    def run_child(self, request):  # type: ignore[no-untyped-def]
        self.run_requests.append(request)
        assignment = self.plan.schedule()[request.child_index - 1]
        self.assert_request(request, assignment)
        workflow = {
            "mode": "BENCHMARK",
            "namespace": self.plan.namespace.to_dict(),
            "baseline_ref": self.plan.baseline.to_dict(),
            "proposer_profile": assignment.proposer_profile.to_dict(),
            "history_cutoff": self.plan.history_cutoff,
            "cohort_id": self.plan.cohort_id,
            "prompt_protocol_digest": self.plan.prompt_protocol_digest,
            "feedback_snapshot_digest": self.plan.feedback_snapshot_digest,
            "campaign_id": request.campaign_id,
            "campaign_snapshot_digest": request.campaign_snapshot_digest,
            "campaign_snapshot": dict(request.campaign_snapshot),
            "budget": {
                "max_candidates": self.plan.max_candidates,
                "max_wall_seconds": self.plan.max_wall_seconds,
                "max_consecutive_failures": self.plan.max_consecutive_failures,
                "stop_after_promotion": True,
            },
            "campaign_child": {
                "child_id": request.child_id,
                "child_index": request.child_index,
                "controller_run_id": request.controller_run_id,
                "baseline_revision_id": request.baseline_revision_id,
            },
        }
        workflow = {**workflow, "snapshot_digest": canonical_sha256(workflow)}
        candidate_hash = hashlib.sha256(request.controller_run_id.encode()).hexdigest()
        self.store.create_run(
            run_id=request.controller_run_id,
            deadline_epoch=request.child_deadline_epoch,
            config={},
            initial_best_hash="a" * 64,
            namespace_id=self.plan.namespace.namespace_id,
            resolved_config_digest=workflow["snapshot_digest"],
            workflow_snapshot=workflow,
            baseline_ref=self.plan.baseline.to_dict(),
            history_cutoff=self.plan.history_cutoff,
        )
        iteration = self.store.create_iteration(
            request.controller_run_id, 1, "a" * 64
        )
        proposal_request = {
            "namespace_id": self.plan.namespace.namespace_id,
            "baseline_ref": self.plan.baseline.to_dict(),
            "history_cutoff": self.plan.history_cutoff,
            "proposer_profile": assignment.proposer_profile.to_dict(),
            "recent_feedback_digest": self.plan.feedback_snapshot_digest,
            "prompt_protocol": {
                "id": "proposal-v1",
                "revision": "v1",
                "digest": self.plan.prompt_protocol_digest,
            },
        }
        attempt = self.store.create_proposal_attempt(
            run_id=request.controller_run_id,
            iteration_id=iteration["id"],
            attempt_uid="attempt-" + candidate_hash[:24],
            parent_artifact_id=str(self.plan.baseline.artifact_id),
            proposer_profile=assignment.proposer_profile.to_dict(),
            prompt_protocol=proposal_request["prompt_protocol"],
            request=proposal_request,
        )
        self.store.finish_proposal_attempt(
            attempt["id"],
            status="SUCCEEDED",
            candidate_artifact_id="source-sha256-v1:" + candidate_hash,
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.00002,
            latency_ms=1,
        )
        self.store.accept_candidate(
            iteration["id"],
            candidate_hash=candidate_hash,
            hypothesis="fixture",
            rationale="fixture",
            candidate_path="candidate.py",
        )
        self.store.update_iteration(
            iteration["id"],
            status="COMPLETED",
            stage="DONE",
            outcome="POLICY_REJECTED",
        )
        self.store.update_run(request.controller_run_id, status="STOPPED")
        result = BoundedRunResult(
            RunnerOutcome.STOPPED,
            BudgetAmount(
                candidates=1,
                wall_ms=1,
                gpu_ms=0,
                tokens=15,
                cost_microusd=20,
            ),
            {
                "controller_run_id": request.controller_run_id,
                "controller_status": "STOPPED",
                "stop_reason": None,
                "proposal_attempts": 1,
                "evaluation_attempts": 0,
                "usage_unavailable": {},
            },
        )
        self.results[request.controller_run_id] = result
        return result

    def recover_child(self, request):  # type: ignore[no-untyped-def]
        return self.results.get(request.controller_run_id)

    def assert_request(self, request, assignment):  # type: ignore[no-untyped-def]
        expected = benchmark_controller_run_id(
            request.campaign_id, self.plan, assignment
        )
        if request.controller_run_id != expected:
            raise AssertionError("executor changed the deterministic run ID")


if __name__ == "__main__":
    unittest.main()
