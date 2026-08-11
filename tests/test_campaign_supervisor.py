from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest

from kernel_research.campaign import (
    BoundedRunResult,
    BudgetAmount,
    CampaignStore,
    CampaignSupervisor,
    ChildRunPlan,
    DataIntegrityRunnerFailure,
    HardRunnerFailure,
    PromotionEvidence,
    RunnerOutcome,
)
from kernel_research.platform.identity import BaselineRef, ExecutionEnvironmentDigest


NAMESPACE = "sha256:" + "a" * 64
OTHER_NAMESPACE = "sha256:" + "b" * 64
SEED = "source-sha256-v1:" + "1" * 64
CANDIDATE = "bundle-sha256-v1:" + "2" * 64
CHECKPOINT = "sha256:" + "3" * 64
ENVIRONMENT = ExecutionEnvironmentDigest.resolved(
    evaluator_image_digest="sha256:" + "4" * 64,
    toolchain_digest="sha256:" + "5" * 64,
    framework_digest="sha256:" + "6" * 64,
    operator_abi_digest="sha256:" + "7" * 64,
    build_flags_digest="sha256:" + "8" * 64,
)


def seed_ref(
    *, namespace: str = NAMESPACE, campaign_id: str = "campaign"
) -> BaselineRef:
    return BaselineRef.create(
        namespace=namespace,
        artifact_id=SEED,
        source="campaign",
        revision=f"{campaign_id}-deployment-seed-v2",
        execution_environment=ENVIRONMENT,
    )


class ScriptedRunner:
    def __init__(self, *, result=None, recovered_result=None, error=None) -> None:
        self.result = result
        self.recovered_result = recovered_result
        self.error = error
        self.run_requests = []
        self.recovery_requests = []

    def run_child(self, request):
        self.run_requests.append(request)
        if self.error is not None:
            raise self.error
        return self.result

    def recover_child(self, request):
        self.recovery_requests.append(request)
        if self.error is not None:
            raise self.error
        return self.recovered_result

    def prove_staged_promotion(self, **values):  # type: ignore[no-untyped-def]
        return values["evidence"]


def promotion(*, namespace: str = NAMESPACE) -> PromotionEvidence:
    return PromotionEvidence(
        artifact_id=CANDIDATE,
        namespace_id=namespace,
        parent_baseline_ref=seed_ref(namespace=namespace).to_dict(),
        execution_environment=ENVIRONMENT.to_dict(),
        primary_experiment_uid="experiment-primary",
        confirmation_experiment_uid="experiment-confirmation",
        checkpoint_digest=CHECKPOINT,
        policy_snapshot={"minimum_speedup": 1.01},
    )


class CampaignSupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.store = CampaignStore(Path(self.temporary.name) / "campaign.sqlite3")

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def create_campaign(
        self,
        *,
        campaign_id: str = "campaign",
        mode: str = "DISCOVERY",
        lineage: bool = True,
        budget: BudgetAmount | None = None,
    ) -> dict:
        campaign = self.store.create_campaign(
            campaign_id=campaign_id,
            namespace_id=NAMESPACE,
            mode=mode,
            snapshot={"profile": "frozen-v1"},
            budget_limit=budget
            or BudgetAmount(
                candidates=20,
                wall_ms=2_000_000,
                gpu_ms=1_000_000,
                tokens=200_000,
                cost_microusd=10_000_000,
            ),
            initial_baseline_ref=seed_ref(campaign_id=campaign_id),
            initial_policy_snapshot={"policy": "deployment"},
            allow_staged_lineage=lineage,
        )
        self.store.start_campaign(campaign_id)
        return campaign

    @staticmethod
    def plan(controller_run_id: str = "controller-1") -> ChildRunPlan:
        return ChildRunPlan(
            controller_run_id=controller_run_id,
            proposer_profile={"model": "deepseek", "harness": "opencode"},
            reservation=BudgetAmount(
                candidates=2,
                wall_ms=60_000,
                gpu_ms=45_000,
                tokens=10_000,
                cost_microusd=500_000,
            ),
            max_candidates=2,
            max_wall_seconds=60,
            max_consecutive_failures=3,
        )

    def supervisor(
        self, runner: ScriptedRunner, *, resource_id: str = "metax-c500:1"
    ) -> CampaignSupervisor:
        return CampaignSupervisor(
            self.store,
            runner,
            resource_id=resource_id,
            lease_ttl_seconds=120,
            clock=lambda: 100.0,
        )

    def lease_status(self, resource_id: str = "metax-c500:1") -> str:
        return self.store.connection.execute(
            """
            SELECT status FROM resource_leases
            WHERE resource_id = ?
            ORDER BY fencing_epoch DESC LIMIT 1
            """,
            (resource_id,),
        ).fetchone()[0]

    def test_executes_one_bounded_child_and_settles_actual_usage(self) -> None:
        self.create_campaign()
        actual = BudgetAmount(
            candidates=1,
            wall_ms=12_000,
            gpu_ms=8_000,
            tokens=2_000,
            cost_microusd=50_000,
        )
        runner = ScriptedRunner(
            result=BoundedRunResult(
                RunnerOutcome.STOPPED, actual, {"stop_reason": "controller limit"}
            )
        )

        report = self.supervisor(runner).execute_child("campaign", self.plan())

        self.assertEqual(report["disposition"], "STOPPED")
        self.assertEqual(report["budget_action"]["actual"], actual.to_dict())
        self.assertEqual(report["campaign"]["status"], "RUNNING")
        self.assertEqual(self.lease_status(), "RELEASED")
        self.assertEqual(len(runner.run_requests), 1)
        request = runner.run_requests[0]
        self.assertEqual(request.baseline_artifact_id, SEED)
        self.assertEqual(request.fencing_epoch, 1)
        self.assertEqual(request.max_candidates, 2)
        self.assertEqual(request.max_wall_seconds, 60)
        self.assertEqual(request.max_consecutive_failures, 3)
        self.assertTrue(request.stop_after_promotion)
        self.assertFalse(request.recovery)
        self.assertEqual(request.campaign_snapshot, {"profile": "frozen-v1"})
        self.assertTrue(request.campaign_snapshot_digest.startswith("sha256:"))

    def test_pending_intent_after_crash_reuses_its_fencing_lease_once(self) -> None:
        self.create_campaign()
        plan = self.plan()
        self.store.reserve_budget(
            "campaign",
            idempotency_key=plan.budget_idempotency_key,
            action_kind="CHILD_RUN",
            amount=plan.reservation,
        )
        self.store.create_child_run(
            "campaign",
            controller_run_id=plan.controller_run_id,
            proposer_profile=plan.proposer_profile,
            max_candidates=plan.max_candidates,
            max_wall_seconds=plan.max_wall_seconds,
            max_consecutive_failures=plan.max_consecutive_failures,
        )
        lease = self.store.acquire_resource(
            "campaign",
            resource_id="metax-c500:1",
            ttl_seconds=120,
            now_epoch=100,
        )
        runner = ScriptedRunner(
            result=BoundedRunResult(
                RunnerOutcome.STOPPED, BudgetAmount(candidates=1, wall_ms=1_000)
            )
        )

        report = self.supervisor(runner).execute_child("campaign", plan)

        self.assertEqual(report["disposition"], "STOPPED")
        self.assertEqual(len(runner.run_requests), 1)
        self.assertEqual(runner.run_requests[0].fencing_epoch, lease.fencing_epoch)
        self.assertEqual(
            self.store.connection.execute(
                "SELECT COUNT(*) FROM resource_leases WHERE resource_id = ?",
                (lease.resource_id,),
            ).fetchone()[0],
            1,
        )

    def test_running_child_uses_recovery_lookup_and_never_runs_again(self) -> None:
        self.create_campaign()
        plan = self.plan()
        self.store.reserve_budget(
            "campaign",
            idempotency_key=plan.budget_idempotency_key,
            action_kind="CHILD_RUN",
            amount=plan.reservation,
        )
        child = self.store.create_child_run(
            "campaign",
            controller_run_id=plan.controller_run_id,
            proposer_profile=plan.proposer_profile,
            max_candidates=plan.max_candidates,
            max_wall_seconds=plan.max_wall_seconds,
            max_consecutive_failures=plan.max_consecutive_failures,
        )
        self.store.acquire_resource(
            "campaign",
            resource_id="metax-c500:1",
            ttl_seconds=120,
            now_epoch=100,
        )
        self.store.start_child_run(
            child["id"], controller_run_id=plan.controller_run_id
        )
        recovered_usage = BudgetAmount(candidates=1, wall_ms=10_000, gpu_ms=9_000)
        runner = ScriptedRunner(
            result=BoundedRunResult(RunnerOutcome.FAILED, BudgetAmount()),
            recovered_result=BoundedRunResult(
                RunnerOutcome.STOPPED, recovered_usage, {"reconciled": True}
            ),
        )

        report = self.supervisor(runner).execute_child("campaign", plan)

        self.assertTrue(report["recovered"])
        self.assertFalse(report["runner_invoked"])
        self.assertEqual(len(runner.run_requests), 0)
        self.assertEqual(len(runner.recovery_requests), 1)
        self.assertTrue(runner.recovery_requests[0].recovery)
        self.assertEqual(report["budget_action"]["actual"], recovered_usage.to_dict())

    def test_unrecoverable_running_child_is_unknown_and_retry_is_read_only(self) -> None:
        self.create_campaign()
        plan = self.plan()
        self.store.reserve_budget(
            "campaign",
            idempotency_key=plan.budget_idempotency_key,
            action_kind="CHILD_RUN",
            amount=plan.reservation,
        )
        child = self.store.create_child_run(
            "campaign",
            controller_run_id=plan.controller_run_id,
            proposer_profile=plan.proposer_profile,
            max_candidates=plan.max_candidates,
            max_wall_seconds=plan.max_wall_seconds,
            max_consecutive_failures=plan.max_consecutive_failures,
        )
        self.store.acquire_resource(
            "campaign",
            resource_id="metax-c500:1",
            ttl_seconds=120,
            now_epoch=100,
        )
        self.store.start_child_run(
            child["id"], controller_run_id=plan.controller_run_id
        )
        runner = ScriptedRunner(recovered_result=None)
        supervisor = self.supervisor(runner)

        first = supervisor.execute_child("campaign", plan)
        second = supervisor.execute_child("campaign", plan)

        self.assertEqual(first["disposition"], "UNKNOWN_GPU_OUTCOME")
        self.assertEqual(first["campaign"]["status"], "PAUSED_UNKNOWN_OUTCOME")
        self.assertEqual(first["budget_action"]["actual"], plan.reservation.to_dict())
        self.assertEqual(self.lease_status(), "QUARANTINED")
        self.assertEqual(len(runner.recovery_requests), 1)
        self.assertEqual(len(runner.run_requests), 0)
        self.assertTrue(second["recovered"])
        self.assertFalse(second["runner_invoked"])

    def test_expired_running_lease_fails_unknown_without_recovery_call(self) -> None:
        self.create_campaign()
        plan = self.plan()
        self.store.reserve_budget(
            "campaign",
            idempotency_key=plan.budget_idempotency_key,
            action_kind="CHILD_RUN",
            amount=plan.reservation,
        )
        child = self.store.create_child_run(
            "campaign",
            controller_run_id=plan.controller_run_id,
            proposer_profile=plan.proposer_profile,
            max_candidates=plan.max_candidates,
            max_wall_seconds=plan.max_wall_seconds,
            max_consecutive_failures=plan.max_consecutive_failures,
        )
        self.store.acquire_resource(
            "campaign",
            resource_id="metax-c500:1",
            ttl_seconds=120,
            now_epoch=100,
        )
        self.store.start_child_run(
            child["id"], controller_run_id=plan.controller_run_id
        )
        runner = ScriptedRunner(
            recovered_result=BoundedRunResult(
                RunnerOutcome.STOPPED, BudgetAmount(candidates=1)
            )
        )
        supervisor = CampaignSupervisor(
            self.store,
            runner,
            resource_id="metax-c500:1",
            lease_ttl_seconds=120,
            clock=lambda: 221.0,
        )

        report = supervisor.execute_child("campaign", plan)

        self.assertEqual(report["disposition"], "UNKNOWN_GPU_OUTCOME")
        self.assertEqual(runner.recovery_requests, [])
        self.assertEqual(self.lease_status(), "QUARANTINED")

    def test_terminal_write_window_reconciles_budget_and_lease_idempotently(self) -> None:
        self.create_campaign()
        plan = self.plan()
        self.store.reserve_budget(
            "campaign",
            idempotency_key=plan.budget_idempotency_key,
            action_kind="CHILD_RUN",
            amount=plan.reservation,
        )
        child = self.store.create_child_run(
            "campaign",
            controller_run_id=plan.controller_run_id,
            proposer_profile=plan.proposer_profile,
            max_candidates=plan.max_candidates,
            max_wall_seconds=plan.max_wall_seconds,
            max_consecutive_failures=plan.max_consecutive_failures,
        )
        lease = self.store.acquire_resource(
            "campaign",
            resource_id="metax-c500:1",
            ttl_seconds=120,
            now_epoch=100,
        )
        self.store.start_child_run(
            child["id"], controller_run_id=plan.controller_run_id
        )
        actual = BudgetAmount(candidates=1, wall_ms=5_000, gpu_ms=4_000)
        stored_result = {
            "schema_version": 1,
            "controller_run_id": plan.controller_run_id,
            "budget_idempotency_key": plan.budget_idempotency_key,
            "outcome": "STOPPED",
            "usage": actual.to_dict(),
            "details": {"simulated_crash": "after terminal write"},
            "resource_lease": {
                "resource_id": lease.resource_id,
                "fencing_epoch": lease.fencing_epoch,
            },
            "promotion": None,
        }
        self.store.finish_child_run(
            child["id"], status="STOPPED", result=stored_result
        )
        runner = ScriptedRunner()
        supervisor = self.supervisor(runner)

        first = supervisor.execute_child("campaign", plan)
        second = supervisor.execute_child("campaign", plan)

        self.assertEqual(first["budget_action"]["status"], "SETTLED")
        self.assertEqual(first["budget_action"]["actual"], actual.to_dict())
        self.assertEqual(self.lease_status(), "RELEASED")
        self.assertEqual(len(runner.run_requests), 0)
        self.assertEqual(len(runner.recovery_requests), 0)
        self.assertEqual(second["budget_action"]["id"], first["budget_action"]["id"])

    def test_invalid_terminal_reconciliation_escalates_to_integrity_pause(self) -> None:
        self.create_campaign()
        plan = self.plan()
        self.store.reserve_budget(
            "campaign",
            idempotency_key=plan.budget_idempotency_key,
            action_kind="CHILD_RUN",
            amount=plan.reservation,
        )
        child = self.store.create_child_run(
            "campaign",
            controller_run_id=plan.controller_run_id,
            proposer_profile=plan.proposer_profile,
            max_candidates=plan.max_candidates,
            max_wall_seconds=plan.max_wall_seconds,
            max_consecutive_failures=plan.max_consecutive_failures,
        )
        self.store.start_child_run(
            child["id"], controller_run_id=plan.controller_run_id
        )
        self.store.finish_child_run(
            child["id"], status="STOPPED", result={"untrusted": "shape"}
        )

        with self.assertRaisesRegex(ValueError, "result schema is invalid"):
            self.supervisor(ScriptedRunner()).execute_child("campaign", plan)

        self.assertEqual(
            self.store.get_campaign("campaign")["status"],
            "PAUSED_DATA_INTEGRITY",
        )

    def test_promotion_pauses_then_staged_lineage_advances_by_cas(self) -> None:
        campaign = self.create_campaign()
        runner = ScriptedRunner(
            result=BoundedRunResult(
                RunnerOutcome.PROMOTED,
                BudgetAmount(candidates=1, wall_ms=20_000, gpu_ms=15_000),
                {"speedup": 1.04},
                promotion(),
            )
        )
        supervisor = self.supervisor(runner)

        report = supervisor.execute_child("campaign", self.plan())

        self.assertEqual(report["campaign"]["status"], "PAUSED_OPERATOR")
        self.assertEqual(
            report["campaign"]["active_baseline_revision_id"],
            campaign["active_baseline_revision_id"],
        )
        revision = supervisor.advance_staged_lineage(
            "campaign", child_id=report["child"]["id"], idempotency_key="advance-1"
        )
        again = supervisor.advance_staged_lineage(
            "campaign", child_id=report["child"]["id"], idempotency_key="advance-1"
        )
        self.assertEqual(revision["id"], again["id"])
        self.assertEqual(revision["parent_revision_id"], campaign["active_baseline_revision_id"])
        self.assertEqual(revision["artifact_id"], CANDIDATE)
        self.assertEqual(revision["policy_snapshot"]["checkpoint_digest"], CHECKPOINT)
        self.assertEqual(self.store.get_campaign("campaign")["status"], "RUNNING")

    def test_benchmark_may_record_promotion_but_never_advance(self) -> None:
        campaign = self.create_campaign(
            campaign_id="benchmark", mode="BENCHMARK", lineage=False
        )
        runner = ScriptedRunner(
            result=BoundedRunResult(
                RunnerOutcome.PROMOTED,
                BudgetAmount(candidates=1, wall_ms=10_000),
                {},
                promotion(),
            )
        )
        supervisor = self.supervisor(runner)
        report = supervisor.execute_child("benchmark", self.plan("benchmark-run"))

        with self.assertRaisesRegex(ValueError, "benchmark"):
            supervisor.advance_staged_lineage(
                "benchmark",
                child_id=report["child"]["id"],
                idempotency_key="forbidden",
            )
        self.assertEqual(
            self.store.get_campaign("benchmark")["active_baseline_revision_id"],
            campaign["active_baseline_revision_id"],
        )

    def test_failure_classes_pause_and_quarantine_without_hidden_retry(self) -> None:
        cases = (
            (
                "hard",
                HardRunnerFailure("hard device fault"),
                "HARD_FAILED",
                "PAUSED_HARD_FAILURE",
            ),
            (
                "integrity",
                DataIntegrityRunnerFailure("artifact mismatch"),
                "DATA_INTEGRITY",
                "PAUSED_DATA_INTEGRITY",
            ),
        )
        for index, (name, error, outcome, campaign_status) in enumerate(cases):
            with self.subTest(name=name):
                campaign_id = f"campaign-{index}"
                resource_id = f"metax-c500:{index + 1}"
                self.create_campaign(campaign_id=campaign_id)
                runner = ScriptedRunner(error=error)
                report = self.supervisor(runner, resource_id=resource_id).execute_child(
                    campaign_id, self.plan(f"controller-{index}")
                )
                self.assertEqual(report["disposition"], outcome)
                self.assertEqual(report["campaign"]["status"], campaign_status)
                self.assertEqual(self.lease_status(resource_id), "QUARANTINED")

    def test_quarantined_resource_requires_same_campaign_doctor_resume(self) -> None:
        self.create_campaign()
        hard_runner = ScriptedRunner(error=HardRunnerFailure("hard fault"))
        supervisor = self.supervisor(hard_runner)
        supervisor.execute_child("campaign", self.plan())
        self.assertEqual(self.lease_status(), "QUARANTINED")

        self.create_campaign(campaign_id="other")
        with self.assertRaisesRegex(ValueError, "quarantined"):
            self.store.acquire_resource(
                "other",
                resource_id="metax-c500:1",
                ttl_seconds=120,
                now_epoch=100,
            )

        evidence = {
            "schema_version": 1,
            "kind": "CAMPAIGN_RESUME_DOCTOR",
            "status": "SUCCESS",
            "campaign_id": "campaign",
            "resource_id": "metax-c500:1",
            "observed_epoch": time.time() + 1,
            "namespace_id": NAMESPACE,
            "config_digest": "sha256:" + "d" * 64,
            "execution_environment": ENVIRONMENT.to_dict(),
            "doctor_result": {
                "status": "SUCCESS",
                "c500_probe": {
                    "environment": {"compile_probe_status": "PASSED"}
                },
            },
        }
        recorded = self.store.record_doctor_evidence(
            "campaign", evidence=evidence
        )
        self.store.resume_campaign(
            "campaign", doctor_evidence_digest=recorded["digest"]
        )
        recovered_runner = ScriptedRunner(
            result=BoundedRunResult(
                RunnerOutcome.STOPPED,
                BudgetAmount(candidates=1, wall_ms=1_000),
            )
        )
        report = self.supervisor(recovered_runner).execute_child(
            "campaign", self.plan("controller-after-doctor")
        )
        self.assertEqual(report["disposition"], "STOPPED")
        self.assertEqual(self.lease_status(), "RELEASED")

    def test_unclassified_exception_is_persisted_unknown_before_reraise(self) -> None:
        self.create_campaign()
        runner = ScriptedRunner(error=RuntimeError("runner transport vanished"))

        with self.assertRaisesRegex(RuntimeError, "transport vanished"):
            self.supervisor(runner).execute_child("campaign", self.plan())

        child = self.store.list_child_runs("campaign")[0]
        self.assertEqual(child["status"], "UNKNOWN_GPU_OUTCOME")
        self.assertEqual(
            self.store.get_campaign("campaign")["status"],
            "PAUSED_UNKNOWN_OUTCOME",
        )
        self.assertEqual(self.lease_status(), "QUARANTINED")

    def test_cross_namespace_promotion_fails_closed_as_data_integrity(self) -> None:
        self.create_campaign()
        runner = ScriptedRunner(
            result=BoundedRunResult(
                RunnerOutcome.PROMOTED,
                BudgetAmount(candidates=1, wall_ms=10_000),
                {},
                promotion(namespace=OTHER_NAMESPACE),
            )
        )

        report = self.supervisor(runner).execute_child("campaign", self.plan())

        self.assertEqual(report["disposition"], "DATA_INTEGRITY")
        self.assertEqual(report["child"]["status"], "FAILED")
        self.assertEqual(report["campaign"]["status"], "PAUSED_DATA_INTEGRITY")
        self.assertEqual(report["budget_action"]["actual"], self.plan().reservation.to_dict())

    def test_insufficient_worst_case_budget_pauses_without_creating_child(self) -> None:
        self.create_campaign(
            budget=BudgetAmount(
                candidates=1,
                wall_ms=59_999,
                gpu_ms=100_000,
                tokens=100_000,
                cost_microusd=100_000,
            )
        )
        runner = ScriptedRunner()

        report = self.supervisor(runner).execute_child("campaign", self.plan())

        self.assertEqual(report["disposition"], "PAUSED_BUDGET")
        self.assertIsNone(report["child"])
        self.assertEqual(self.store.list_child_runs("campaign"), [])
        self.assertEqual(len(runner.run_requests), 0)

    def test_plan_requires_reservation_to_cover_unchanged_single_run_caps(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate limit"):
            ChildRunPlan(
                controller_run_id="too-many",
                proposer_profile={},
                reservation=BudgetAmount(candidates=6, wall_ms=60_000),
                max_candidates=6,
                max_wall_seconds=60,
            )
        with self.assertRaisesRegex(ValueError, "cover the child wall"):
            ChildRunPlan(
                controller_run_id="under-reserved",
                proposer_profile={},
                reservation=BudgetAmount(candidates=1, wall_ms=59_999),
                max_candidates=1,
                max_wall_seconds=60,
            )
        for field in ("gpu_ms", "tokens", "cost_microusd"):
            values = {
                "candidates": 1,
                "wall_ms": 60_000,
                "gpu_ms": 60_000,
                "tokens": 1,
                "cost_microusd": 1,
            }
            values[field] = 0
            with self.subTest(missing_axis=field), self.assertRaisesRegex(
                ValueError, "positive"
            ):
                ChildRunPlan(
                    controller_run_id=f"missing-{field}",
                    proposer_profile={},
                    reservation=BudgetAmount(**values),
                    max_candidates=1,
                    max_wall_seconds=60,
                )

    def test_store_child_intent_and_active_lease_queries_are_idempotent(self) -> None:
        self.create_campaign()
        plan = self.plan()
        first = self.store.create_child_run(
            "campaign",
            controller_run_id=plan.controller_run_id,
            proposer_profile=plan.proposer_profile,
            max_candidates=plan.max_candidates,
            max_wall_seconds=plan.max_wall_seconds,
            max_consecutive_failures=plan.max_consecutive_failures,
        )
        again = self.store.create_child_run(
            "campaign",
            controller_run_id=plan.controller_run_id,
            proposer_profile=plan.proposer_profile,
            max_candidates=plan.max_candidates,
            max_wall_seconds=plan.max_wall_seconds,
            max_consecutive_failures=plan.max_consecutive_failures,
        )
        self.assertEqual(first["id"], again["id"])
        self.assertEqual(
            self.store.get_child_run_by_controller_run_id(plan.controller_run_id)["id"],
            first["id"],
        )
        lease = self.store.acquire_resource(
            "campaign",
            resource_id="metax-c500:1",
            ttl_seconds=120,
            now_epoch=100,
        )
        self.assertEqual(
            self.store.get_active_resource_lease("metax-c500:1"), lease
        )
        with self.assertRaisesRegex(ValueError, "different child intent"):
            self.store.create_child_run(
                "campaign",
                controller_run_id=plan.controller_run_id,
                proposer_profile={"model": "other"},
                max_candidates=plan.max_candidates,
                max_wall_seconds=plan.max_wall_seconds,
                max_consecutive_failures=plan.max_consecutive_failures,
            )


if __name__ == "__main__":
    unittest.main()
