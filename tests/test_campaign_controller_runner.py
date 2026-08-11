from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import json
import tempfile
import time
import unittest
from unittest import mock

from kernel_research.autorun.controller import (
    ControllerDataIntegrityError,
    ResearchController,
)
from kernel_research.autorun.errors import ControlledRuntimeError
from kernel_research.autorun.models import ProposalV1
from kernel_research.autorun.runtime import CommandResult
from kernel_research.autorun.store import ControllerStore
from kernel_research.campaign.controller_runner import (
    ResearchControllerRunner,
    legacy_campaign_snapshot,
    trusted_child_reservation,
    trusted_resume_doctor,
)
from kernel_research.campaign.benchmark import (
    BenchmarkArm,
    BenchmarkCohortPlan,
    CURRENT_PROMPT_PROTOCOL_DIGEST,
    benchmark_controller_run_id,
    bind_benchmark_campaign_snapshot,
)
from kernel_research.campaign import cli as campaign_cli
from kernel_research.campaign.cli import main as campaign_main
from kernel_research.campaign.models import BudgetAmount, ResourceLease
from kernel_research.campaign.store import CampaignStore
from kernel_research.campaign.supervisor import (
    BoundedRunRequest,
    CampaignSupervisor,
    ChildRunPlan,
    DataIntegrityRunnerFailure,
    HardRunnerFailure,
    PromotionEvidence,
    RunnerOutcome,
    UnknownRunnerOutcome,
    child_budget_idempotency_key,
)
from kernel_research.constants import DOCTOR_TIMEOUT_SEC
from kernel_research.history import HistoryStore
from kernel_research.platform.artifacts import ArtifactId
from kernel_research.platform.canonical import canonical_sha256
from kernel_research.platform.identity import (
    BaselineRef,
    ExecutionEnvironmentDigest,
    ExperimentIdentity,
)
from kernel_research.platform.profiles import (
    BUILTIN_PROFILE_REGISTRY,
    CURRENT_RESEARCH_NAMESPACE,
    LEGACY_RESEARCH_NAMESPACE,
    ResearchNamespace,
)
from kernel_research.recovery import verify_checkpoint

from tests.test_autorun import (
    FakeEvaluator,
    NoopRunner,
    SEED,
    SEED_HASH,
    StaticProposer,
    _baseline,
    _config,
    _proposal_value,
    _with_different_block_size_n,
)
from tests.test_controller_v2_integration import AuditedStaticProposer


class FixtureController(ResearchController):
    def doctor(self, **_values):  # type: ignore[no-untyped-def]
        return {"status": "SUCCESS", "errors": [], "identity": {}}


class CountingDoctorController(FixtureController):
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.doctor_calls = 0

    def doctor(self, **values):  # type: ignore[no-untyped-def]
        self.doctor_calls += 1
        return super().doctor(**values)


class TrustedResumeDoctorFixture:
    def __init__(self, config, *, status: str = "SUCCESS") -> None:  # type: ignore[no-untyped-def]
        self.config = config
        self.status = status
        self.calls: list[dict] = []
        self.guard_calls: list[dict] = []

    def doctor(self, **values):  # type: ignore[no-untyped-def]
        self.calls.append(dict(values))
        return {
            "schema_version": 1,
            "command": "doctor",
            "status": self.status,
            "errors": [] if self.status == "SUCCESS" else ["probe failed"],
            "c500_probe": {
                "status": self.status,
                "environment": {
                    "compile_probe_status": (
                        "PASSED" if self.status == "SUCCESS" else "FAILED"
                    )
                },
            },
        }

    def _resolved_execution_environment(self, namespace):  # type: ignore[no-untyped-def]
        return ResearchController(
            self.config
        )._resolved_execution_environment(namespace)

    def campaign_resume_doctor(self, **values):  # type: ignore[no-untyped-def]
        self.guard_calls.append(dict(values))
        return self.doctor(
            require_secret=True,
            allowed_best_hashes=None,
            run_id=values["run_id"],
        )


class DockerDoctorRunner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, argv, **values):  # type: ignore[no-untyped-def]
        self.calls.append({"argv": tuple(argv), **dict(values)})
        return CommandResult(
            argv=tuple(argv),
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "SUCCESS",
                    "environment": {"compile_probe_status": "PASSED"},
                }
            ),
            stderr="",
        )

    def remove_exact_container(self, *_values):  # type: ignore[no-untyped-def]
        return None


class CampaignControllerRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.config = _config(self.root, max_candidates=5)
        (self.config.repository_dir / "kernel.py").write_text(
            SEED, encoding="utf-8"
        )
        (self.config.repository_dir / "program.md").write_text(
            "Change only kernel.py.", encoding="utf-8"
        )
        _baseline(self.config.state_dir)
        self.snapshot = legacy_campaign_snapshot()
        self.lease = ResourceLease(
            resource_id="gpu1",
            campaign_id="campaign-1",
            fencing_epoch=7,
            expires_epoch=1_000.0,
            status="ACTIVE",
        )
        self.campaign_store = CampaignStore(
            self.root / "campaign" / "campaign.sqlite3"
        )
        self.baseline_revision_id = "baseline-seed-1"
        self.baseline_ref = BaselineRef.create(
            namespace=LEGACY_RESEARCH_NAMESPACE,
            artifact_id=ArtifactId.source_sha256(SEED_HASH),
            source="campaign",
            revision=self.baseline_revision_id,
        ).to_dict()
        self.child_id = 1
        self.factories: list[FixtureController] = []
        source = _with_different_block_size_n(SEED)
        self.proposal = ProposalV1.from_value(
            _proposal_value(source), expected_parent_hash=SEED_HASH
        )
        self.evaluator = FakeEvaluator()

        def factory(config):  # type: ignore[no-untyped-def]
            def proposer_factory(
                _run_id: str, iteration_index: int, run_dir: Path
            ) -> AuditedStaticProposer:
                return AuditedStaticProposer(
                    self.proposal,
                    run_dir=run_dir,
                    iteration_index=iteration_index,
                )

            controller = FixtureController(
                config,
                evaluator=self.evaluator,
                runner=NoopRunner(),
                proposer_factory=proposer_factory,
                clock=lambda: 100.0,
            )
            self.factories.append(controller)
            return controller

        self.runner = ResearchControllerRunner(
            self.config,
            campaign_store=self.campaign_store,
            fence_lookup=lambda resource_id: (
                self.lease if resource_id == self.lease.resource_id else None
            ),
            controller_factory=factory,
            clock=lambda: 100.0,
        )

    def tearDown(self) -> None:
        self.campaign_store.close()
        self.temporary.cleanup()

    def test_trusted_reservation_covers_every_external_usage_axis(self) -> None:
        remaining = BudgetAmount(
            candidates=5,
            wall_ms=60_000,
            gpu_ms=60_000,
            tokens=8_000_000,
            cost_microusd=5_000_000,
        )
        reservation = trusted_child_reservation(
            proposer_profile=self.proposer_profile(),
            campaign_remaining=remaining,
            max_candidates=1,
            max_wall_seconds=60,
            max_consecutive_failures=3,
        )
        self.assertEqual(
            reservation,
            BudgetAmount(
                candidates=1,
                wall_ms=60_000,
                gpu_ms=60_000,
                tokens=8_000_000,
                cost_microusd=5_000_000,
            ),
        )
        with self.assertRaisesRegex(ValueError, "trusted child worst case"):
            trusted_child_reservation(
                proposer_profile=self.proposer_profile(),
                campaign_remaining=BudgetAmount(
                    candidates=5,
                    wall_ms=60_000,
                    gpu_ms=60_000,
                    tokens=7_999_999,
                    cost_microusd=5_000_000,
                ),
                max_candidates=1,
                max_wall_seconds=60,
                max_consecutive_failures=3,
            )

    def provision_campaign(
        self,
        snapshot: dict,
        *,
        controller_run_id: str = "controller-child-1",
        reserve: bool = True,
        initial_baseline_ref: BaselineRef | None = None,
        allow_staged_lineage: bool = False,
    ) -> None:
        namespace = ResearchNamespace.from_value(snapshot["namespace"])
        campaign = self.campaign_store.create_campaign(
            campaign_id="campaign-1",
            namespace_id=namespace.namespace_id,
            mode=snapshot["mode"],
            snapshot=snapshot,
            budget_limit=BudgetAmount(
                candidates=5,
                wall_ms=300_000,
                gpu_ms=300_000,
                tokens=8_000_000,
                cost_microusd=5_000_000,
            ),
            initial_artifact_id=(
                str(ArtifactId.source_sha256(SEED_HASH))
                if initial_baseline_ref is None
                else None
            ),
            initial_baseline_ref=initial_baseline_ref,
            initial_policy_snapshot={"source": "deployment"},
            allow_staged_lineage=allow_staged_lineage,
        )
        self.baseline_revision_id = campaign["active_baseline_revision_id"]
        self.baseline_ref = self.campaign_store.list_baseline_revisions(
            "campaign-1"
        )[0]["baseline_ref"]
        self.campaign_store.start_campaign("campaign-1")
        child = self.campaign_store.create_child_run(
            "campaign-1",
            controller_run_id=controller_run_id,
            proposer_profile=self.proposer_profile(),
            max_candidates=1,
            max_wall_seconds=60,
            max_consecutive_failures=3,
        )
        self.child_id = int(child["id"])
        self.campaign_store.start_child_run(
            self.child_id, controller_run_id=controller_run_id
        )
        self.lease = self.campaign_store.acquire_resource(
            "campaign-1",
            resource_id="gpu1",
            ttl_seconds=900,
            now_epoch=100,
        )
        if reserve:
            reservation = self.request(run_id=controller_run_id).budget_limit
            self.campaign_store.reserve_budget(
                "campaign-1",
                idempotency_key=child_budget_idempotency_key(controller_run_id),
                action_kind="CHILD_RUN",
                amount=reservation,
            )

    @staticmethod
    def proposer_profile(*, harness: str = "opencode") -> dict:
        profile_id = (
            "opencode-deepseek-v4-pro"
            if harness == "opencode"
            else "direct-api-deepseek-v4-pro"
        )
        return BUILTIN_PROFILE_REGISTRY.get(
            kind="proposer", profile_id=profile_id, revision="v1"
        ).ref.to_dict()

    def record_resolved_baseline(
        self, namespace: ResearchNamespace
    ) -> BaselineRef:
        environment = ResearchController(
            self.config
        )._resolved_execution_environment(namespace)
        artifact_id = ArtifactId.source_sha256(SEED_HASH)
        history_baseline = BaselineRef.create(
            namespace=namespace,
            artifact_id=artifact_id,
            source="deployment",
            revision=f"{namespace.namespace_id[-12:]}-history-seed",
            execution_environment=environment,
        )
        identity = ExperimentIdentity.create(
            namespace=namespace,
            mode="DISCOVERY",
            candidate_artifact_id=artifact_id,
            parent_artifact_id=artifact_id,
            baseline=history_baseline,
            execution_environment=environment,
            stage="baseline",
            suite="full",
            replicate_kind="primary",
            replicate_index=0,
            run_id=f"resolved-seed-{namespace.namespace_id[-12:]}",
            iteration=0,
        )
        with HistoryStore(
            self.config.state_dir / "history.sqlite3",
            state_dir=self.config.state_dir,
        ) as history:
            history.ensure_namespace(namespace.namespace_id, namespace.to_dict())
            history.record_experiment(
                candidate_source=SEED,
                status="SUCCESS",
                backend="c500",
                suite="full",
                promotable=True,
                aggregate_score=1.0,
                identity=identity,
                result={
                    "status": "SUCCESS",
                    "promotion": {"phase": "baseline", "confirmed": True},
                },
            )
        return BaselineRef.create(
            namespace=namespace,
            artifact_id=artifact_id,
            source="campaign",
            revision=f"campaign-seed-{namespace.namespace_id[-12:]}",
            execution_environment=environment,
        )

    def request(
        self,
        *,
        recovery: bool = False,
        proposer_profile: dict | None = None,
        run_id: str = "controller-child-1",
        snapshot: dict | None = None,
    ) -> BoundedRunRequest:
        selected_snapshot = self.snapshot if snapshot is None else snapshot
        namespace = ResearchNamespace.from_value(selected_snapshot["namespace"])
        return BoundedRunRequest(
            campaign_id="campaign-1",
            namespace_id=namespace.namespace_id,
            mode=selected_snapshot["mode"],
            campaign_snapshot_digest=canonical_sha256(selected_snapshot),
            campaign_snapshot=selected_snapshot,
            child_id=self.child_id,
            child_index=1,
            controller_run_id=run_id,
            baseline_revision_id=self.baseline_revision_id,
            baseline_artifact_id=str(ArtifactId.source_sha256(SEED_HASH)),
            proposer_profile=proposer_profile or self.proposer_profile(),
            budget_limit=BudgetAmount(
                candidates=1,
                wall_ms=60_000,
                gpu_ms=60_000,
                tokens=8_000_000,
                cost_microusd=5_000_000,
            ),
            max_candidates=1,
            max_wall_seconds=60,
            max_consecutive_failures=3,
            stop_after_promotion=True,
            resource_id="gpu1",
            fencing_epoch=self.lease.fencing_epoch,
            lease_expires_epoch=self.lease.expires_epoch,
            child_deadline_epoch=160.0,
            baseline_ref=self.baseline_ref,
            recovery=recovery,
        )

    def persist_hard_terminal(
        self, *, run_id: str, iteration_outcome: str
    ) -> BoundedRunRequest:
        request = self.request(recovery=True, run_id=run_id)
        context = self.runner._validate_request(request, recovery=True)
        bounded = self.runner._bounded_config(request, context["proposer"])
        baseline = self.runner._campaign_baseline(request)
        snapshot = self.runner._workflow_snapshot(
            request,
            bounded_config=bounded,
            proposer=context["proposer"],
            baseline_ref=baseline,
            history_cutoff=1,
        )
        with ControllerStore(self.runner.controller_db) as store:
            store.create_run(
                run_id=run_id,
                deadline_epoch=200,
                config=bounded.redacted_dict(),
                initial_best_hash=SEED_HASH,
                namespace_id=request.namespace_id,
                resolved_config_digest=snapshot["snapshot_digest"],
                workflow_snapshot=snapshot,
                baseline_ref=baseline.to_dict(),
                history_cutoff=1,
            )
            iteration = store.create_iteration(run_id, 1, SEED_HASH)
            store.update_iteration(
                int(iteration["id"]),
                status="COMPLETED",
                stage="DONE",
                outcome=iteration_outcome,
            )
            store.update_run(run_id, status="HARD_FAILED")
        return request

    def test_runs_fixed_campaign_baseline_and_exports_confirmed_evidence(self) -> None:
        self.provision_campaign(self.snapshot)
        request = self.request()

        result = self.runner.run_child(request)

        self.assertEqual(result.outcome, RunnerOutcome.PROMOTED)
        self.assertEqual(result.usage.candidates, 1)
        self.assertEqual(result.usage.tokens, request.budget_limit.tokens)
        self.assertIsNotNone(result.promotion)
        assert result.promotion is not None
        self.assertEqual(
            result.promotion.namespace_id,
            LEGACY_RESEARCH_NAMESPACE.namespace_id,
        )
        self.assertNotEqual(
            result.promotion.primary_experiment_uid,
            result.promotion.confirmation_experiment_uid,
        )
        self.assertTrue(result.promotion.checkpoint_digest.startswith("sha256:"))
        with ControllerStore(self.runner.controller_db) as store:
            run = store.get_run(request.controller_run_id)
            attempts = store.list_evaluation_attempts(request.controller_run_id)
        self.assertEqual(run["status"], "PROMOTED")
        self.assertEqual(run["baseline_ref"]["source"], "campaign")
        self.assertEqual(
            run["baseline_ref"]["revision"], request.baseline_revision_id
        )
        self.assertEqual(
            run["workflow_snapshot"]["resource_lease"]["fencing_epoch"],
            self.lease.fencing_epoch,
        )
        self.assertEqual(
            run["workflow_snapshot"]["campaign_budget_reservation"][
                "reservation"
            ],
            request.budget_limit.to_dict(),
        )
        self.assertEqual(
            run["workflow_snapshot"]["campaign_budget_reservation"][
                "reservation_idempotency_key"
            ],
            child_budget_idempotency_key(request.controller_run_id),
        )
        self.assertEqual(len(attempts), 4)
        with HistoryStore(
            self.config.state_dir / "history.sqlite3",
            state_dir=self.config.state_dir,
        ) as history:
            self.assertIsNotNone(
                history.get_experiment_by_uid(
                    result.promotion.primary_experiment_uid
                )
            )
            self.assertIsNotNone(
                history.get_experiment_by_uid(
                    result.promotion.confirmation_experiment_uid
                )
            )

        stages_before = list(self.evaluator.stages)
        recovered = self.runner.recover_child(self.request(recovery=True))
        self.assertIsNotNone(recovered)
        assert recovered is not None and recovered.promotion is not None
        self.assertEqual(
            recovered.promotion.checkpoint_digest,
            result.promotion.checkpoint_digest,
        )
        self.assertEqual(self.evaluator.stages, stages_before)

        self.campaign_store.finish_child_run(
            self.child_id,
            status="PROMOTED",
            result={"promotion": result.promotion.to_dict()},
        )
        output = io.StringIO()
        with redirect_stdout(output):
            nomination_code = campaign_main(
                [
                    "oj",
                    "nominate",
                    "--database",
                    str(self.campaign_store.path),
                    "--campaign-id",
                    "campaign-1",
                    "--nomination-type",
                    "PRIMARY_SUBMISSION",
                    "--child-id",
                    str(self.child_id),
                    "--state-dir",
                    str(self.config.state_dir),
                ]
            )
        self.assertEqual(nomination_code, 0)
        nomination = json.loads(output.getvalue())["result"]
        self.assertEqual(
            nomination["local_metrics"]["eligibility"],
            "PRIMARY_SUBMISSION",
        )
        self.assertEqual(
            nomination["artifact_id"], result.promotion.artifact_id
        )

        checkpoints = list(
            self.config.checkpoint_dir.glob(f"{request.controller_run_id}-*")
        )
        self.assertEqual(len(checkpoints), 1)
        self.assertTrue((checkpoints[0] / "campaign.sqlite3").is_file())
        manifest = json.loads(
            (checkpoints[0] / "manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["campaign_id"], "campaign-1")
        self.assertEqual(
            manifest["databases"]["campaign"]["integrity_check"], "ok"
        )
        verified = verify_checkpoint(checkpoints[0])
        self.assertEqual(
            verified["databases"]["campaign"]["integrity_check"], "ok"
        )

    def test_supervisor_executes_and_advances_only_proven_lineage(self) -> None:
        environment = ResearchController(
            self.config
        )._resolved_execution_environment(LEGACY_RESEARCH_NAMESPACE)
        artifact_id = ArtifactId.source_sha256(SEED_HASH)
        history_baseline = BaselineRef.create(
            namespace=LEGACY_RESEARCH_NAMESPACE,
            artifact_id=artifact_id,
            source="deployment",
            revision="resolved-history-seed",
            execution_environment=environment,
        )
        identity = ExperimentIdentity.create(
            namespace=LEGACY_RESEARCH_NAMESPACE,
            mode="DISCOVERY",
            candidate_artifact_id=artifact_id,
            parent_artifact_id=artifact_id,
            baseline=history_baseline,
            execution_environment=environment,
            stage="baseline",
            suite="full",
            replicate_kind="primary",
            replicate_index=0,
            run_id="resolved-history-seed",
            iteration=0,
        )
        with HistoryStore(
            self.config.state_dir / "history.sqlite3",
            state_dir=self.config.state_dir,
        ) as history:
            history.ensure_namespace(
                LEGACY_RESEARCH_NAMESPACE.namespace_id,
                LEGACY_RESEARCH_NAMESPACE.to_dict(),
            )
            history.record_experiment(
                candidate_source=SEED,
                status="SUCCESS",
                backend="c500",
                suite="full",
                promotable=True,
                aggregate_score=1.0,
                identity=identity,
                result={
                    "status": "SUCCESS",
                    "promotion": {"phase": "baseline", "confirmed": True},
                },
            )
        campaign_seed = BaselineRef.create(
            namespace=LEGACY_RESEARCH_NAMESPACE,
            artifact_id=artifact_id,
            source="campaign",
            revision="campaign-resolved-seed",
            execution_environment=environment,
        )
        campaign = self.campaign_store.create_campaign(
            campaign_id="campaign-1",
            namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
            mode="DISCOVERY",
            snapshot=self.snapshot,
            budget_limit=BudgetAmount(
                candidates=5,
                wall_ms=300_000,
                gpu_ms=300_000,
                tokens=8_000_000,
                cost_microusd=5_000_000,
            ),
            initial_baseline_ref=campaign_seed,
            initial_policy_snapshot={"source": "deployment"},
            allow_staged_lineage=True,
        )
        self.campaign_store.start_campaign("campaign-1")
        trusted_runner = ResearchControllerRunner(
            self.config,
            campaign_store=self.campaign_store,
            fence_lookup=self.campaign_store.get_active_resource_lease,
            controller_factory=self.runner.controller_factory,
            clock=lambda: 100.0,
        )
        supervisor = CampaignSupervisor(
            self.campaign_store,
            trusted_runner,
            resource_id="gpu1",
            lease_ttl_seconds=120,
            clock=lambda: 100.0,
        )
        reservation = trusted_child_reservation(
            proposer_profile=self.proposer_profile(),
            campaign_remaining=BudgetAmount.from_mapping(
                self.campaign_store.budget_status("campaign-1")["remaining"]
            ),
            max_candidates=1,
            max_wall_seconds=60,
            max_consecutive_failures=3,
        )

        report = supervisor.execute_child(
            "campaign-1",
            ChildRunPlan(
                controller_run_id="controller-child-1",
                proposer_profile=self.proposer_profile(),
                reservation=reservation,
                max_candidates=1,
                max_wall_seconds=60,
                max_consecutive_failures=3,
            ),
        )

        self.assertEqual(report["disposition"], "PROMOTED", report)
        self.assertEqual(report["campaign"]["status"], "PAUSED_OPERATOR")
        promotion_value = report["child"]["result"]["promotion"]
        proven = PromotionEvidence.from_mapping(promotion_value)
        environment_value = proven.execution_environment
        components = dict(environment_value["components"])
        components["framework_digest"] = "sha256:" + "f" * 64
        mismatched_environment = ExecutionEnvironmentDigest.resolved(
            **components
        )
        parent = BaselineRef.from_value(dict(proven.parent_baseline_ref))
        mismatched_parent = BaselineRef.create(
            namespace=parent.namespace_id,
            artifact_id=parent.artifact_id,
            source=parent.source,
            revision=parent.revision,
            execution_environment=mismatched_environment,
        )
        with self.assertRaisesRegex(
            DataIntegrityRunnerFailure, "History identity differs"
        ):
            trusted_runner.prove_staged_promotion(
                campaign_id="campaign-1",
                controller_run_id="controller-child-1",
                evidence=PromotionEvidence(
                    artifact_id=proven.artifact_id,
                    namespace_id=proven.namespace_id,
                    parent_baseline_ref=mismatched_parent.to_dict(),
                    execution_environment=mismatched_environment.to_dict(),
                    primary_experiment_uid=proven.primary_experiment_uid,
                    confirmation_experiment_uid=(
                        proven.confirmation_experiment_uid
                    ),
                    checkpoint_digest=proven.checkpoint_digest,
                    policy_snapshot=proven.policy_snapshot,
                ),
            )
        revision = supervisor.advance_staged_lineage(
            "campaign-1",
            child_id=int(report["child"]["id"]),
            idempotency_key="advance-controller-child-1",
        )
        self.assertEqual(
            revision["parent_revision_id"],
            campaign["active_baseline_revision_id"],
        )
        self.assertEqual(
            revision["artifact_id"],
            report["child"]["result"]["promotion"]["artifact_id"],
        )

    def test_trusted_runner_rejects_python_under_reservation_before_controller(
        self,
    ) -> None:
        self.campaign_store.create_campaign(
            campaign_id="campaign-1",
            namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
            mode="DISCOVERY",
            snapshot=self.snapshot,
            budget_limit=BudgetAmount(
                candidates=5,
                wall_ms=300_000,
                gpu_ms=300_000,
                tokens=8_000_000,
                cost_microusd=5_000_000,
            ),
            initial_artifact_id=str(ArtifactId.source_sha256(SEED_HASH)),
            initial_policy_snapshot={"source": "deployment"},
            allow_staged_lineage=False,
        )
        self.campaign_store.start_campaign("campaign-1")
        supervisor = CampaignSupervisor(
            self.campaign_store,
            self.runner,
            resource_id="gpu1",
            lease_ttl_seconds=120,
            clock=lambda: 100.0,
        )

        report = supervisor.execute_child(
            "campaign-1",
            ChildRunPlan(
                controller_run_id="underreported-python-plan",
                proposer_profile=self.proposer_profile(),
                reservation=BudgetAmount(
                    candidates=1,
                    wall_ms=60_000,
                    gpu_ms=1,
                    tokens=1,
                    cost_microusd=1,
                ),
                max_candidates=1,
                max_wall_seconds=60,
                max_consecutive_failures=3,
            ),
        )

        self.assertEqual(report["disposition"], "DATA_INTEGRITY")
        self.assertEqual(
            report["campaign"]["status"], "PAUSED_DATA_INTEGRITY"
        )
        self.assertEqual(self.factories, [])
        self.assertEqual(self.evaluator.stages, [])
        self.assertEqual(
            report["child"]["result"]["details"]["error_type"],
            "DataIntegrityRunnerFailure",
        )

    def test_missing_usage_fields_are_charged_at_each_reserved_ceiling(
        self,
    ) -> None:
        self.provision_campaign(self.snapshot)
        request = self.request()
        self.runner.run_child(request)
        with ControllerStore(self.runner.controller_db) as store:
            store.connection.execute(
                """
                UPDATE evaluation_attempts SET started_at = NULL
                WHERE run_id = ?
                """,
                (request.controller_run_id,),
            )
            store.connection.execute(
                """
                UPDATE proposal_attempts
                SET input_tokens = NULL, output_tokens = NULL, cost_usd = NULL
                WHERE run_id = ?
                """,
                (request.controller_run_id,),
            )
            store.connection.commit()
            run = store.get_run(request.controller_run_id)

        usage, details = self.runner._usage(
            request,
            {**run, "valid_candidates": None, "updated_at": None},
        )

        self.assertEqual(usage, request.budget_limit)
        self.assertEqual(
            set(details["usage_unavailable"]),
            {"candidates", "wall_ms", "gpu_ms", "tokens", "cost_microusd"},
        )

    def test_cli_has_no_caller_supplied_reservation_and_rejects_low_budget(
        self,
    ) -> None:
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            campaign_cli.build_parser().parse_args(
                [
                    "child",
                    "execute",
                    "--database",
                    str(self.campaign_store.path),
                    "--config",
                    str(self.root / "unused-config.json"),
                    "--campaign-id",
                    "campaign-1",
                    "--controller-run-id",
                    "cli-child",
                    "--proposer-profile",
                    str(self.root / "proposer.json"),
                    "--reserved-gpu-ms",
                    "1",
                ]
            )

        self.campaign_store.create_campaign(
            campaign_id="campaign-1",
            namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
            mode="DISCOVERY",
            snapshot=self.snapshot,
            budget_limit=BudgetAmount(
                candidates=1,
                wall_ms=60_000,
                gpu_ms=60_000,
                tokens=7_999_999,
                cost_microusd=5_000_000,
            ),
            initial_artifact_id=str(ArtifactId.source_sha256(SEED_HASH)),
            initial_policy_snapshot={"source": "deployment"},
            allow_staged_lineage=False,
        )
        self.campaign_store.start_campaign("campaign-1")
        profile_path = self.root / "proposer.json"
        profile_path.write_text(
            json.dumps(self.proposer_profile()), encoding="utf-8"
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = campaign_main(
                [
                    "child",
                    "execute",
                    "--database",
                    str(self.campaign_store.path),
                    "--config",
                    str(self.root / "unused-config.json"),
                    "--campaign-id",
                    "campaign-1",
                    "--controller-run-id",
                    "cli-child",
                    "--proposer-profile",
                    str(profile_path),
                    "--max-candidates",
                    "1",
                    "--max-wall-seconds",
                    "60",
                ]
            )
        self.assertEqual(code, 2)
        self.assertEqual(stdout.getvalue(), "")
        failure = json.loads(stderr.getvalue())
        self.assertIn("trusted child worst case", failure["error"])
        self.assertEqual(
            self.campaign_store.list_child_runs("campaign-1"), []
        )
        self.assertFalse(self.runner.controller_db.exists())

    def test_run_child_never_takes_over_an_existing_intent(self) -> None:
        self.provision_campaign(self.snapshot)
        request = self.request()
        self.runner.run_child(request)
        factory_count = len(self.factories)

        with self.assertRaisesRegex(
            DataIntegrityRunnerFailure, "existing Controller intent"
        ):
            self.runner.run_child(request)

        self.assertEqual(len(self.factories), factory_count)

    def test_recovery_absence_never_creates_or_invokes_controller(self) -> None:
        self.provision_campaign(self.snapshot, controller_run_id="missing-run")
        result = self.runner.recover_child(
            self.request(recovery=True, run_id="missing-run")
        )

        self.assertIsNone(result)
        self.assertEqual(self.factories, [])
        self.assertFalse(self.runner.controller_db.exists())

    def test_campaign_checkpoint_requires_its_conventional_database(self) -> None:
        self.provision_campaign(self.snapshot)
        request = self.request()
        context = self.runner._validate_request(request, recovery=False)
        bounded = self.runner._bounded_config(request, context["proposer"])
        baseline = self.runner._campaign_baseline(request)
        snapshot = self.runner._workflow_snapshot(
            request,
            bounded_config=bounded,
            proposer=context["proposer"],
            baseline_ref=baseline,
            history_cutoff=1,
        )
        with ControllerStore(self.runner.controller_db) as store:
            store.create_run(
                run_id=request.controller_run_id,
                deadline_epoch=200,
                config=bounded.redacted_dict(),
                initial_best_hash=SEED_HASH,
                namespace_id=request.namespace_id,
                resolved_config_digest=snapshot["snapshot_digest"],
                workflow_snapshot=snapshot,
                baseline_ref=baseline.to_dict(),
                history_cutoff=1,
            )
        controller = self.runner._controller(bounded)
        self.campaign_store.close()
        self.campaign_store.path.rename(self.root / "nonconventional-campaign.sqlite3")

        with self.assertRaisesRegex(
            ControlledRuntimeError, "exactly one conventional"
        ):
            controller.checkpoint(request.controller_run_id)

        self.assertEqual(list(self.config.checkpoint_dir.iterdir()), [])

    def test_untrusted_harness_and_stale_fence_fail_closed(self) -> None:
        with self.assertRaisesRegex(
            DataIntegrityRunnerFailure, "safe local container"
        ):
            self.runner.run_child(self.request(run_id="../escape"))
        snapshot_with_secret = {**self.snapshot, "api_key": "must-not-persist"}
        with self.assertRaisesRegex(
            DataIntegrityRunnerFailure, "trusted schema"
        ):
            self.runner.run_child(self.request(snapshot=snapshot_with_secret))
        with self.assertRaisesRegex(
            DataIntegrityRunnerFailure, "OpenCode/DeepSeek"
        ):
            self.runner.run_child(
                self.request(proposer_profile=self.proposer_profile(harness="direct"))
            )
        self.assertFalse(self.runner.controller_db.exists())

        self.provision_campaign(self.snapshot)
        stale_request = self.request()
        self.lease = ResourceLease(
            resource_id="gpu1",
            campaign_id="campaign-1",
            fencing_epoch=8,
            expires_epoch=1_000.0,
            status="ACTIVE",
        )
        with self.assertRaisesRegex(
            DataIntegrityRunnerFailure, "fencing token"
        ):
            self.runner.run_child(stale_request)
        self.assertFalse(self.runner.controller_db.exists())

    def test_benchmark_snapshot_freezes_cohort_fields(self) -> None:
        proposer = BUILTIN_PROFILE_REGISTRY.get(
            kind="proposer",
            profile_id="opencode-deepseek-v4-pro",
            revision="v1",
        ).ref
        frozen_feedback = [
            {
                "run_id": "frozen-prior-run",
                "iteration_index": 1,
                "parent_hash": SEED_HASH,
                "candidate_hash": "f" * 64,
                "hypothesis": "frozen hypothesis",
                "outcome": "QUICK_REJECTED",
                "result_summary": {"status": "WRONG_ANSWER", "cases": []},
            }
        ]
        feedback_digest = canonical_sha256(frozen_feedback)
        plan = BenchmarkCohortPlan(
            cohort_id="cohort-1",
            namespace=LEGACY_RESEARCH_NAMESPACE,
            baseline=BaselineRef.from_value(self.baseline_ref),
            history_cutoff=1,
            prompt_protocol_digest=CURRENT_PROMPT_PROTOCOL_DIGEST,
            feedback_snapshot_digest=feedback_digest,
            arms=(
                BenchmarkArm("pro-a", proposer),
                BenchmarkArm("pro-b", proposer),
            ),
            repetitions=1,
            max_candidates=1,
            max_wall_seconds=60,
            max_consecutive_failures=3,
        )
        benchmark = bind_benchmark_campaign_snapshot(legacy_campaign_snapshot(
            mode="BENCHMARK",
            cohort_id="cohort-1",
            history_cutoff=1,
            prompt_protocol_digest=CURRENT_PROMPT_PROTOCOL_DIGEST,
            feedback_snapshot_digest=feedback_digest,
        ), plan, feedback_snapshot=frozen_feedback)
        assignment = plan.schedule()[0]
        run_id = benchmark_controller_run_id("campaign-1", plan, assignment)
        self.provision_campaign(
            benchmark,
            controller_run_id=run_id,
            initial_baseline_ref=plan.baseline,
        )
        request = self.request(snapshot=benchmark, run_id=run_id)

        result = self.runner.run_child(request)

        self.assertEqual(result.outcome, RunnerOutcome.PROMOTED)
        with ControllerStore(self.runner.controller_db) as store:
            run = store.get_run(request.controller_run_id)
            proposal_attempts = store.list_proposal_attempts(
                request.controller_run_id
            )
            evaluation_attempts = store.list_evaluation_attempts(
                request.controller_run_id
            )
        workflow = run["workflow_snapshot"]
        self.assertEqual(workflow["mode"], "BENCHMARK")
        self.assertEqual(workflow["cohort_id"], "cohort-1")
        self.assertEqual(
            workflow["benchmark_plan_digest"], plan.snapshot["snapshot_digest"]
        )
        self.assertEqual(
            workflow["campaign_child"]["baseline_revision_id"],
            request.baseline_revision_id,
        )
        self.assertEqual(run["baseline_ref"], plan.baseline.to_dict())
        self.assertNotEqual(
            request.baseline_revision_id, plan.baseline.revision
        )
        self.assertEqual(run["history_cutoff"], 1)
        self.assertEqual(
            workflow["feedback_snapshot_digest"], feedback_digest
        )
        self.assertEqual(
            proposal_attempts[0]["request"]["recent_feedback_digest"],
            feedback_digest,
        )
        self.assertTrue(evaluation_attempts)
        self.assertTrue(
            all(
                attempt["request"]["feedback_digest"] == feedback_digest
                for attempt in evaluation_attempts
            )
        )

    def test_current_resolved_campaign_starts_as_comparable_without_relabeling(self) -> None:
        snapshot = legacy_campaign_snapshot(
            namespace=CURRENT_RESEARCH_NAMESPACE
        )
        baseline = self.record_resolved_baseline(CURRENT_RESEARCH_NAMESPACE)
        self.provision_campaign(snapshot, initial_baseline_ref=baseline)
        request = self.request(snapshot=snapshot)
        context = self.runner._validate_request(request, recovery=False)
        bounded = self.runner._bounded_config(request, context["proposer"])
        workflow = self.runner._workflow_snapshot(
            request,
            bounded_config=bounded,
            proposer=context["proposer"],
            baseline_ref=baseline,
            history_cutoff=self.runner._history_cutoff(),
        )
        self.assertEqual(
            workflow["namespace"], CURRENT_RESEARCH_NAMESPACE.to_dict()
        )
        self.assertTrue(workflow["scientifically_comparable"])
        self.assertIsNone(workflow["compatibility_mode"])
        self.assertEqual(
            workflow["execution_environment"],
            workflow["runtime_execution_environment"],
        )

        controller = FixtureController(
            bounded,
            evaluator=self.evaluator,
            runner=NoopRunner(),
            clock=lambda: 161.0,
        )
        run = controller.start_from_snapshot(
            run_id=request.controller_run_id,
            workflow_snapshot=workflow,
            baseline_ref=baseline,
            history_cutoff=workflow["history_cutoff"],
        )
        self.assertEqual(run["status"], "BUDGET_EXHAUSTED")
        self.assertEqual(
            run["namespace_id"], CURRENT_RESEARCH_NAMESPACE.namespace_id
        )

    def test_current_unknown_and_unregistered_targets_fail_closed(self) -> None:
        current_snapshot = legacy_campaign_snapshot(
            namespace=CURRENT_RESEARCH_NAMESPACE
        )
        self.provision_campaign(current_snapshot)
        with self.assertRaisesRegex(
            DataIntegrityRunnerFailure, "CURRENT Campaigns require a resolved"
        ):
            self.runner._validate_request(
                self.request(snapshot=current_snapshot), recovery=False
            )

        forged = dict(current_snapshot)
        forged_namespace = dict(forged["namespace"])
        forged_namespace["namespace_id"] = "sha256:" + "f" * 64
        forged["namespace"] = forged_namespace
        with self.assertRaisesRegex(
            DataIntegrityRunnerFailure, "active built-in target"
        ):
            self.runner._validate_request(
                replace(
                    self.request(snapshot=current_snapshot),
                    campaign_snapshot=forged,
                    campaign_snapshot_digest=canonical_sha256(forged),
                ),
                recovery=False,
            )

    def test_trusted_resume_doctor_binds_fixed_call_config_and_quarantine(self) -> None:
        self.campaign_store.create_campaign(
            campaign_id="campaign-1",
            namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
            mode="DISCOVERY",
            snapshot=self.snapshot,
            budget_limit=BudgetAmount(candidates=1),
            initial_artifact_id=str(ArtifactId.source_sha256(SEED_HASH)),
            initial_policy_snapshot={},
            allow_staged_lineage=False,
        )
        self.campaign_store.start_campaign("campaign-1")
        lease = self.campaign_store.acquire_resource(
            "campaign-1", resource_id="gpu1", ttl_seconds=60, now_epoch=100
        )
        self.campaign_store.release_resource(
            lease, quarantine=True, reason="hard fault"
        )
        self.campaign_store.pause_campaign(
            "campaign-1",
            status="PAUSED_HARD_FAILURE",
            reason="hard fault",
        )
        fixture = TrustedResumeDoctorFixture(self.config)
        evidence = trusted_resume_doctor(
            self.config,
            campaign_store=self.campaign_store,
            campaign=self.campaign_store.get_campaign("campaign-1"),
            resource_id="gpu1",
            controller_factory=lambda _config: fixture,
            clock=lambda: time.time() + 1,
        )
        self.assertEqual(
            fixture.calls,
            [
                {
                    "require_secret": True,
                    "allowed_best_hashes": None,
                    "run_id": fixture.calls[0]["run_id"],
                }
            ],
        )
        self.assertTrue(
            str(fixture.calls[0]["run_id"]).startswith("campaign-resume-")
        )
        self.assertEqual(
            fixture.guard_calls[0]["quarantine_fencing_epoch"],
            lease.fencing_epoch,
        )
        self.assertEqual(evidence["resource_id"], "gpu1")
        self.assertEqual(
            evidence["config_digest"], canonical_sha256(self.config.redacted_dict())
        )
        recorded = self.campaign_store.record_doctor_evidence(
            "campaign-1", evidence=evidence
        )
        self.assertEqual(
            recorded["quarantine_fencing_epoch"], lease.fencing_epoch
        )
        resumed = self.campaign_store.resume_campaign(
            "campaign-1", doctor_evidence_digest=recorded["digest"]
        )
        self.assertEqual(resumed["status"], "RUNNING")

    def test_failed_resume_doctor_and_arbitrary_digest_cannot_resume(self) -> None:
        self.campaign_store.create_campaign(
            campaign_id="campaign-1",
            namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
            mode="DISCOVERY",
            snapshot=self.snapshot,
            budget_limit=BudgetAmount(candidates=1),
            initial_artifact_id=str(ArtifactId.source_sha256(SEED_HASH)),
            initial_policy_snapshot={},
            allow_staged_lineage=False,
        )
        self.campaign_store.start_campaign("campaign-1")
        lease = self.campaign_store.acquire_resource(
            "campaign-1", resource_id="gpu1", ttl_seconds=60, now_epoch=100
        )
        self.campaign_store.release_resource(
            lease, quarantine=True, reason="unknown outcome"
        )
        self.campaign_store.pause_campaign(
            "campaign-1",
            status="PAUSED_UNKNOWN_OUTCOME",
            reason="unknown outcome",
        )
        failed = TrustedResumeDoctorFixture(self.config, status="FAILED")
        with self.assertRaisesRegex(ValueError, "did not pass"):
            trusted_resume_doctor(
                self.config,
                campaign_store=self.campaign_store,
                campaign=self.campaign_store.get_campaign("campaign-1"),
                resource_id="gpu1",
                controller_factory=lambda _config: failed,
            )
        with self.assertRaisesRegex(ValueError, "recorded by this Campaign"):
            self.campaign_store.resume_campaign(
                "campaign-1",
                doctor_evidence_digest="sha256:" + "d" * 64,
            )

    def test_production_resume_doctor_docker_hook_rechecks_quarantine(self) -> None:
        self.campaign_store.create_campaign(
            campaign_id="campaign-1",
            namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
            mode="DISCOVERY",
            snapshot=self.snapshot,
            budget_limit=BudgetAmount(candidates=1),
            initial_artifact_id=str(ArtifactId.source_sha256(SEED_HASH)),
            initial_policy_snapshot={},
            allow_staged_lineage=False,
        )
        self.campaign_store.start_campaign("campaign-1")
        lease = self.campaign_store.acquire_resource(
            "campaign-1", resource_id="gpu1", ttl_seconds=60, now_epoch=100
        )
        self.campaign_store.release_resource(
            lease, quarantine=True, reason="hard fault"
        )
        self.campaign_store.pause_campaign(
            "campaign-1", status="PAUSED_HARD_FAILURE", reason="hard fault"
        )
        command_runner = DockerDoctorRunner()

        def factory(config):  # type: ignore[no-untyped-def]
            controller = ResearchController(config, runner=command_runner)
            controller._verify_repository = lambda **_values: {}  # type: ignore[method-assign]
            controller._prepare_framework = lambda: None  # type: ignore[method-assign]
            controller._inspect_image = lambda image: {  # type: ignore[method-assign]
                "image": image,
                "available": True,
                "repo_digests": image,
                "error": None,
            }
            return controller

        with mock.patch.object(
            type(self.config), "validate_host", return_value=[]
        ):
            evidence = trusted_resume_doctor(
                self.config,
                campaign_store=self.campaign_store,
                campaign=self.campaign_store.get_campaign("campaign-1"),
                resource_id="gpu1",
                controller_factory=factory,
                clock=lambda: time.time() + 1,
            )
        self.assertEqual(evidence["status"], "SUCCESS")
        self.assertEqual(len(command_runner.calls), 1)
        self.assertEqual(
            command_runner.calls[0]["timeout_sec"], DOCTOR_TIMEOUT_SEC
        )

    def test_production_resume_doctor_rejects_state_change_before_launch(self) -> None:
        self.campaign_store.create_campaign(
            campaign_id="campaign-1",
            namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
            mode="DISCOVERY",
            snapshot=self.snapshot,
            budget_limit=BudgetAmount(candidates=1),
            initial_artifact_id=str(ArtifactId.source_sha256(SEED_HASH)),
            initial_policy_snapshot={},
            allow_staged_lineage=False,
        )
        self.campaign_store.start_campaign("campaign-1")
        lease = self.campaign_store.acquire_resource(
            "campaign-1", resource_id="gpu1", ttl_seconds=60, now_epoch=100
        )
        self.campaign_store.release_resource(
            lease, quarantine=True, reason="unknown outcome"
        )
        self.campaign_store.pause_campaign(
            "campaign-1",
            status="PAUSED_UNKNOWN_OUTCOME",
            reason="unknown outcome",
        )
        command_runner = DockerDoctorRunner()

        def stale_factory(config):  # type: ignore[no-untyped-def]
            self.campaign_store.connection.execute(
                "UPDATE campaigns SET status = 'RUNNING' WHERE id = 'campaign-1'"
            )
            self.campaign_store.connection.commit()
            controller = ResearchController(config, runner=command_runner)
            controller._verify_repository = lambda **_values: {}  # type: ignore[method-assign]
            controller._prepare_framework = lambda: None  # type: ignore[method-assign]
            controller._inspect_image = lambda image: {  # type: ignore[method-assign]
                "image": image,
                "available": True,
                "repo_digests": image,
                "error": None,
            }
            return controller

        with mock.patch.object(
            type(self.config), "validate_host", return_value=[]
        ), self.assertRaisesRegex(
            ControllerDataIntegrityError,
            "quarantine fence is stale or mismatched",
        ):
            trusted_resume_doctor(
                self.config,
                campaign_store=self.campaign_store,
                campaign=self.campaign_store.get_campaign("campaign-1"),
                resource_id="gpu1",
                controller_factory=stale_factory,
            )
        self.assertEqual(command_runner.calls, [])

    def test_resume_cli_runs_trusted_doctor_and_has_no_digest_escape_hatch(self) -> None:
        self.campaign_store.create_campaign(
            campaign_id="campaign-1",
            namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
            mode="DISCOVERY",
            snapshot=self.snapshot,
            budget_limit=BudgetAmount(candidates=1),
            initial_artifact_id=str(ArtifactId.source_sha256(SEED_HASH)),
            initial_policy_snapshot={},
            allow_staged_lineage=False,
        )
        self.campaign_store.start_campaign("campaign-1")
        lease = self.campaign_store.acquire_resource(
            "campaign-1", resource_id="gpu1", ttl_seconds=60, now_epoch=100
        )
        self.campaign_store.release_resource(
            lease, quarantine=True, reason="hard fault"
        )
        self.campaign_store.pause_campaign(
            "campaign-1", status="PAUSED_HARD_FAILURE", reason="hard fault"
        )
        config_path = self.root / "campaign-resume-config.json"
        config_path.write_text("{}", encoding="utf-8")
        environment = ResearchController(
            self.config
        )._resolved_execution_environment(LEGACY_RESEARCH_NAMESPACE)
        evidence = {
            "schema_version": 1,
            "kind": "CAMPAIGN_RESUME_DOCTOR",
            "status": "SUCCESS",
            "campaign_id": "campaign-1",
            "resource_id": "gpu1",
            "observed_epoch": time.time() + 1,
            "namespace_id": LEGACY_RESEARCH_NAMESPACE.namespace_id,
            "config_digest": canonical_sha256(self.config.redacted_dict()),
            "execution_environment": environment.to_dict(),
            "doctor_result": {
                "status": "SUCCESS",
                "c500_probe": {
                    "environment": {"compile_probe_status": "PASSED"}
                },
            },
        }
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            campaign_cli, "trusted_resume_doctor", return_value=evidence
        ) as doctor, mock.patch.object(
            campaign_cli.ControllerConfig, "load", return_value=self.config
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            code = campaign_main(
                [
                    "resume",
                    "--database",
                    str(self.campaign_store.path),
                    "--campaign-id",
                    "campaign-1",
                    "--config",
                    str(config_path),
                    "--resource-id",
                    "gpu1",
                ]
            )
        self.assertEqual((code, stderr.getvalue()), (0, ""))
        self.assertEqual(json.loads(stdout.getvalue())["result"]["status"], "RUNNING")
        self.assertEqual(doctor.call_count, 1)
        recorded = self.campaign_store.get_doctor_evidence(
            self.campaign_store.get_campaign("campaign-1")[
                "pause_evidence_digest"
            ]
        )
        self.assertEqual(recorded["resource_id"], "gpu1")
        self.assertEqual(
            recorded["quarantine_fencing_epoch"], lease.fencing_epoch
        )
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            campaign_cli.build_parser().parse_args(
                [
                    "resume",
                    "--database",
                    str(self.campaign_store.path),
                    "--campaign-id",
                    "campaign-1",
                    "--doctor-evidence-digest",
                    "sha256:" + "d" * 64,
                ]
            )

    def test_campaign_start_doctor_must_fit_frozen_child_deadline(self) -> None:
        self.provision_campaign(self.snapshot)
        request = self.request()
        context = self.runner._validate_request(request, recovery=False)
        bounded = self.runner._bounded_config(request, context["proposer"])
        baseline = self.runner._campaign_baseline(request)
        snapshot = self.runner._workflow_snapshot(
            request,
            bounded_config=bounded,
            proposer=context["proposer"],
            baseline_ref=baseline,
            history_cutoff=1,
        )
        controller = CountingDoctorController(
            bounded,
            evaluator=self.evaluator,
            runner=NoopRunner(),
            proposer_factory=lambda *_args: StaticProposer(self.proposal),
            clock=lambda: 100.0,
            trusted_action_timeout=lambda action: (
                61.0 if action == "doctor" else None
            ),
        )

        run = controller.start_from_snapshot(
            run_id=request.controller_run_id,
            workflow_snapshot=snapshot,
            baseline_ref=baseline,
            history_cutoff=1,
        )

        self.assertEqual(run["status"], "BUDGET_EXHAUSTED")
        self.assertEqual(run["deadline_epoch"], request.child_deadline_epoch)
        self.assertEqual(run["consecutive_failures"], 0)
        self.assertEqual(controller.doctor_calls, 0)

    def test_campaign_resume_doctor_rechecks_deadline_before_launch(self) -> None:
        self.provision_campaign(self.snapshot)
        request = self.request(recovery=True)
        context = self.runner._validate_request(request, recovery=True)
        bounded = self.runner._bounded_config(request, context["proposer"])
        baseline = self.runner._campaign_baseline(request)
        snapshot = self.runner._workflow_snapshot(
            request,
            bounded_config=bounded,
            proposer=context["proposer"],
            baseline_ref=baseline,
            history_cutoff=1,
        )
        controller = CountingDoctorController(
            bounded,
            evaluator=self.evaluator,
            runner=NoopRunner(),
            proposer_factory=lambda *_args: StaticProposer(self.proposal),
            clock=lambda: 100.0,
            trusted_action_timeout=lambda action: (
                61.0 if action == "doctor" else None
            ),
        )
        with ControllerStore(controller.controller_db) as store:
            store.create_run(
                run_id=request.controller_run_id,
                deadline_epoch=request.child_deadline_epoch,
                config=bounded.redacted_dict(),
                initial_best_hash=SEED_HASH,
                namespace_id=request.namespace_id,
                resolved_config_digest=snapshot["snapshot_digest"],
                workflow_snapshot=snapshot,
                baseline_ref=baseline.to_dict(),
                history_cutoff=1,
            )

        run = controller.resume(request.controller_run_id)

        self.assertEqual(run["status"], "BUDGET_EXHAUSTED")
        self.assertEqual(run["consecutive_failures"], 0)
        self.assertEqual(controller.doctor_calls, 0)

    def test_hard_terminal_subclasses_are_mapped_without_reexecution(self) -> None:
        self.provision_campaign(self.snapshot, reserve=False)
        cases = (
            ("unknown-run", "UNKNOWN_GPU_OUTCOME", UnknownRunnerOutcome),
            ("hard-run", "HARD_FAILURE", HardRunnerFailure),
            (
                "integrity-run",
                "DATA_INTEGRITY_FAILURE",
                DataIntegrityRunnerFailure,
            ),
        )
        for run_id, outcome, failure in cases:
            with self.subTest(outcome=outcome):
                reservation = self.request(run_id=run_id).budget_limit
                key = child_budget_idempotency_key(run_id)
                self.campaign_store.reserve_budget(
                    "campaign-1",
                    idempotency_key=key,
                    action_kind="CHILD_RUN",
                    amount=reservation,
                )
                try:
                    request = self.persist_hard_terminal(
                        run_id=run_id, iteration_outcome=outcome
                    )
                    stage_count = len(self.evaluator.stages)
                    with self.assertRaises(failure):
                        self.runner.recover_child(request)
                    self.assertEqual(len(self.evaluator.stages), stage_count)
                finally:
                    self.campaign_store.cancel_budget(
                        "campaign-1", idempotency_key=key
                    )


if __name__ == "__main__":
    unittest.main()
