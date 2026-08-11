from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from kernel_research.autorun.controller import (
    ActionBudgetExhausted,
    ControllerDataIntegrityError,
    DockerEvaluator,
    ResearchController,
)
from kernel_research.autorun.models import ProposalV1
from kernel_research.autorun.proposal import ProposalRequest, Proposer
from kernel_research.autorun.store import ControllerStore
from kernel_research.campaign.controller_runner import legacy_campaign_snapshot
from kernel_research.campaign.models import BudgetAmount
from kernel_research.campaign.store import CampaignStore
from kernel_research.platform.artifacts import ArtifactId
from kernel_research.platform.identity import BaselineRef
from kernel_research.platform.profiles import LEGACY_RESEARCH_NAMESPACE

from tests.test_autorun import (
    FakeEvaluator,
    NoopRunner,
    SEED,
    SEED_HASH,
    _baseline,
    _config,
    _proposal_value,
    _with_different_block_size_n,
)


class CountingProposer(Proposer):
    def __init__(self, proposal: ProposalV1) -> None:
        self.proposal = proposal
        self.calls = 0

    def propose(self, request: ProposalRequest) -> ProposalV1:
        self.calls += 1
        if request.parent_candidate_hash != self.proposal.parent_candidate_hash:
            raise AssertionError("unexpected proposal parent")
        return self.proposal


class DeadlineAndFencingTests(unittest.TestCase):
    def test_docker_doctor_guard_runs_immediately_before_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root)

            class NeverRunner:
                def __init__(self) -> None:
                    self.calls = 0

                def run(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
                    self.calls += 1
                    raise AssertionError("Docker runner must not start")

            runner = NeverRunner()

            def reject(_run_id: str, _action: str, _timeout: float) -> None:
                raise ActionBudgetExhausted("fixture deadline")

            evaluator = DockerEvaluator(
                config,
                controller_dir=config.controller_dir,
                runner=runner,  # type: ignore[arg-type]
                before_container_start=reject,
            )

            with self.assertRaisesRegex(
                ActionBudgetExhausted, "fixture deadline"
            ):
                evaluator.doctor(run_id="campaign-run")
            self.assertEqual(runner.calls, 0)

    def _controller_fixture(
        self,
        root: Path,
        *,
        timeout,
    ) -> tuple[
        ResearchController,
        CountingProposer,
        FakeEvaluator,
    ]:
        config = _config(root, max_candidates=1)
        (config.repository_dir / "program.md").write_text(
            "Change only kernel.py.", encoding="utf-8"
        )
        _baseline(config.state_dir)
        proposal = ProposalV1.from_value(
            _proposal_value(_with_different_block_size_n(SEED)),
            expected_parent_hash=SEED_HASH,
        )
        proposer = CountingProposer(proposal)
        evaluator = FakeEvaluator()
        controller = ResearchController(
            config,
            evaluator=evaluator,
            runner=NoopRunner(),
            proposer_factory=lambda *_args: proposer,
            clock=lambda: 100.0,
            trusted_action_timeout=timeout,
        )
        return controller, proposer, evaluator

    @staticmethod
    def _create_legacy_run(
        controller: ResearchController, *, deadline_epoch: float
    ) -> str:
        run_id = "deadline-run"
        with ControllerStore(controller.controller_db) as store:
            store.create_run(
                run_id=run_id,
                deadline_epoch=deadline_epoch,
                config=controller.config.redacted_dict(),
                initial_best_hash=SEED_HASH,
            )
        return run_id

    def test_proposer_timeout_does_not_start_or_consume_failure_budget(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller, proposer, evaluator = self._controller_fixture(
                Path(temporary),
                timeout=lambda action: 10.0 if action == "proposer" else None,
            )
            run_id = self._create_legacy_run(
                controller, deadline_epoch=105.0
            )

            run = controller._run_loop(run_id)

            self.assertEqual(run["status"], "BUDGET_EXHAUSTED")
            self.assertEqual(run["consecutive_failures"], 0)
            self.assertEqual(proposer.calls, 0)
            self.assertEqual(evaluator.stages, [])
            with ControllerStore(controller.controller_db) as store:
                iteration = store.latest_iteration(run_id)
                attempts = store.list_proposal_attempts(run_id)
            assert iteration is not None
            self.assertEqual(iteration["outcome"], "BUDGET_EXHAUSTED")
            self.assertEqual(attempts[0]["status"], "FAILED")
            self.assertTrue(attempts[0]["result"]["launch_skipped"])

    def test_evaluator_timeout_keeps_attempt_pending_and_failure_budget_zero(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            controller, proposer, evaluator = self._controller_fixture(
                Path(temporary),
                timeout=lambda action: 10.0 if action == "evaluator" else None,
            )
            run_id = self._create_legacy_run(
                controller, deadline_epoch=105.0
            )

            run = controller._run_loop(run_id)

            self.assertEqual(run["status"], "BUDGET_EXHAUSTED")
            self.assertEqual(run["consecutive_failures"], 0)
            self.assertEqual(proposer.calls, 1)
            self.assertEqual(evaluator.stages, [])
            with ControllerStore(controller.controller_db) as store:
                iteration = store.latest_iteration(run_id)
                attempts = store.list_evaluation_attempts(run_id)
            assert iteration is not None
            self.assertEqual(iteration["outcome"], "BUDGET_EXHAUSTED")
            self.assertEqual(len(attempts), 1)
            self.assertEqual(attempts[0]["status"], "PENDING")

    def test_production_evaluator_guard_closes_as_budget_not_unknown_gpu(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root, max_candidates=1)
            (config.repository_dir / "program.md").write_text(
                "Change only kernel.py.", encoding="utf-8"
            )
            _baseline(config.state_dir)
            proposal = ProposalV1.from_value(
                _proposal_value(_with_different_block_size_n(SEED)),
                expected_parent_hash=SEED_HASH,
            )

            class NeverRunner:
                def __init__(self) -> None:
                    self.calls = 0

                def run(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
                    self.calls += 1
                    raise AssertionError("evaluator container must not start")

                def remove_exact_container(
                    self, *_args, **_kwargs
                ):  # type: ignore[no-untyped-def]
                    return None

            runner = NeverRunner()
            controller = ResearchController(
                config,
                runner=runner,  # type: ignore[arg-type]
                proposer_factory=lambda *_args: CountingProposer(proposal),
                clock=lambda: 100.0,
            )
            run_id = self._create_legacy_run(
                controller, deadline_epoch=105.0
            )

            run = controller._run_loop(run_id)

            self.assertEqual(run["status"], "BUDGET_EXHAUSTED")
            self.assertEqual(run["consecutive_failures"], 0)
            self.assertEqual(runner.calls, 0)
            with ControllerStore(controller.controller_db) as store:
                attempts = store.list_evaluation_attempts(run_id)
                iteration = store.latest_iteration(run_id)
            self.assertEqual(attempts[0]["status"], "FAILED")
            self.assertTrue(attempts[0]["result"]["launch_skipped"])
            assert iteration is not None
            self.assertEqual(iteration["outcome"], "BUDGET_EXHAUSTED")

    def test_production_proposer_guard_rejects_before_docker_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root, max_candidates=1)
            (config.repository_dir / "program.md").write_text(
                "Change only kernel.py.", encoding="utf-8"
            )
            _baseline(config.state_dir)

            class NeverRunner:
                def __init__(self) -> None:
                    self.calls = 0

                def run(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
                    self.calls += 1
                    raise AssertionError("proposer container must not start")

                def remove_exact_container(
                    self, *_args, **_kwargs
                ):  # type: ignore[no-untyped-def]
                    return None

            runner = NeverRunner()
            controller = ResearchController(
                config,
                evaluator=FakeEvaluator(),
                runner=runner,  # type: ignore[arg-type]
                clock=lambda: 100.0,
            )
            run_id = self._create_legacy_run(
                controller, deadline_epoch=105.0
            )

            run = controller._run_loop(run_id)

            self.assertEqual(run["status"], "BUDGET_EXHAUSTED")
            self.assertEqual(run["consecutive_failures"], 0)
            self.assertEqual(runner.calls, 0)
            with ControllerStore(controller.controller_db) as store:
                attempts = store.list_proposal_attempts(run_id)
            self.assertEqual(attempts[0]["status"], "FAILED")
            self.assertTrue(attempts[0]["result"]["launch_skipped"])

    def test_campaign_guard_reloads_live_child_and_exact_fencing_owner(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root, max_candidates=1)
            store = CampaignStore(root / "campaign.sqlite3")
            try:
                campaign_snapshot = legacy_campaign_snapshot()
                campaign = store.create_campaign(
                    campaign_id="campaign-deadline",
                    namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                    mode="DISCOVERY",
                    snapshot=campaign_snapshot,
                    budget_limit=BudgetAmount(
                        candidates=1,
                        wall_ms=60_000,
                        gpu_ms=60_000,
                        tokens=1_000_000,
                        cost_microusd=1_000_000,
                    ),
                    initial_artifact_id=str(
                        ArtifactId.source_sha256(SEED_HASH)
                    ),
                    initial_policy_snapshot={"source": "deployment"},
                    allow_staged_lineage=False,
                )
                store.start_campaign("campaign-deadline")
                child = store.create_child_run(
                    "campaign-deadline",
                    controller_run_id="campaign-controller-run",
                    proposer_profile={"fixture": True},
                    max_candidates=1,
                    max_wall_seconds=60,
                    max_consecutive_failures=1,
                )
                store.start_child_run(
                    int(child["id"]),
                    controller_run_id="campaign-controller-run",
                )
                lease = store.acquire_resource(
                    "campaign-deadline",
                    resource_id="gpu1",
                    ttl_seconds=100,
                    now_epoch=100.0,
                )
                controller = ResearchController(
                    config,
                    evaluator=FakeEvaluator(),
                    clock=lambda: 100.0,
                    trusted_action_timeout=lambda _action: 10.0,
                )
                baseline = BaselineRef.create(
                    namespace=LEGACY_RESEARCH_NAMESPACE,
                    artifact_id=ArtifactId.source_sha256(SEED_HASH),
                    source="campaign",
                    revision=campaign["active_baseline_revision_id"],
                )
                run = {
                    "id": "campaign-controller-run",
                    "deadline_epoch": 160.0,
                    "namespace_id": LEGACY_RESEARCH_NAMESPACE.namespace_id,
                    "baseline_ref": baseline.to_dict(),
                    "workflow_snapshot": {
                        "schema_version": 2,
                        "mode": "DISCOVERY",
                        "campaign_id": "campaign-deadline",
                        "campaign_snapshot_digest": campaign["snapshot_digest"],
                        "campaign_child": {
                            "child_id": int(child["id"]),
                            "child_index": int(child["child_index"]),
                            "controller_run_id": "campaign-controller-run",
                        },
                        "resource_lease": {
                            "resource_id": lease.resource_id,
                            "fencing_epoch": lease.fencing_epoch,
                            "expires_epoch": lease.expires_epoch,
                        },
                        "child_deadline_epoch": 160.0,
                        "budget": {"max_wall_seconds": 60},
                    },
                }

                controller._authorize_external_action(
                    "campaign-controller-run",
                    action="evaluator",
                    configured_timeout=None,
                    run_override=run,
                )
                store.release_resource(lease, reason="test fencing change")
                with self.assertRaisesRegex(
                    ControllerDataIntegrityError, "fencing ownership changed"
                ):
                    controller._authorize_external_action(
                        "campaign-controller-run",
                        action="evaluator",
                        configured_timeout=None,
                        run_override=run,
                    )
            finally:
                store.close()

    def test_campaign_window_shortage_is_normal_budget_exhaustion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(root, max_candidates=1)
            store = CampaignStore(root / "campaign.sqlite3")
            try:
                snapshot = legacy_campaign_snapshot()
                campaign = store.create_campaign(
                    campaign_id="campaign-short",
                    namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                    mode="DISCOVERY",
                    snapshot=snapshot,
                    budget_limit=BudgetAmount(
                        candidates=1,
                        wall_ms=60_000,
                        gpu_ms=60_000,
                        tokens=1_000_000,
                        cost_microusd=1_000_000,
                    ),
                    initial_artifact_id=str(
                        ArtifactId.source_sha256(SEED_HASH)
                    ),
                    initial_policy_snapshot={"source": "deployment"},
                    allow_staged_lineage=False,
                )
                store.start_campaign("campaign-short")
                child = store.create_child_run(
                    "campaign-short",
                    controller_run_id="short-run",
                    proposer_profile={"fixture": True},
                    max_candidates=1,
                    max_wall_seconds=60,
                    max_consecutive_failures=1,
                )
                store.start_child_run(
                    int(child["id"]), controller_run_id="short-run"
                )
                lease = store.acquire_resource(
                    "campaign-short",
                    resource_id="gpu1",
                    ttl_seconds=100,
                    now_epoch=100.0,
                )
                controller = ResearchController(
                    config,
                    evaluator=FakeEvaluator(),
                    clock=lambda: 100.0,
                    trusted_action_timeout=lambda _action: 10.0,
                )
                run = {
                    "id": "short-run",
                    "deadline_epoch": 105.0,
                    "namespace_id": LEGACY_RESEARCH_NAMESPACE.namespace_id,
                    "baseline_ref": BaselineRef.create(
                        namespace=LEGACY_RESEARCH_NAMESPACE,
                        artifact_id=ArtifactId.source_sha256(SEED_HASH),
                        source="campaign",
                        revision=campaign["active_baseline_revision_id"],
                    ).to_dict(),
                    "workflow_snapshot": {
                        "schema_version": 2,
                        "mode": "DISCOVERY",
                        "campaign_id": "campaign-short",
                        "campaign_snapshot_digest": campaign["snapshot_digest"],
                        "campaign_child": {
                            "child_id": int(child["id"]),
                            "child_index": int(child["child_index"]),
                            "controller_run_id": "short-run",
                        },
                        "resource_lease": {
                            "resource_id": lease.resource_id,
                            "fencing_epoch": lease.fencing_epoch,
                            "expires_epoch": lease.expires_epoch,
                        },
                        "child_deadline_epoch": 105.0,
                        "budget": {"max_wall_seconds": 60},
                    },
                }
                with self.assertRaisesRegex(
                    ActionBudgetExhausted, "does not fit"
                ):
                    controller._authorize_external_action(
                        "short-run",
                        action="doctor",
                        configured_timeout=None,
                        run_override=run,
                    )
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
