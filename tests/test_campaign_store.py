from __future__ import annotations

from pathlib import Path
import sqlite3
import tempfile
import time
import unittest

from kernel_research.campaign import BudgetAmount, CampaignStore
from kernel_research.platform.identity import BaselineRef, ExecutionEnvironmentDigest


NAMESPACE = "sha256:" + "a" * 64
OTHER_NAMESPACE = "sha256:" + "b" * 64
SEED = "source-sha256-v1:" + "1" * 64
CANDIDATE = "bundle-sha256-v1:" + "2" * 64
BUNDLE_OBJECT = "bundle-sha256-v1:" + "3" * 64
MANIFEST = "sha256:" + "4" * 64
DOCTOR = "sha256:" + "5" * 64
ENVIRONMENT = ExecutionEnvironmentDigest.resolved(
    evaluator_image_digest="sha256:" + "6" * 64,
    toolchain_digest="sha256:" + "7" * 64,
    framework_digest="sha256:" + "8" * 64,
    operator_abi_digest="sha256:" + "9" * 64,
    build_flags_digest="sha256:" + "0" * 64,
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


def doctor_evidence(
    *, campaign_id: str = "campaign", resource_id: str = "metax-c500:1"
) -> dict:
    return {
        "schema_version": 1,
        "kind": "CAMPAIGN_RESUME_DOCTOR",
        "status": "SUCCESS",
        "campaign_id": campaign_id,
        "resource_id": resource_id,
        "observed_epoch": time.time() + 1,
        "namespace_id": NAMESPACE,
        "config_digest": DOCTOR,
        "execution_environment": ENVIRONMENT.to_dict(),
        "doctor_result": {
            "status": "SUCCESS",
            "c500_probe": {
                "environment": {"compile_probe_status": "PASSED"}
            },
        },
    }


def create(store: CampaignStore, *, campaign_id: str = "campaign", mode: str = "DISCOVERY", lineage: bool = True):
    return store.create_campaign(
        campaign_id=campaign_id,
        namespace_id=NAMESPACE,
        mode=mode,
        snapshot={"profile": "frozen-v1"},
        budget_limit=BudgetAmount(
            candidates=10,
            wall_ms=1_000_000,
            gpu_ms=500_000,
            tokens=100_000,
            cost_microusd=5_000_000,
        ),
        initial_baseline_ref=seed_ref(campaign_id=campaign_id),
        initial_policy_snapshot={"policy": "deployment"},
        allow_staged_lineage=lineage,
    )


class CampaignStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.store = CampaignStore(Path(self.temporary.name) / "campaign.sqlite3")

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def test_campaign_freezes_namespace_snapshot_limits_and_seed(self) -> None:
        campaign = create(self.store)
        self.assertEqual(campaign["status"], "CREATED")
        self.assertEqual(campaign["namespace_id"], NAMESPACE)
        revisions = self.store.list_baseline_revisions("campaign")
        self.assertEqual(len(revisions), 1)
        self.assertEqual(revisions[0]["revision_kind"], "DEPLOYMENT_SEED")
        self.assertEqual(revisions[0]["artifact_id"], SEED)
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.connection.execute(
                "UPDATE campaigns SET namespace_id = ? WHERE id = 'campaign'",
                (OTHER_NAMESPACE,),
            )
        self.store.connection.rollback()

    def test_budget_reservation_is_worst_case_idempotent_and_settled(self) -> None:
        create(self.store)
        self.store.start_campaign("campaign")
        reserved = BudgetAmount(candidates=5, wall_ms=500_000, gpu_ms=250_000, tokens=50_000, cost_microusd=2_000_000)
        first = self.store.reserve_budget(
            "campaign", idempotency_key="run-1", action_kind="CHILD_RUN", amount=reserved
        )
        again = self.store.reserve_budget(
            "campaign", idempotency_key="run-1", action_kind="CHILD_RUN", amount=reserved
        )
        self.assertEqual(first["id"], again["id"])
        with self.assertRaisesRegex(ValueError, "frozen limit"):
            self.store.reserve_budget(
                "campaign",
                idempotency_key="too-large",
                action_kind="CHILD_RUN",
                amount=BudgetAmount(candidates=6, wall_ms=600_000),
            )
        settled = self.store.settle_budget(
            "campaign",
            idempotency_key="run-1",
            actual=BudgetAmount(candidates=2, wall_ms=100_000, gpu_ms=80_000, tokens=10_000, cost_microusd=500_000),
        )
        self.assertEqual(settled["status"], "SETTLED")
        self.assertEqual(
            self.store.budget_status("campaign")["committed_or_reserved"]["candidates"],
            2,
        )
        with self.assertRaisesRegex(ValueError, "immutable"):
            self.store.settle_budget(
                "campaign",
                idempotency_key="run-1",
                actual=BudgetAmount(candidates=6),
            )

    def test_child_runs_preserve_single_run_safety_and_pause_on_unknown(self) -> None:
        create(self.store)
        self.store.start_campaign("campaign")
        with self.assertRaisesRegex(ValueError, "at most six hours"):
            self.store.create_child_run(
                "campaign", proposer_profile={}, max_wall_seconds=21_601
            )
        child = self.store.create_child_run("campaign", proposer_profile={"id": "p"})
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.create_child_run("campaign", proposer_profile={"id": "other"})
        started = self.store.start_child_run(child["id"], controller_run_id="controller-1")
        self.assertTrue(started["stop_after_promotion"])
        lease = self.store.acquire_resource(
            "campaign",
            resource_id="metax-c500:1",
            ttl_seconds=60,
            now_epoch=100,
        )
        finished = self.store.finish_child_run(
            child["id"], status="UNKNOWN_GPU_OUTCOME", result={"candidate": CANDIDATE}
        )
        self.store.release_resource(
            lease, quarantine=True, reason="unknown GPU outcome"
        )
        self.assertEqual(finished["status"], "UNKNOWN_GPU_OUTCOME")
        self.assertEqual(
            self.store.get_campaign("campaign")["status"], "PAUSED_UNKNOWN_OUTCOME"
        )
        with self.assertRaisesRegex(ValueError, "doctor evidence"):
            self.store.resume_campaign("campaign")
        recorded = self.store.record_doctor_evidence(
            "campaign", evidence=doctor_evidence()
        )
        self.assertEqual(
            self.store.resume_campaign(
                "campaign", doctor_evidence_digest=recorded["digest"]
            )["status"],
            "RUNNING",
        )

    def test_doctor_clearance_is_bound_to_latest_quarantined_fence(self) -> None:
        create(self.store)
        self.store.start_campaign("campaign")
        lease_a = self.store.acquire_resource(
            "campaign", resource_id="resource-a", ttl_seconds=60, now_epoch=100
        )
        self.store.release_resource(
            lease_a, quarantine=True, reason="hard fault A"
        )
        self.store.pause_campaign(
            "campaign", status="PAUSED_HARD_FAILURE", reason="hard fault A"
        )
        evidence_a = self.store.record_doctor_evidence(
            "campaign",
            evidence=doctor_evidence(resource_id="resource-a"),
        )
        self.assertEqual(
            evidence_a["quarantine_fencing_epoch"], lease_a.fencing_epoch
        )
        self.store.resume_campaign(
            "campaign", doctor_evidence_digest=evidence_a["digest"]
        )

        lease_b = self.store.acquire_resource(
            "campaign", resource_id="resource-b", ttl_seconds=60, now_epoch=200
        )
        self.store.release_resource(
            lease_b, quarantine=True, reason="hard fault B"
        )
        self.store.pause_campaign(
            "campaign", status="PAUSED_HARD_FAILURE", reason="hard fault B"
        )
        with self.assertRaisesRegex(ValueError, "latest Campaign quarantine"):
            self.store.resume_campaign(
                "campaign", doctor_evidence_digest=evidence_a["digest"]
            )
        with self.assertRaisesRegex(ValueError, "latest Campaign quarantine"):
            self.store.record_doctor_evidence(
                "campaign",
                evidence=doctor_evidence(resource_id="resource-a"),
            )

        evidence_b = self.store.record_doctor_evidence(
            "campaign",
            evidence=doctor_evidence(resource_id="resource-b"),
        )
        self.assertEqual(
            evidence_b["quarantine_fencing_epoch"], lease_b.fencing_epoch
        )
        self.store.resume_campaign(
            "campaign", doctor_evidence_digest=evidence_b["digest"]
        )
        with self.assertRaisesRegex(ValueError, "trusted doctor clearance"):
            self.store.acquire_resource(
                "campaign", resource_id="resource-a", ttl_seconds=60, now_epoch=300
            )
        cleared = self.store.acquire_resource(
            "campaign", resource_id="resource-b", ttl_seconds=60, now_epoch=300
        )
        self.assertGreater(cleared.fencing_epoch, lease_b.fencing_epoch)

    def test_unknown_or_missing_seed_environment_cannot_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "resolved execution environment"):
            self.store.create_campaign(
                campaign_id="legacy-unknown",
                namespace_id=NAMESPACE,
                mode="DISCOVERY",
                snapshot={"profile": "legacy"},
                budget_limit=BudgetAmount(candidates=1),
                initial_artifact_id=SEED,
                initial_policy_snapshot={},
                allow_staged_lineage=True,
            )

        campaign = create(self.store)
        parent = campaign["active_baseline_revision_id"]
        self.store.connection.execute(
            "DROP TRIGGER baseline_revision_ref_immutable"
        )
        self.store.connection.execute(
            "UPDATE baseline_revisions SET baseline_ref_json = NULL WHERE id = ?",
            (parent,),
        )
        self.store.connection.commit()
        self.store.start_campaign("campaign")
        with self.assertRaisesRegex(ValueError, "legacy Campaign revision"):
            self.store.advance_baseline(
                "campaign",
                expected_parent_revision_id=parent,
                parent_baseline_ref=seed_ref(),
                artifact_id=CANDIDATE,
                primary_experiment_uid="old-primary",
                confirmation_experiment_uid="old-confirmation",
                evidence_namespace_id=NAMESPACE,
                evidence_execution_environment=ENVIRONMENT,
                policy_snapshot={},
                idempotency_key="old-revision-must-not-stage",
            )

    def test_resource_fencing_prevents_overlap_and_stale_release(self) -> None:
        create(self.store)
        self.store.start_campaign("campaign")
        first = self.store.acquire_resource(
            "campaign", resource_id="metax-c500:1", ttl_seconds=10, now_epoch=100
        )
        with self.assertRaisesRegex(ValueError, "active unexpired"):
            self.store.acquire_resource(
                "campaign", resource_id="metax-c500:1", ttl_seconds=10, now_epoch=105
            )
        second = self.store.acquire_resource(
            "campaign", resource_id="metax-c500:1", ttl_seconds=10, now_epoch=111
        )
        self.assertEqual(second.fencing_epoch, first.fencing_epoch + 1)
        with self.assertRaisesRegex(ValueError, "stale"):
            self.store.release_resource(first)
        quarantined = self.store.release_resource(
            second, quarantine=True, reason="ATU fault"
        )
        self.assertEqual(quarantined.status, "QUARANTINED")

    def test_lineage_uses_namespace_and_parent_compare_and_swap(self) -> None:
        campaign = create(self.store)
        parent = campaign["active_baseline_revision_id"]
        self.store.start_campaign("campaign")
        revision = self.store.advance_baseline(
            "campaign",
            expected_parent_revision_id=parent,
            parent_baseline_ref=seed_ref(),
            artifact_id=CANDIDATE,
            primary_experiment_uid="experiment-primary",
            confirmation_experiment_uid="experiment-confirmation",
            evidence_namespace_id=NAMESPACE,
            evidence_execution_environment=ENVIRONMENT,
            policy_snapshot={"threshold": 1.01},
            idempotency_key="promotion-1",
        )
        again = self.store.advance_baseline(
            "campaign",
            expected_parent_revision_id=parent,
            parent_baseline_ref=seed_ref(),
            artifact_id=CANDIDATE,
            primary_experiment_uid="experiment-primary",
            confirmation_experiment_uid="experiment-confirmation",
            evidence_namespace_id=NAMESPACE,
            evidence_execution_environment=ENVIRONMENT,
            policy_snapshot={"threshold": 1.01},
            idempotency_key="promotion-1",
        )
        self.assertEqual(revision["id"], again["id"])
        self.assertEqual(
            self.store.get_campaign("campaign")["active_baseline_revision_id"],
            revision["id"],
        )
        with self.assertRaisesRegex(ValueError, "another namespace"):
            self.store.advance_baseline(
                "campaign",
                expected_parent_revision_id=revision["id"],
                parent_baseline_ref=revision["baseline_ref"],
                artifact_id="bundle-sha256-v1:" + "6" * 64,
                primary_experiment_uid="p2",
                confirmation_experiment_uid="c2",
                evidence_namespace_id=OTHER_NAMESPACE,
                evidence_execution_environment=ENVIRONMENT,
                policy_snapshot={},
                idempotency_key="promotion-2",
            )

        create(self.store, campaign_id="benchmark", mode="BENCHMARK", lineage=False)
        self.store.start_campaign("benchmark")
        benchmark_parent = self.store.get_campaign("benchmark")["active_baseline_revision_id"]
        with self.assertRaisesRegex(ValueError, "benchmark"):
            self.store.advance_baseline(
                "benchmark",
                expected_parent_revision_id=benchmark_parent,
                parent_baseline_ref=seed_ref(campaign_id="benchmark"),
                artifact_id=CANDIDATE,
                primary_experiment_uid="bp",
                confirmation_experiment_uid="bc",
                evidence_namespace_id=NAMESPACE,
                evidence_execution_environment=ENVIRONMENT,
                policy_snapshot={},
                idempotency_key="no-advance",
            )

    def test_oj_feedback_is_whitelisted_and_cannot_move_baseline(self) -> None:
        campaign = create(self.store)
        baseline = campaign["active_baseline_revision_id"]
        nomination = self.store.create_oj_nomination(
            "campaign",
            nomination_type="PRIMARY_SUBMISSION",
            artifact_id=CANDIDATE,
            bundle_object_id=BUNDLE_OBJECT,
            manifest_digest=MANIFEST,
            local_metrics={"aggregate_speedup": 1.05},
            baseline_revision_id=baseline,
        )
        self.store.mark_oj_exported(nomination["id"])
        with self.assertRaisesRegex(ValueError, "whitelist"):
            self.store.record_oj_feedback(
                nomination["id"],
                submission_id="submission-1",
                verdict="PROMOTE_LOCALLY",
                score=100.0,
            )
        feedback = self.store.record_oj_feedback(
            nomination["id"],
            submission_id="submission-1",
            verdict="ACCEPTED",
            score=100.0,
            note="manual result",
        )
        self.assertEqual(feedback["status"], "FEEDBACK_RECORDED")
        self.assertEqual(
            self.store.get_campaign("campaign")["active_baseline_revision_id"], baseline
        )
        self.assertEqual(
            set(self.store.oj_agent_feedback(nomination["id"])),
            {
                "nomination_type",
                "artifact_id",
                "verdict",
                "score",
            },
        )
        self.assertNotIn(
            "manual result",
            self.store.oj_agent_feedback(nomination["id"]).values(),
        )


if __name__ == "__main__":
    unittest.main()
