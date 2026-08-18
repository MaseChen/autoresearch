from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch

import kernel_research.campaign.soak_collector as soak_collector
from kernel_research.autorun.models import ControllerConfig
from kernel_research.autorun.runtime import CommandResult
from kernel_research.autorun.store import ControllerStore
from kernel_research.campaign.models import BudgetAmount
from kernel_research.campaign.soak import SoakGate
from kernel_research.campaign.soak_collector import (
    COUNT_FIELDS,
    SoakObservationCollector,
)
from kernel_research.campaign.store import CampaignStore
from kernel_research.platform.canonical import canonical_json_text, canonical_sha256
from kernel_research.platform.identity import (
    BaselineRef,
    ExecutionEnvironmentDigest,
)
from kernel_research.profiler_contract import (
    PROFILER_ACTIVATION_PROFILE_DIGEST,
    PROFILER_BUILD_PROFILE_DIGEST,
    profiler_activation_profile_snapshot,
    profiler_build_profile_snapshot,
)


NAMESPACE_A = "sha256:" + "a" * 64
NAMESPACE_B = "sha256:" + "b" * 64
SOURCE = "source-sha256-v1:" + "1" * 64
CANDIDATE = "c" * 64
HARD_CANDIDATE = "d" * 64
ENVIRONMENT = ExecutionEnvironmentDigest.resolved(
    evaluator_image_digest="sha256:" + "6" * 64,
    toolchain_digest="sha256:" + "7" * 64,
    framework_digest="sha256:" + "8" * 64,
    operator_abi_digest="sha256:" + "9" * 64,
    build_flags_digest="sha256:" + "0" * 64,
)


def _utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )


class InventoryRunner:
    def __init__(self, names: tuple[str, ...] = ()) -> None:
        self.names = names
        self.fail = False
        self.timed_out = False
        self.output_limited = False
        self.returned_argv: tuple[str, ...] | None = None
        self.stdout_override: str | None = None
        self.git_head: str | None = None
        self.git_status = ""
        self.calls: list[tuple[str, ...]] = []

    def run(
        self,
        argv,
        *,
        input_text,
        timeout_sec,
        max_output_bytes,
        container_name=None,
        docker_binary=None,
    ) -> CommandResult:
        selected = tuple(argv)
        self.calls.append(selected)
        if selected[0] == "git" and "rev-parse" in selected:
            stdout = "" if self.git_head is None else self.git_head + "\n"
        elif selected[0] == "git" and "status" in selected:
            stdout = self.git_status
        else:
            stdout = "".join(f"{name}\n" for name in self.names)
        if self.stdout_override is not None:
            stdout = self.stdout_override
        return CommandResult(
            argv=self.returned_argv or selected,
            returncode=1 if self.fail else 0,
            stdout="" if self.fail else stdout,
            stderr="inventory failed" if self.fail else "",
            timed_out=self.timed_out,
            output_limited=self.output_limited,
        )


class CampaignSoakCollectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.repository = self.root / "repository"
        self.state = self.root / "state"
        self.controller_dir = self.root / "controller"
        self.checkpoints = self.root / "checkpoints"
        for directory in (
            self.repository,
            self.state,
            self.controller_dir,
            self.checkpoints,
        ):
            directory.mkdir()
        self.campaign_path = self.root / "campaign" / "campaign.sqlite3"
        self.campaign_path.parent.mkdir()
        self.campaign = CampaignStore(self.campaign_path)
        self.controller = ControllerStore(
            self.controller_dir / "controller.sqlite3"
        )
        self.runner = InventoryRunner()
        self.config = ControllerConfig(
            repository_dir=self.repository,
            state_dir=self.state,
            controller_dir=self.controller_dir,
            checkpoint_dir=self.checkpoints,
            docker_binary=self.root / "trusted-docker",
            proposer_image="proposer@sha256:" + "2" * 64,
            evaluator_image="evaluator@sha256:" + "3" * 64,
            deepseek_key_file=self.root / "secret",
            gpu_devices=(
                self.root / "gpu-a",
                self.root / "gpu-b",
                self.root / "gpu-c",
            ),
            evaluator_cache_dir=self.root / "cache",
            expected_git_commit="4" * 40,
            expected_kernel_hash="5" * 64,
        )
        self.collector = self._collector()

    def tearDown(self) -> None:
        if self.controller is not None:
            self.controller.close()
        if self.campaign is not None:
            self.campaign.close()
        self.temporary.cleanup()

    def _collector(self) -> SoakObservationCollector:
        return SoakObservationCollector(
            self.config,
            campaign_database=self.campaign_path,
            runner=self.runner,
            invariant_provider=lambda: {
                "git_commit": self.config.expected_git_commit,
                "tracked_and_untracked_worktree_clean": True,
            },
        )

    def _seed_campaign(self) -> str:
        campaign = self.campaign.create_campaign(
            campaign_id="campaign",
            namespace_id=NAMESPACE_A,
            mode="DISCOVERY",
            snapshot={"campaign": "frozen"},
            budget_limit=BudgetAmount(
                candidates=20,
                wall_ms=10_000_000,
                gpu_ms=10_000_000,
                tokens=1_000_000,
                cost_microusd=10_000_000,
            ),
            initial_artifact_id=SOURCE,
            initial_policy_snapshot={"policy": "frozen"},
            allow_staged_lineage=False,
        )
        return str(campaign["active_baseline_revision_id"])

    def _controller_run(self, run_id: str, namespace: str) -> None:
        baseline = BaselineRef.create(
            namespace=namespace,
            artifact_id=SOURCE,
            source="deployment",
            revision=f"{run_id}-seed",
        )
        self.controller.create_run(
            run_id=run_id,
            deadline_epoch=time.time() + 3600,
            config={"run": run_id},
            initial_best_hash="1" * 64,
            namespace_id=namespace,
            resolved_config_digest="sha256:" + "6" * 64,
            workflow_snapshot={"run": run_id},
            baseline_ref=baseline.to_dict(),
            history_cutoff=0,
        )

    def _iteration(
        self, run_id: str, index: int, candidate: str, outcome: str
    ) -> None:
        iteration = self.controller.create_iteration(
            run_id, index, parent_hash="1" * 64
        )
        self.controller.update_iteration(
            int(iteration["id"]),
            candidate_hash=candidate,
            outcome=outcome,
            status="COMPLETED",
            stage="DONE",
        )

    def test_clean_observation_is_canonical_and_uses_fixed_docker_argv(self) -> None:
        first = self.collector.collect(
            interval_start_epoch=1000, interval_end_epoch=1100
        )
        second = self.collector.collect(
            interval_start_epoch=1000, interval_end_epoch=1100
        )
        self.assertEqual(first.status, "AVAILABLE")
        self.assertEqual(first.evidence_id, second.evidence_id)
        self.assertTrue(all(value == 0 for value in first.counts.values()))
        self.assertRegex(first.evidence_id, r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(
            self.runner.calls[-1],
            (
                str(self.config.docker_binary),
                "ps",
                "-a",
                "--filter",
                "name=kar-",
                "--format",
                "{{.Names}}",
            ),
        )
        self.assertEqual(
            set(first.sources), {"campaign", "controller", "docker"}
        )
        for source in first.sources.values():
            self.assertRegex(source["digest"], r"^sha256:[0-9a-f]{64}$")
            self.assertIn("cursor", source)

    def test_actual_empty_ledgers_cannot_complete_twenty_four_hours(self) -> None:
        now = [1000.0]
        gate = SoakGate(
            self.campaign,
            gate_id="empty-runtime",
            collector=self.collector,
            clock=lambda: now[0],
        )
        gate.start()
        result = None
        for _ in range(24 * 60 * 60 // 300):
            now[0] += 300
            result = gate.heartbeat()
        assert result is not None
        self.assertEqual(result["outcome"], "PENDING_ACTIVITY")
        self.assertEqual(result["generation"]["status"], "ACTIVE")
        self.assertEqual(result["generation"]["accumulated_seconds"], 86400)

    def test_lease_overlap_orphan_container_and_reserved_budget_are_derived(self) -> None:
        self._seed_campaign()
        self.campaign.start_campaign("campaign")
        self.campaign.reserve_budget(
            "campaign",
            idempotency_key="orphan-reservation",
            action_kind="CHILD_RUN",
            amount=BudgetAmount(
                candidates=1,
                wall_ms=1000,
                gpu_ms=1000,
                tokens=1000,
                cost_microusd=1000,
            ),
        )
        self.campaign.connection.executemany(
            """
            INSERT INTO resource_leases(
                resource_id, fencing_epoch, campaign_id, status,
                acquired_at, expires_epoch, released_at, reason
            ) VALUES ('gpu1', ?, 'campaign', 'RELEASED', ?, ?, ?, 'test')
            """,
            (
                (1, _utc(1000), 2000, _utc(1800)),
                (2, _utc(1500), 2500, _utc(2200)),
            ),
        )
        self.campaign.connection.commit()
        self.runner.names = ("kar-orphan",)
        observation = self.collector.collect(
            interval_start_epoch=1000, interval_end_epoch=2500
        )
        self.assertEqual(observation.status, "AVAILABLE")
        self.assertEqual(observation.counts["lease_overlap_count"], 1)
        self.assertEqual(observation.counts["orphan_container_count"], 1)
        self.assertEqual(observation.counts["budget_leak_count"], 1)

    def test_abandoned_canary_reservation_requires_complete_trusted_proof(
        self,
    ) -> None:
        campaign_id = "profile-image-canary-abandoned"
        action_key = "profile-image-canary-v1"
        recipe_id = "metax-hardware-counters-v1"
        profiler_image = "profiler@sha256:" + "a" * 64
        build_digest = "sha256:" + "b" * 64
        activation_digest = "sha256:" + "c" * 64
        baseline = BaselineRef.create(
            namespace=NAMESPACE_A,
            artifact_id=SOURCE,
            source="campaign",
            revision="canary-deployment-seed-v2",
            execution_environment=ENVIRONMENT,
        )
        self.campaign.create_campaign(
            campaign_id=campaign_id,
            namespace_id=NAMESPACE_A,
            mode="DISCOVERY",
            snapshot={
                "kind": "PROFILE_IMAGE_CANARY",
                "profiler_image": profiler_image,
                "profiler_build_profile_digest": build_digest,
                "profiler_activation_profile_digest": activation_digest,
                "recipes": [recipe_id],
            },
            budget_limit=BudgetAmount(wall_ms=1_800_000, gpu_ms=900_000),
            initial_baseline_ref=baseline,
            initial_policy_snapshot={"kind": "PROFILE_IMAGE_CANARY"},
            allow_staged_lineage=False,
        )
        self.campaign.start_campaign(campaign_id)
        self.campaign.reserve_budget(
            campaign_id,
            idempotency_key=action_key,
            action_kind="PROFILE_IMAGE_CANARY",
            amount=BudgetAmount(wall_ms=1_800_000, gpu_ms=900_000),
        )
        lease = self.campaign.acquire_resource(
            campaign_id,
            resource_id="gpu1",
            ttl_seconds=1_830,
        )
        intent_material = {
            "schema_version": 1,
            "kind": "PROFILE_IMAGE_CANARY_ATTEMPT",
            "campaign_id": campaign_id,
            "budget_action_key": action_key,
            "recipe_id": recipe_id,
            "case_id": "quick_decode_gate_up",
            "requires_gpu": True,
            "resource_id": "gpu1",
            "fencing_epoch": lease.fencing_epoch,
            "profiler_image": profiler_image,
            "profiler_build_profile_digest": build_digest,
            "profiler_activation_profile_digest": activation_digest,
            "container_name": "kar-profile-canary-test",
            "argv": ["docker", "run", "--rm", profiler_image],
            "argv_digest": canonical_sha256(
                ["docker", "run", "--rm", profiler_image]
            ),
            "timeout_seconds": 900.0,
            "expected_worker_echo": {"recipe_id": recipe_id},
        }
        intent = {
            **intent_material,
            "attempt_id": (
                "profile-canary-attempt-"
                + canonical_sha256(intent_material).split(":", 1)[1]
            ),
        }
        self.campaign.record_profile_canary_attempt_intent(
            campaign_id,
            intent=intent,
        )
        private_diagnostic = {
            "schema_version": 1,
            "kind": "PROFILE_IMAGE_CANARY_PRIVATE_DIAGNOSTIC",
            "attempt_id": intent["attempt_id"],
            "campaign_id": campaign_id,
            "budget_action_key": action_key,
            "recipe_id": recipe_id,
            "classification": "UNKNOWN",
            "reason_code": "NONZERO_EXIT",
            "cause": {
                "type": "ValueError",
                "message": "bounded profiler returned non-zero",
                "message_truncated": False,
            },
            "command": {
                "available": True,
                "argv": intent["argv"],
                "returncode": 17,
                "timed_out": False,
                "output_limited": False,
                "stdout": {"encoding": "utf-8", "content": "", "byte_size": 0},
                "stderr": {"encoding": "utf-8", "content": "failure", "byte_size": 7},
            },
            "outcome": None,
            "worker_files": {},
            "file_inventory": {
                "entries": [],
                "truncated": False,
                "scan_errors": [],
            },
        }
        diagnostic_bytes = canonical_json_text(private_diagnostic).encode("utf-8")
        diagnostic_object_id = "sha256:" + hashlib.sha256(
            diagnostic_bytes
        ).hexdigest()
        diagnostic_digest = diagnostic_object_id.split(":", 1)[1]
        diagnostic_path = (
            self.controller_dir
            / "objects"
            / "sha256"
            / diagnostic_digest[:2]
            / diagnostic_digest[2:]
        )
        diagnostic_path.parent.mkdir(parents=True)
        diagnostic_path.write_bytes(diagnostic_bytes)
        self.campaign.record_profile_canary_attempt_diagnostic(
            campaign_id,
            diagnostic={
                "schema_version": 1,
                "kind": "PROFILE_IMAGE_CANARY_DIAGNOSTIC",
                "attempt_id": intent["attempt_id"],
                "campaign_id": campaign_id,
                "budget_action_key": action_key,
                "recipe_id": recipe_id,
                "diagnostic_object_id": diagnostic_object_id,
                "diagnostic_object_bytes": len(diagnostic_bytes),
                "classification": "UNKNOWN",
                "reason_code": "NONZERO_EXIT",
                "returncode": 17,
                "timed_out": False,
                "output_limited": False,
                "outcome_status": None,
                "inventory_entries": 0,
            },
        )
        self.campaign.release_resource(
            lease,
            quarantine=True,
            reason="unknown canary outcome",
        )
        self.campaign.pause_campaign(
            campaign_id,
            status="PAUSED_UNKNOWN_OUTCOME",
            reason="unknown canary outcome",
        )
        doctor = self.campaign.record_doctor_evidence(
            campaign_id,
            evidence={
                "schema_version": 1,
                "kind": "CAMPAIGN_RESUME_DOCTOR",
                "status": "SUCCESS",
                "campaign_id": campaign_id,
                "resource_id": "gpu1",
                "observed_epoch": time.time() + 1,
                "namespace_id": NAMESPACE_A,
                "config_digest": "sha256:" + "f" * 64,
                "execution_environment": ENVIRONMENT.to_dict(),
                "doctor_result": {
                    "status": "SUCCESS",
                    "c500_probe": {
                        "environment": {"compile_probe_status": "PASSED"}
                    },
                },
            },
        )
        self.campaign.abandon_profile_canary(
            campaign_id,
            doctor_evidence_digest=doctor["digest"],
            budget_action_key=action_key,
        )

        trusted = self.collector.collect(
            interval_start_epoch=0,
            interval_end_epoch=time.time() + 5,
        )
        self.assertEqual(trusted.status, "AVAILABLE", trusted.to_dict())
        self.assertEqual(trusted.counts["budget_leak_count"], 0)
        self.assertEqual(
            trusted.sources["campaign"]["summary"][
                "profile_canary_abandonment_count"
            ],
            1,
        )

        diagnostic_path.write_text("corrupted", encoding="utf-8")
        corrupt_diagnostic = self.collector.collect(
            interval_start_epoch=0,
            interval_end_epoch=time.time() + 5,
        )
        self.assertEqual(corrupt_diagnostic.status, "UNAVAILABLE")
        self.assertIn(
            "CAMPAIGN_DB_UNAVAILABLE",
            corrupt_diagnostic.unavailable_reasons,
        )
        diagnostic_path.write_bytes(diagnostic_bytes)

        self.campaign.connection.execute(
            """
            UPDATE resource_leases SET status = 'QUARANTINED'
            WHERE campaign_id = ? AND fencing_epoch = ?
            """,
            (campaign_id, lease.fencing_epoch),
        )
        self.campaign.connection.commit()
        tampered = self.collector.collect(
            interval_start_epoch=0,
            interval_end_epoch=time.time() + 5,
        )
        self.assertEqual(tampered.status, "UNAVAILABLE")
        self.assertIn(
            "CAMPAIGN_DB_UNAVAILABLE",
            tampered.unavailable_reasons,
        )

    def test_private_canary_diagnostic_reader_is_bounded_and_canonical(
        self,
    ) -> None:
        objects = self.controller_dir / "objects" / "sha256"

        def store(payload: bytes) -> str:
            digest = hashlib.sha256(payload).hexdigest()
            path = objects / digest[:2] / digest[2:]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            return "sha256:" + digest

        valid_payload = canonical_json_text({"status": "private"}).encode()
        valid_id = store(valid_payload)
        value, byte_size = soak_collector._read_private_canary_diagnostic(
            self.config,
            valid_id,
        )
        self.assertEqual(value, {"status": "private"})
        self.assertEqual(byte_size, len(valid_payload))

        for label, payload, message in (
            ("empty", b"", "bounded"),
            ("invalid utf8", b"\xff", "strict JSON"),
            ("scalar", b"[]", "JSON object"),
            ("noncanonical", b'{"status": "private"}', "canonical"),
        ):
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError, message
            ):
                soak_collector._read_private_canary_diagnostic(
                    self.config,
                    store(payload),
                )

        oversized = b"x" * (
            soak_collector.PROFILE_CANARY_DIAGNOSTIC_MAX_BYTES + 1
        )
        with self.assertRaisesRegex(ValueError, "bounded"):
            soak_collector._read_private_canary_diagnostic(
                self.config,
                store(oversized),
            )

    def test_hard_retry_counts_only_specific_iteration_in_same_namespace(self) -> None:
        baseline_revision = self._seed_campaign()
        self.campaign.connection.execute(
            """
            INSERT INTO child_runs(
                campaign_id, child_index, controller_run_id, status,
                baseline_revision_id, proposer_profile_json, max_candidates,
                max_wall_seconds, max_consecutive_failures,
                stop_after_promotion, created_at, result_json
            ) VALUES (
                'campaign', 1, 'bad-run', 'HARD_FAILED', ?, '{}', 5,
                21600, 3, 1, ?, '{}'
            )
            """,
            (baseline_revision, _utc(time.time())),
        )
        self.campaign.connection.commit()
        self._controller_run("bad-run", NAMESPACE_A)
        # This ordinary candidate belongs to the bad child but did not cause
        # the hard/unknown GPU outcome and therefore must not be quarantined.
        self._iteration("bad-run", 1, CANDIDATE, "FULL_REJECTED")
        self._iteration("bad-run", 2, HARD_CANDIDATE, "HARD_FAILURE")
        self.controller.update_run("bad-run", status="HARD_FAILED")

        self._controller_run("other-namespace", NAMESPACE_B)
        self._iteration(
            "other-namespace", 1, HARD_CANDIDATE, "FULL_REJECTED"
        )
        self._controller_run("same-namespace-ordinary", NAMESPACE_A)
        self._iteration(
            "same-namespace-ordinary", 1, CANDIDATE, "FULL_REJECTED"
        )
        before_retry = self.collector.collect(
            interval_start_epoch=0, interval_end_epoch=time.time() + 5
        )
        self.assertEqual(before_retry.counts["hard_fault_retry_count"], 0)

        self._controller_run("same-namespace-retry", NAMESPACE_A)
        self._iteration(
            "same-namespace-retry", 1, HARD_CANDIDATE, "FULL_REJECTED"
        )
        after_retry = self.collector.collect(
            interval_start_epoch=0, interval_end_epoch=time.time() + 5
        )
        self.assertEqual(after_retry.counts["hard_fault_retry_count"], 1)

    def test_missing_db_fields_and_docker_failure_are_unavailable(self) -> None:
        self.runner.fail = True
        docker_failure = self.collector.collect(
            interval_start_epoch=1000, interval_end_epoch=1100
        )
        self.assertEqual(docker_failure.status, "UNAVAILABLE")
        self.assertIn(
            "DOCKER_INVENTORY_UNAVAILABLE",
            docker_failure.unavailable_reasons,
        )

        self.runner.fail = False
        self.controller.close()
        self.controller = None  # type: ignore[assignment]
        connection = sqlite3.connect(
            self.controller_dir / "controller.sqlite3"
        )
        connection.execute("DROP TABLE events")
        connection.close()
        db_failure = self.collector.collect(
            interval_start_epoch=1000, interval_end_epoch=1100
        )
        self.assertEqual(db_failure.status, "UNAVAILABLE")
        self.assertIn(
            "CONTROLLER_DB_UNAVAILABLE", db_failure.unavailable_reasons
        )

    def test_observation_schema_and_content_digests_reject_tampering(self) -> None:
        observation = self.collector.collect(
            interval_start_epoch=1000, interval_end_epoch=1100
        )
        counts = dict(observation.counts)
        missing_count = dict(counts)
        missing_count.pop(COUNT_FIELDS[0])
        negative_count = dict(counts)
        negative_count[COUNT_FIELDS[0]] = -1
        cases = (
            ("private seal", {"_trusted_seal": object()}, TypeError),
            ("schema version", {"schema_version": 2}, ValueError),
            ("collector revision", {"collector_revision": "future"}, ValueError),
            ("status", {"status": "UNKNOWN"}, ValueError),
            ("boolean epoch", {"interval_start_epoch": True}, ValueError),
            ("non-finite epoch", {"interval_end_epoch": math.inf}, ValueError),
            ("backwards interval", {"interval_end_epoch": 999}, ValueError),
            ("snapshot shape", {"invariant_snapshot": []}, ValueError),
            ("source shape", {"sources": []}, ValueError),
            ("strict JSON", {"invariant_snapshot": {"x": math.nan}}, ValueError),
            ("count schema", {"counts": missing_count}, ValueError),
            ("negative count", {"counts": negative_count}, ValueError),
            ("empty reason", {"status": "UNAVAILABLE", "unavailable_reasons": ("",)}, ValueError),
            (
                "duplicate reason",
                {
                    "status": "UNAVAILABLE",
                    "unavailable_reasons": ("SOURCE", "SOURCE"),
                },
                ValueError,
            ),
            ("availability mismatch", {"status": "UNAVAILABLE"}, ValueError),
            (
                "snapshot digest",
                {"invariant_snapshot_digest": "sha256:" + "0" * 64},
                ValueError,
            ),
            ("evidence digest", {"evidence_id": "sha256:" + "0" * 64}, ValueError),
        )
        for label, changes, error in cases:
            with self.subTest(label=label), self.assertRaises(error):
                replace(observation, **changes)

    def test_public_collection_rejects_invalid_intervals(self) -> None:
        for start, end in ((True, 1), (-1, 1), (0, math.nan), (2, 1)):
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                self.collector.collect(
                    interval_start_epoch=start, interval_end_epoch=end
                )

    def test_git_identity_probe_is_fixed_and_fails_closed(self) -> None:
        runner = InventoryRunner()
        runner.git_head = self.config.expected_git_commit
        collector = SoakObservationCollector(
            self.config,
            campaign_database=self.campaign_path,
            runner=runner,
        )
        clean = collector.collect(
            interval_start_epoch=1000, interval_end_epoch=1100
        )
        self.assertEqual(clean.status, "AVAILABLE")
        self.assertEqual(clean.invariant_snapshot["code"]["git_commit"], runner.git_head)
        self.assertEqual(
            clean.invariant_snapshot["profiler"],
            {
                "build_profile_digest": PROFILER_BUILD_PROFILE_DIGEST,
                "activation_profile_digest": (
                    PROFILER_ACTIVATION_PROFILE_DIGEST
                ),
                "build_profile": profiler_build_profile_snapshot(),
                "activation_profile": profiler_activation_profile_snapshot(),
            },
        )
        self.assertEqual(
            runner.calls[:2],
            [
                (
                    "git", "-c", "core.fsmonitor=false", "-C",
                    str(self.repository), "rev-parse", "HEAD",
                ),
                (
                    "git", "-c", "core.fsmonitor=false", "-C",
                    str(self.repository), "status", "--porcelain",
                    "--untracked-files=all",
                ),
            ],
        )

        runner.git_status = "?? untracked-candidate.py\n"
        dirty = collector.collect(
            interval_start_epoch=1100, interval_end_epoch=1200
        )
        self.assertEqual(dirty.status, "UNAVAILABLE")
        self.assertIn(
            "INVARIANT_PROBE_UNAVAILABLE", dirty.unavailable_reasons
        )
        self.assertEqual(dirty.sources["docker"]["status"], "AVAILABLE")

        runner.git_status = ""
        runner.git_head = "0" * 40
        wrong_head = collector.collect(
            interval_start_epoch=1200, interval_end_epoch=1300
        )
        self.assertIn(
            "INVARIANT_PROBE_UNAVAILABLE", wrong_head.unavailable_reasons
        )

    def test_invariant_provider_must_return_strict_json_object(self) -> None:
        providers = (
            lambda: [],
            lambda: {"invalid": math.nan},
            lambda: (_ for _ in ()).throw(OSError("probe failed")),
        )
        for provider in providers:
            with self.subTest(provider=provider):
                collector = SoakObservationCollector(
                    self.config,
                    campaign_database=self.campaign_path,
                    runner=self.runner,
                    invariant_provider=provider,
                )
                observation = collector.collect(
                    interval_start_epoch=1000, interval_end_epoch=1100
                )
                self.assertEqual(observation.status, "UNAVAILABLE")
                self.assertIn(
                    "INVARIANT_PROBE_UNAVAILABLE",
                    observation.unavailable_reasons,
                )

    def test_bounded_runner_and_docker_inventory_failures_are_unavailable(self) -> None:
        for field in ("timed_out", "output_limited"):
            setattr(self.runner, field, True)
            with self.subTest(field=field):
                observation = self.collector.collect(
                    interval_start_epoch=1000, interval_end_epoch=1100
                )
                self.assertIn(
                    "DOCKER_INVENTORY_UNAVAILABLE",
                    observation.unavailable_reasons,
                )
            setattr(self.runner, field, False)

        self.runner.returned_argv = ("tampered",)
        tampered = self.collector.collect(
            interval_start_epoch=1100, interval_end_epoch=1200
        )
        self.assertIn(
            "DOCKER_INVENTORY_UNAVAILABLE", tampered.unavailable_reasons
        )
        self.runner.returned_argv = None

        for stdout in (
            "foreign-container\n",
            "kar-has whitespace\n",
            "kar-duplicate\nkar-duplicate\n",
        ):
            self.runner.stdout_override = stdout
            with self.subTest(stdout=stdout):
                malformed = self.collector.collect(
                    interval_start_epoch=1200, interval_end_epoch=1300
                )
                self.assertIn(
                    "DOCKER_INVENTORY_UNAVAILABLE",
                    malformed.unavailable_reasons,
                )
        self.runner.stdout_override = "\nkar-orphan\n\n"
        blank_lines = self.collector.collect(
            interval_start_epoch=1300, interval_end_epoch=1400
        )
        self.assertEqual(blank_lines.status, "AVAILABLE")
        self.assertEqual(blank_lines.counts["orphan_container_count"], 1)

    def test_missing_expected_active_container_fails_closed(self) -> None:
        self._controller_run("active-run", NAMESPACE_A)
        iteration = self.controller.create_iteration(
            "active-run", 1, parent_hash="1" * 64
        )
        self.controller.update_iteration(
            int(iteration["id"]), active_container="kar-expected"
        )
        observation = self.collector.collect(
            interval_start_epoch=0, interval_end_epoch=time.time() + 5
        )
        self.assertIn(
            "DOCKER_INVENTORY_UNAVAILABLE", observation.unavailable_reasons
        )

    def test_schema_versions_and_corrupt_json_are_source_unavailable(self) -> None:
        self.campaign.connection.execute("PRAGMA user_version = 9")
        wrong_campaign_version = self.collector.collect(
            interval_start_epoch=1000, interval_end_epoch=1100
        )
        self.assertIn(
            "CAMPAIGN_DB_UNAVAILABLE",
            wrong_campaign_version.unavailable_reasons,
        )
        self.campaign.connection.execute("PRAGMA user_version = 1")

        self.controller.connection.execute("PRAGMA user_version = 9")
        wrong_controller_version = self.collector.collect(
            interval_start_epoch=1100, interval_end_epoch=1200
        )
        self.assertIn(
            "CONTROLLER_DB_UNAVAILABLE",
            wrong_controller_version.unavailable_reasons,
        )
        self.controller.connection.execute("PRAGMA user_version = 3")

        self._seed_campaign()
        self.campaign.connection.execute(
            "UPDATE baseline_revisions SET policy_snapshot_json = '{'"
        )
        self.campaign.connection.commit()
        corrupt_json = self.collector.collect(
            interval_start_epoch=1200, interval_end_epoch=1300
        )
        self.assertIn(
            "CAMPAIGN_DB_UNAVAILABLE", corrupt_json.unavailable_reasons
        )

    def test_invalid_fencing_cursor_and_controller_timestamp_fail_closed(self) -> None:
        self._seed_campaign()
        self.campaign.connection.execute("PRAGMA ignore_check_constraints = ON")
        self.campaign.connection.execute(
            """
            INSERT INTO resource_leases(
                resource_id, fencing_epoch, campaign_id, status,
                acquired_at, expires_epoch, released_at, reason
            ) VALUES ('gpu-corrupt', -1, 'campaign', 'RELEASED', ?, 2000, ?, 'test')
            """,
            (_utc(1000), _utc(1100)),
        )
        self.campaign.connection.commit()
        bad_cursor = self.collector.collect(
            interval_start_epoch=1000, interval_end_epoch=1100
        )
        self.assertIn(
            "CAMPAIGN_DB_UNAVAILABLE", bad_cursor.unavailable_reasons
        )

        self._controller_run("bad-time", NAMESPACE_A)
        self.controller.connection.execute(
            "UPDATE runs SET updated_at = 'timezone-missing' WHERE id = 'bad-time'"
        )
        self.controller.connection.commit()
        bad_timestamp = self.collector.collect(
            interval_start_epoch=1100, interval_end_epoch=1200
        )
        self.assertIn(
            "CONTROLLER_DB_UNAVAILABLE", bad_timestamp.unavailable_reasons
        )

    def test_confirmed_attempt_and_staged_revision_activity_are_derived(self) -> None:
        seed_revision = self._seed_campaign()
        seed_row = self.campaign.connection.execute(
            "SELECT baseline_ref_json FROM baseline_revisions WHERE id = ?",
            (seed_revision,),
        ).fetchone()
        assert seed_row is not None
        self.campaign.connection.execute(
            """
            INSERT INTO baseline_revisions(
                id, campaign_id, revision_index, namespace_id,
                parent_revision_id, artifact_id, baseline_ref_json,
                revision_kind, primary_experiment_uid,
                confirmation_experiment_uid, policy_snapshot_json,
                idempotency_key, created_at
            ) VALUES (
                'staged-1', 'campaign', 1, ?, ?, ?, ?, 'STAGED',
                'primary-uid', 'confirmation-uid', '{}', 'stage-once', ?
            )
            """,
            (NAMESPACE_A, seed_revision, SOURCE, seed_row[0], _utc(time.time())),
        )
        self.campaign.connection.commit()

        self._controller_run("confirmed-run", NAMESPACE_A)
        iteration = self.controller.create_iteration(
            "confirmed-run", 1, parent_hash="1" * 64
        )
        self.controller.create_evaluation_attempt(
            experiment_uid="confirmation-uid",
            run_id="confirmed-run",
            iteration_id=int(iteration["id"]),
            stage="CONFIRMATION",
            suite="full",
        )
        self.controller.start_evaluation_attempt("confirmation-uid")
        self.controller.finish_evaluation_attempt(
            "confirmation-uid", status="SUCCEEDED", result={"correct": True}
        )
        self.controller.link_history_experiment("confirmation-uid", 7)

        observation = self.collector.collect(
            interval_start_epoch=0, interval_end_epoch=time.time() + 5
        )
        staged = observation.sources["campaign"]["summary"]["activity"][
            "staged_revisions"
        ]
        confirmed = observation.sources["controller"]["summary"]["activity"][
            "confirmed_experiment_uids"
        ]
        self.assertEqual(staged[0]["revision_id"], "staged-1")
        self.assertIn("confirmation-uid", confirmed)

    def test_root_direct_or_missing_campaign_database_is_rejected(self) -> None:
        root_direct = self.root / "campaign.sqlite3"
        sqlite3.connect(root_direct).close()
        with self.assertRaisesRegex(ValueError, "conventional absolute path"):
            SoakObservationCollector(
                self.config,
                campaign_database=root_direct,
                runner=self.runner,
                invariant_provider=lambda: {},
            )

        missing_root = self.root / "missing-runtime"
        missing_config = replace(
            self.config,
            state_dir=missing_root / "state",
            controller_dir=missing_root / "controller",
            checkpoint_dir=missing_root / "checkpoints",
        )
        missing = missing_root / "campaign" / "campaign.sqlite3"
        with self.assertRaisesRegex(ValueError, "regular non-symlink"):
            SoakObservationCollector(
                missing_config,
                campaign_database=missing,
                runner=self.runner,
                invariant_provider=lambda: {},
            )

    def test_runtime_root_and_read_only_database_failures_are_closed(self) -> None:
        with patch.object(Path, "resolve", side_effect=OSError("unavailable")):
            with self.assertRaisesRegex(ValueError, "runtime root"):
                SoakObservationCollector(
                    self.config,
                    campaign_database=self.campaign_path,
                    runner=self.runner,
                    invariant_provider=lambda: {},
                )

        self.controller.close()
        self.controller = None  # type: ignore[assignment]
        (self.controller_dir / "controller.sqlite3").unlink()
        missing_controller = self.collector.collect(
            interval_start_epoch=1000, interval_end_epoch=1100
        )
        self.assertIn(
            "CONTROLLER_DB_UNAVAILABLE",
            missing_controller.unavailable_reasons,
        )

    def test_foreign_key_and_non_string_json_corruption_are_unavailable(self) -> None:
        self.campaign.close()
        self.campaign = None  # type: ignore[assignment]
        connection = sqlite3.connect(self.campaign_path)
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            """
            INSERT INTO baseline_revisions(
                id, campaign_id, revision_index, namespace_id,
                parent_revision_id, artifact_id, baseline_ref_json,
                revision_kind, primary_experiment_uid,
                confirmation_experiment_uid, policy_snapshot_json,
                idempotency_key, created_at
            ) VALUES (
                'grafted', 'missing-campaign', 0, ?, NULL, ?, '{}',
                'DEPLOYMENT_SEED', NULL, NULL, '{}', NULL, ?
            )
            """,
            (NAMESPACE_A, SOURCE, _utc(1000)),
        )
        connection.commit()
        connection.close()
        grafted = self.collector.collect(
            interval_start_epoch=1000, interval_end_epoch=1100
        )
        self.assertIn("CAMPAIGN_DB_UNAVAILABLE", grafted.unavailable_reasons)

        connection = sqlite3.connect(self.campaign_path)
        connection.execute("DELETE FROM baseline_revisions WHERE id = 'grafted'")
        connection.execute(
            """
            INSERT INTO campaigns(
                id, namespace_id, mode, status, created_at, updated_at,
                snapshot_digest, snapshot_json, max_candidates, max_wall_ms,
                max_gpu_ms, max_tokens, max_cost_microusd,
                active_baseline_revision_id, allow_staged_lineage,
                stop_reason, pause_evidence_digest
            ) VALUES (
                'json-corrupt', ?, 'DISCOVERY', 'CREATED', ?, ?, ?, ?,
                0, 0, 0, 0, 0, NULL, 0, NULL, NULL
            )
            """,
            (
                NAMESPACE_A,
                _utc(1000),
                _utc(1000),
                "sha256:" + "0" * 64,
                sqlite3.Binary(b"{}"),
            ),
        )
        connection.commit()
        connection.close()
        non_string_json = self.collector.collect(
            interval_start_epoch=1100, interval_end_epoch=1200
        )
        self.assertIn(
            "CAMPAIGN_DB_UNAVAILABLE", non_string_json.unavailable_reasons
        )

    def test_campaign_status_revision_and_lease_corruption_fail_closed(self) -> None:
        seed_revision = self._seed_campaign()
        self.campaign.connection.execute("PRAGMA ignore_check_constraints = ON")
        self.campaign.connection.execute(
            "UPDATE campaigns SET status = 'CORRUPT' WHERE id = 'campaign'"
        )
        self.campaign.connection.commit()
        bad_campaign = self.collector.collect(
            interval_start_epoch=1000, interval_end_epoch=1100
        )
        self.assertIn("CAMPAIGN_DB_UNAVAILABLE", bad_campaign.unavailable_reasons)
        self.campaign.connection.execute(
            "UPDATE campaigns SET status = 'CREATED' WHERE id = 'campaign'"
        )

        self.campaign.connection.execute(
            "UPDATE baseline_revisions SET revision_kind = 'CORRUPT' WHERE id = ?",
            (seed_revision,),
        )
        self.campaign.connection.commit()
        bad_revision = self.collector.collect(
            interval_start_epoch=1100, interval_end_epoch=1200
        )
        self.assertIn("CAMPAIGN_DB_UNAVAILABLE", bad_revision.unavailable_reasons)
        self.campaign.connection.execute(
            "UPDATE baseline_revisions SET revision_kind = 'DEPLOYMENT_SEED' WHERE id = ?",
            (seed_revision,),
        )

        lease_values = (
            ("", "RELEASED", _utc(1000), 1100, _utc(1050)),
            ("gpu-invalid-status", "CORRUPT", _utc(1000), 1100, _utc(1050)),
            ("gpu-backwards", "RELEASED", _utc(1200), 1100, None),
        )
        for index, values in enumerate(lease_values, start=1):
            self.campaign.connection.execute("DELETE FROM resource_leases")
            self.campaign.connection.execute(
                """
                INSERT INTO resource_leases(
                    resource_id, fencing_epoch, campaign_id, status,
                    acquired_at, expires_epoch, released_at, reason
                ) VALUES (?, ?, 'campaign', ?, ?, ?, ?, 'corrupt')
                """,
                (values[0], index, *values[1:]),
            )
            self.campaign.connection.commit()
            with self.subTest(resource=values[0]):
                observation = self.collector.collect(
                    interval_start_epoch=1000, interval_end_epoch=1300
                )
                self.assertIn(
                    "CAMPAIGN_DB_UNAVAILABLE",
                    observation.unavailable_reasons,
                )

    def test_controller_identity_rows_and_timestamps_fail_closed(self) -> None:
        self._controller_run("corrupt-run", NAMESPACE_A)
        iteration = self.controller.create_iteration(
            "corrupt-run", 1, parent_hash="1" * 64
        )
        self.controller.create_evaluation_attempt(
            experiment_uid="attempt-uid",
            run_id="corrupt-run",
            iteration_id=int(iteration["id"]),
            stage="QUICK",
            suite="quick",
        )

        mutations = (
            (
                "run",
                "UPDATE runs SET updated_at = 7 WHERE id = 'corrupt-run'",
                "UPDATE runs SET updated_at = created_at WHERE id = 'corrupt-run'",
            ),
            (
                "iteration",
                "UPDATE iterations SET candidate_hash = 'not-a-hash' WHERE id = %d"
                % int(iteration["id"]),
                "UPDATE iterations SET candidate_hash = NULL WHERE id = %d"
                % int(iteration["id"]),
            ),
            (
                "event",
                "UPDATE events SET event = '' WHERE run_id = 'corrupt-run'",
                "UPDATE events SET event = 'RESTORED' WHERE run_id = 'corrupt-run'",
            ),
            (
                "attempt",
                "UPDATE evaluation_attempts SET experiment_uid = '' WHERE id = 1",
                "UPDATE evaluation_attempts SET experiment_uid = 'attempt-uid' WHERE id = 1",
            ),
        )
        for label, corrupt, restore in mutations:
            self.controller.connection.execute(corrupt)
            self.controller.connection.commit()
            with self.subTest(label=label):
                observation = self.collector.collect(
                    interval_start_epoch=1000, interval_end_epoch=1100
                )
                self.assertIn(
                    "CONTROLLER_DB_UNAVAILABLE",
                    observation.unavailable_reasons,
                )
            self.controller.connection.execute(restore)
            self.controller.connection.commit()

    def test_wrong_config_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(TypeError, "ControllerConfig"):
            SoakObservationCollector(  # type: ignore[arg-type]
                {}, campaign_database=self.campaign_path
            )

    def test_nonconventional_campaign_database_is_rejected(self) -> None:
        elsewhere = self.root / "elsewhere.sqlite3"
        sqlite3.connect(elsewhere).close()
        with self.assertRaisesRegex(ValueError, "conventional absolute path"):
            SoakObservationCollector(
                self.config,
                campaign_database=elsewhere,
                runner=self.runner,
                invariant_provider=lambda: {},
            )


if __name__ == "__main__":
    unittest.main()
