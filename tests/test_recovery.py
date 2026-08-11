from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest

from kernel_research.autorun.controller import ResearchController
from kernel_research.autorun.store import ControllerStore
from kernel_research.campaign.store import CampaignStore
from kernel_research.campaign.models import BudgetAmount
from kernel_research.history import HistoryStore
from kernel_research.platform.artifacts import ArtifactId
from kernel_research.platform.identity import BaselineRef, ExperimentIdentity
from kernel_research.platform.profiles import LEGACY_RESEARCH_NAMESPACE
from kernel_research.recovery import restore_checkpoint, verify_checkpoint

from test_autorun import FakeEvaluator, SEED, SEED_HASH, _baseline, _config


CAMPAIGN_SUMMARY_TABLES = (
    "campaigns",
    "baseline_revisions",
    "child_runs",
    "budget_actions",
    "resource_leases",
    "oj_nominations",
    "outbox",
    "soak_generations",
    "soak_violations",
)

DATABASE_SUMMARY_TABLES = {
    "controller": (
        "runs",
        "iterations",
        "events",
        "proposal_attempts",
        "evaluation_attempts",
    ),
    "history": (
        "research_namespaces",
        "candidate_artifacts",
        "experiments",
        "case_measurements",
        "experiment_relations",
    ),
    "campaign": CAMPAIGN_SUMMARY_TABLES,
}


def _campaign_database_summary(path: Path) -> dict[str, object]:
    connection = sqlite3.connect(path)
    try:
        counts = {
            table: int(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
            for table in CAMPAIGN_SUMMARY_TABLES
        }
    finally:
        connection.close()
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "integrity_check": "ok",
        "sqlite_version": sqlite3.sqlite_version,
        "table_counts": counts,
    }


def _read_manifest(checkpoint: Path) -> dict[str, object]:
    return json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(checkpoint: Path, manifest: dict[str, object]) -> None:
    (checkpoint / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _refresh_file_record(
    checkpoint: Path,
    manifest: dict[str, object],
    relative: str,
) -> None:
    path = checkpoint / relative
    rows = manifest["files"]
    assert isinstance(rows, list)
    row = next(item for item in rows if item["path"] == relative)
    row["size_bytes"] = path.stat().st_size
    row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_database(
    checkpoint: Path,
    name: str,
    *,
    refresh_counts: bool = True,
) -> None:
    manifest = _read_manifest(checkpoint)
    database = checkpoint / f"{name}.sqlite3"
    summaries = manifest["databases"]
    assert isinstance(summaries, dict)
    summary = summaries[name]
    assert isinstance(summary, dict)
    summary["sha256"] = hashlib.sha256(database.read_bytes()).hexdigest()
    if refresh_counts:
        connection = sqlite3.connect(database)
        try:
            summary["table_counts"] = {
                table: int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
                for table in DATABASE_SUMMARY_TABLES[name]
            }
        finally:
            connection.close()
    _refresh_file_record(checkpoint, manifest, f"{name}.sqlite3")
    _write_manifest(checkpoint, manifest)


class CheckpointRecoveryTests(unittest.TestCase):
    def _checkpoint(self, root: Path) -> Path:
        config = _config(root)
        _baseline(config.state_dir)
        controller = ResearchController(config, evaluator=FakeEvaluator())
        with ControllerStore(controller.controller_db) as store:
            store.create_run(
                run_id="recovery-run",
                deadline_epoch=time.time() + 60,
                config=config.redacted_dict(),
                initial_best_hash=SEED_HASH,
            )
        run_dir = config.controller_dir / "runs" / "recovery-run"
        run_dir.mkdir(parents=True)
        (run_dir / "audit.txt").write_text("audit", encoding="utf-8")
        controller._store_private_object(b"private proposer audit")
        return Path(controller.checkpoint("recovery-run")["path"])

    def _campaign_checkpoint(
        self, root: Path
    ) -> tuple[Path, dict[str, object]]:
        config = _config(root)
        _baseline(config.state_dir)
        baseline = BaselineRef.create(
            namespace=LEGACY_RESEARCH_NAMESPACE,
            artifact_id="source-sha256-v1:" + SEED_HASH,
            source="campaign",
            revision="recovery-campaign-seed",
        )
        proposer_profile = {"kind": "fixture", "revision": "v1"}
        campaign_database = root / "campaign.sqlite3"
        with CampaignStore(campaign_database) as campaign_store:
            campaign = campaign_store.create_campaign(
                campaign_id="recovery-campaign",
                namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                mode="DISCOVERY",
                snapshot={"profile": "fixture"},
                budget_limit=BudgetAmount(),
                initial_baseline_ref=baseline,
                initial_policy_snapshot={"policy": "fixture"},
            )
            campaign_store.start_campaign("recovery-campaign")
            child = campaign_store.create_child_run(
                "recovery-campaign",
                proposer_profile=proposer_profile,
                controller_run_id="recovery-campaign-run",
                max_candidates=config.max_candidates,
                max_wall_seconds=round(config.max_hours * 3600),
                max_consecutive_failures=config.max_consecutive_failures,
            )
            campaign_store.start_child_run(
                child["id"], controller_run_id="recovery-campaign-run"
            )

        workflow_snapshot = {
            "schema_version": 2,
            "mode": "DISCOVERY",
            "campaign_id": "recovery-campaign",
            "campaign_snapshot_digest": campaign["snapshot_digest"],
            "campaign_child": {
                "child_id": child["id"],
                "child_index": child["child_index"],
                "controller_run_id": "recovery-campaign-run",
            },
            "proposer_profile": proposer_profile,
        }
        controller = ResearchController(config, evaluator=FakeEvaluator())
        with ControllerStore(controller.controller_db) as store:
            store.create_run(
                run_id="recovery-campaign-run",
                deadline_epoch=time.time() + 60,
                config=config.redacted_dict(),
                initial_best_hash=SEED_HASH,
                namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                resolved_config_digest="sha256:" + "c" * 64,
                workflow_snapshot=workflow_snapshot,
                baseline_ref=baseline.to_dict(),
                history_cutoff=1,
            )
        checkpoint = Path(
            controller.checkpoint("recovery-campaign-run")["path"]
        )
        return checkpoint, {
            "baseline": baseline,
            "campaign": campaign,
            "child": child,
        }

    def _staged_campaign_checkpoint(self, root: Path) -> Path:
        config = _config(root)
        _baseline(config.state_dir)
        controller = ResearchController(config, evaluator=FakeEvaluator())
        environment = controller._resolved_execution_environment(
            LEGACY_RESEARCH_NAMESPACE
        )
        parent_artifact = ArtifactId.source_sha256(SEED_HASH)
        parent = BaselineRef.create(
            namespace=LEGACY_RESEARCH_NAMESPACE,
            artifact_id=parent_artifact,
            source="campaign",
            revision="recovery-resolved-parent",
            execution_environment=environment,
        )
        history_parent = BaselineRef.create(
            namespace=LEGACY_RESEARCH_NAMESPACE,
            artifact_id=parent_artifact,
            source="deployment",
            revision="recovery-resolved-history-parent",
            execution_environment=environment,
        )
        baseline_identity = ExperimentIdentity.create(
            experiment_uid="00000000-0000-4000-8000-000000000101",
            namespace=LEGACY_RESEARCH_NAMESPACE,
            mode="DISCOVERY",
            candidate_artifact_id=parent_artifact,
            parent_artifact_id=parent_artifact,
            baseline=history_parent,
            execution_environment=environment,
            stage="baseline",
            suite="full",
            replicate_kind="primary",
            run_id="recovery-resolved-history-parent",
            iteration=0,
        )
        candidate = SEED + "\n# staged recovery candidate\n"
        candidate_hash = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        candidate_artifact = ArtifactId.source_sha256(candidate_hash)
        primary_identity = ExperimentIdentity.create(
            experiment_uid="00000000-0000-4000-8000-000000000102",
            namespace=LEGACY_RESEARCH_NAMESPACE,
            mode="DISCOVERY",
            candidate_artifact_id=candidate_artifact,
            parent_artifact_id=parent_artifact,
            baseline=parent,
            execution_environment=environment,
            stage="full_primary",
            suite="full",
            replicate_kind="primary",
            campaign_id="recovery-staged-campaign",
            run_id="recovery-promotion-run",
            iteration=1,
        )
        confirmation_identity = ExperimentIdentity.create(
            experiment_uid="00000000-0000-4000-8000-000000000103",
            namespace=LEGACY_RESEARCH_NAMESPACE,
            mode="DISCOVERY",
            candidate_artifact_id=candidate_artifact,
            parent_artifact_id=parent_artifact,
            baseline=parent,
            execution_environment=environment,
            stage="confirmation",
            suite="full",
            replicate_kind="confirmation",
            campaign_id="recovery-staged-campaign",
            run_id="recovery-promotion-run",
            iteration=1,
        )
        measurement = {
            "name": "recovery-full-case",
            "matched_ratio": 1.0,
            "passed": True,
            "raw_samples": [90.0],
            "baseline_samples": [100.0],
        }
        with HistoryStore(
            config.state_dir / "history.sqlite3", state_dir=config.state_dir
        ) as history:
            history.ensure_namespace(
                LEGACY_RESEARCH_NAMESPACE.namespace_id,
                LEGACY_RESEARCH_NAMESPACE.to_dict(),
            )
            baseline = history.record_experiment(
                candidate_source=SEED,
                status="SUCCESS",
                backend="c500",
                suite="full",
                promotable=True,
                aggregate_score=1.0,
                identity=baseline_identity,
                result={
                    "status": "SUCCESS",
                    "promotion": {"phase": "baseline", "confirmed": True},
                },
                case_measurements=(measurement,),
            )
            primary_result = {
                "status": "SUCCESS",
                "promotion": {
                    "phase": "primary",
                    "reason": "confirmation_required",
                    "baseline_experiment_id": baseline.id,
                    "baseline_candidate_hash": SEED_HASH,
                    "confirmed": False,
                    "decision": {
                        "promoted": False,
                        "needs_confirmation": True,
                        "reason": "confirmation_required",
                    },
                },
            }
            primary = history.record_experiment(
                candidate_source=candidate,
                status="SUCCESS",
                backend="c500",
                suite="full",
                promotable=False,
                aggregate_score=1.1,
                identity=primary_identity,
                baseline_experiment_uid=baseline.experiment_uid,
                result=primary_result,
                case_measurements=(measurement,),
            )
            confirmation_result = {
                "status": "SUCCESS",
                "promotion": {
                    "phase": "confirmation",
                    "reason": "promoted",
                    "baseline_experiment_id": baseline.id,
                    "baseline_candidate_hash": SEED_HASH,
                    "primary_experiment_id": primary.id,
                    "confirmed": True,
                    "decision": {
                        "promoted": True,
                        "needs_confirmation": False,
                        "reason": "promoted",
                    },
                },
            }
            confirmation = history.record_experiment(
                candidate_source=candidate,
                status="SUCCESS",
                backend="c500",
                suite="full",
                promotable=True,
                aggregate_score=1.1,
                identity=confirmation_identity,
                baseline_experiment_uid=baseline.experiment_uid,
                result=confirmation_result,
                case_measurements=(measurement,),
            )
            history.add_experiment_relation(
                source_experiment_uid=primary.experiment_uid,
                target_experiment_uid=confirmation.experiment_uid,
                relation_type="confirmation_of",
                metadata={"baseline_experiment_id": baseline.id},
            )

        with ControllerStore(controller.controller_db) as store:
            store.create_run(
                run_id="recovery-promotion-run",
                deadline_epoch=time.time() + 60,
                config=config.redacted_dict(),
                initial_best_hash=SEED_HASH,
                namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                resolved_config_digest="sha256:" + "a" * 64,
                workflow_snapshot={"schema_version": 2},
                baseline_ref=parent.to_dict(),
                history_cutoff=baseline.id,
            )
            iteration = store.create_iteration(
                "recovery-promotion-run", 1, SEED_HASH
            )
            for record, identity, result in (
                (primary, primary_identity, primary_result),
                (confirmation, confirmation_identity, confirmation_result),
            ):
                store.create_evaluation_attempt(
                    experiment_uid=record.experiment_uid,
                    run_id="recovery-promotion-run",
                    iteration_id=int(iteration["id"]),
                    stage=identity.stage,
                    suite="full",
                    replicate_kind=identity.replicate_kind,
                    replicate_index=identity.replicate_index,
                    candidate_artifact_id=str(candidate_artifact),
                    parent_artifact_id=str(parent_artifact),
                    baseline_ref=parent.to_dict(),
                    condition_digest=identity.condition_digest,
                    request=identity.to_dict(),
                )
                store.start_evaluation_attempt(record.experiment_uid)
                store.finish_evaluation_attempt(
                    record.experiment_uid,
                    status="SUCCEEDED",
                    result=result,
                )
                store.link_history_experiment(
                    record.experiment_uid, record.id
                )
            store.update_run("recovery-promotion-run", status="PROMOTED")

        proposer_profile = {"kind": "fixture", "revision": "v1"}
        campaign_database = root / "campaign.sqlite3"
        with CampaignStore(campaign_database) as campaign_store:
            campaign = campaign_store.create_campaign(
                campaign_id="recovery-staged-campaign",
                namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                mode="DISCOVERY",
                snapshot={"profile": "fixture"},
                budget_limit=BudgetAmount(),
                initial_baseline_ref=parent,
                initial_policy_snapshot={"policy": "fixture"},
                allow_staged_lineage=True,
            )
            campaign_store.start_campaign("recovery-staged-campaign")
            staged = campaign_store.advance_baseline(
                "recovery-staged-campaign",
                expected_parent_revision_id=campaign[
                    "active_baseline_revision_id"
                ],
                parent_baseline_ref=parent,
                artifact_id=str(candidate_artifact),
                primary_experiment_uid=primary.experiment_uid,
                confirmation_experiment_uid=confirmation.experiment_uid,
                evidence_namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                evidence_execution_environment=environment,
                policy_snapshot={"policy": "fixture"},
                idempotency_key="recovery-staged-advance",
            )
            child = campaign_store.create_child_run(
                "recovery-staged-campaign",
                proposer_profile=proposer_profile,
                controller_run_id="recovery-staged-child",
                max_candidates=config.max_candidates,
                max_wall_seconds=round(config.max_hours * 3600),
                max_consecutive_failures=config.max_consecutive_failures,
            )
            campaign_store.start_child_run(
                child["id"], controller_run_id="recovery-staged-child"
            )
        staged_ref = BaselineRef.from_value(staged["baseline_ref"])
        workflow_snapshot = {
            "schema_version": 2,
            "mode": "DISCOVERY",
            "campaign_id": "recovery-staged-campaign",
            "campaign_snapshot_digest": campaign["snapshot_digest"],
            "campaign_child": {
                "child_id": child["id"],
                "child_index": child["child_index"],
                "controller_run_id": "recovery-staged-child",
            },
            "proposer_profile": proposer_profile,
        }
        with ControllerStore(controller.controller_db) as store:
            store.create_run(
                run_id="recovery-staged-child",
                deadline_epoch=time.time() + 60,
                config=config.redacted_dict(),
                initial_best_hash=candidate_hash,
                namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                resolved_config_digest="sha256:" + "b" * 64,
                workflow_snapshot=workflow_snapshot,
                baseline_ref=staged_ref.to_dict(),
                history_cutoff=confirmation.id,
            )
        return Path(controller.checkpoint("recovery-staged-child")["path"])

    def _linked_evaluation_checkpoint(self, root: Path) -> Path:
        config = _config(root)
        _baseline(config.state_dir)
        history = sqlite3.connect(config.state_dir / "history.sqlite3")
        try:
            evidence = history.execute(
                "SELECT id, experiment_uid, artifact_id FROM experiments"
            ).fetchone()
        finally:
            history.close()
        assert evidence is not None
        controller = ResearchController(config, evaluator=FakeEvaluator())
        with ControllerStore(controller.controller_db) as store:
            store.create_run(
                run_id="recovery-run",
                deadline_epoch=time.time() + 60,
                config=config.redacted_dict(),
                initial_best_hash=SEED_HASH,
            )
            iteration = store.create_iteration("recovery-run", 1, SEED_HASH)
            store.create_evaluation_attempt(
                experiment_uid=str(evidence[1]),
                run_id="recovery-run",
                iteration_id=int(iteration["id"]),
                stage="FULL_PRIMARY",
                suite="full",
                candidate_artifact_id=str(evidence[2]),
            )
            store.start_evaluation_attempt(str(evidence[1]))
            store.finish_evaluation_attempt(
                str(evidence[1]),
                status="SUCCEEDED",
                result={"status": "SUCCESS"},
            )
            store.link_history_experiment(str(evidence[1]), int(evidence[0]))
        return Path(controller.checkpoint("recovery-run")["path"])

    def test_restore_verifies_and_publishes_only_a_new_inactive_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint = self._checkpoint(root / "source")
            verified = verify_checkpoint(checkpoint)
            self.assertEqual(verified["run_id"], "recovery-run")
            self.assertEqual(verified["databases"]["controller"]["integrity_check"], "ok")

            destination = root / "restored-runtime"
            report = restore_checkpoint(checkpoint, destination)

            self.assertEqual(report["status"], "READY_FOR_MANUAL_SWITCH")
            self.assertFalse(report["switched_active_runtime"])
            self.assertTrue((destination / "state" / "history.sqlite3").is_file())
            self.assertTrue(
                (destination / "controller" / "controller.sqlite3").is_file()
            )
            self.assertTrue(
                (
                    destination
                    / "controller"
                    / "runs"
                    / "recovery-run"
                    / "audit.txt"
                ).is_file()
            )
            self.assertTrue(
                any(
                    path.is_file()
                    for path in (destination / "controller" / "objects").rglob("*")
                )
            )
            with self.assertRaisesRegex(ValueError, "new runtime root"):
                restore_checkpoint(checkpoint, destination)

    def test_tampering_or_unlisted_files_fail_before_destination_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint = self._checkpoint(root / "source")
            manifest = json.loads(
                (checkpoint / "manifest.json").read_text(encoding="utf-8")
            )
            target = checkpoint / manifest["files"][0]["path"]
            target.write_bytes(target.read_bytes() + b"tamper")
            destination = root / "must-not-exist"
            with self.assertRaisesRegex(ValueError, "verification failed"):
                restore_checkpoint(checkpoint, destination)
            self.assertFalse(destination.exists())

    def test_checkpoint_root_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint = self._checkpoint(root / "source")
            alias = root / "checkpoint-alias"
            os.symlink(checkpoint, alias, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "real directory"):
                verify_checkpoint(alias)

    def test_database_summary_contract_rejects_malformed_or_stale_evidence(self) -> None:
        cases = (
            ("missing-databases", "database summary is missing"),
            ("missing-history", "does not exactly match included databases"),
            ("unexpected-campaign", "does not exactly match included databases"),
            ("invalid-fields", "invalid fields"),
            ("invalid-sha", "invalid SHA-256"),
            ("stale-sha", "SHA-256 does not match"),
            ("bad-integrity", "does not record integrity ok"),
            ("missing-sqlite-version", "has no SQLite version"),
            ("incomplete-counts", "table summary is incomplete"),
            ("boolean-count", "table counts are invalid"),
            ("stale-count", "table counts do not match"),
        )
        for scenario, message in cases:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                checkpoint = self._checkpoint(root / "source")
                manifest = _read_manifest(checkpoint)
                databases = manifest["databases"]
                assert isinstance(databases, dict)
                history = databases["history"]
                assert isinstance(history, dict)

                if scenario == "missing-databases":
                    manifest["databases"] = None
                elif scenario == "missing-history":
                    databases.pop("history")
                elif scenario == "unexpected-campaign":
                    databases["campaign"] = {}
                elif scenario == "invalid-fields":
                    history.pop("sqlite_version")
                elif scenario == "invalid-sha":
                    history["sha256"] = "A" * 64
                elif scenario == "stale-sha":
                    history["sha256"] = "0" * 64
                elif scenario == "bad-integrity":
                    history["integrity_check"] = "corrupt"
                elif scenario == "missing-sqlite-version":
                    history["sqlite_version"] = ""
                elif scenario == "incomplete-counts":
                    counts = history["table_counts"]
                    assert isinstance(counts, dict)
                    counts.pop("experiments")
                elif scenario == "boolean-count":
                    counts = history["table_counts"]
                    assert isinstance(counts, dict)
                    counts["experiments"] = True
                elif scenario == "stale-count":
                    counts = history["table_counts"]
                    assert isinstance(counts, dict)
                    counts["experiments"] += 1
                _write_manifest(checkpoint, manifest)

                with self.assertRaisesRegex(ValueError, message):
                    verify_checkpoint(checkpoint)

    def test_database_summary_checks_live_schema_and_required_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint = self._checkpoint(root / "source")
            controller_database = checkpoint / "controller.sqlite3"
            connection = sqlite3.connect(controller_database)
            try:
                connection.execute("PRAGMA user_version = 999")
                connection.commit()
            finally:
                connection.close()
            _refresh_database(checkpoint, "controller")

            with self.assertRaisesRegex(ValueError, "schema version is unsupported"):
                verify_checkpoint(checkpoint)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint = self._checkpoint(root / "source")
            controller_database = checkpoint / "controller.sqlite3"
            connection = sqlite3.connect(controller_database)
            try:
                connection.execute("DROP TABLE events")
                connection.commit()
            finally:
                connection.close()
            _refresh_database(checkpoint, "controller", refresh_counts=False)

            with self.assertRaisesRegex(ValueError, "cannot be verified"):
                verify_checkpoint(checkpoint)

    def test_non_campaign_checkpoint_cross_checks_controller_and_history_identity(self) -> None:
        cases = (
            ("unknown-run", "does not contain the checkpoint run"),
            ("run-status", "run status differs"),
            ("manifest-config", "config differs from the frozen"),
            ("resolved-config", "resolved checkpoint config differs"),
            ("baseline-namespace", "baseline belongs to another namespace"),
            ("invalid-baseline", "invalid BaselineRef"),
            ("missing-artifact", "does not contain the frozen Run baseline artifact"),
            ("missing-evidence", "does not prove the frozen Run baseline experiment"),
            ("corrupt-artifact", "baseline artifact is absent or corrupted"),
        )
        for scenario, message in cases:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                checkpoint = self._checkpoint(root / "source")
                if scenario in {"unknown-run", "run-status", "manifest-config"}:
                    manifest = _read_manifest(checkpoint)
                    if scenario == "unknown-run":
                        manifest["run_id"] = "another-run"
                    elif scenario == "run-status":
                        manifest["run_status"] = "COMPLETED"
                    else:
                        config = manifest["config"]
                        assert isinstance(config, dict)
                        config["max_candidates"] += 1
                    _write_manifest(checkpoint, manifest)
                elif scenario == "resolved-config":
                    resolved_path = checkpoint / "resolved-config.json"
                    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
                    resolved["max_candidates"] += 1
                    resolved_path.write_text(
                        json.dumps(resolved, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    manifest = _read_manifest(checkpoint)
                    _refresh_file_record(
                        checkpoint, manifest, "resolved-config.json"
                    )
                    _write_manifest(checkpoint, manifest)
                elif scenario in {"baseline-namespace", "invalid-baseline"}:
                    connection = sqlite3.connect(checkpoint / "controller.sqlite3")
                    try:
                        connection.execute(
                            "DROP TRIGGER runs_v3_identity_immutable"
                        )
                        if scenario == "baseline-namespace":
                            connection.execute(
                                "UPDATE runs SET namespace_id = ? WHERE id = ?",
                                ("sha256:" + "d" * 64, "recovery-run"),
                            )
                        else:
                            connection.execute(
                                "UPDATE runs SET baseline_ref_json = ? WHERE id = ?",
                                ("[]", "recovery-run"),
                            )
                        connection.commit()
                    finally:
                        connection.close()
                    _refresh_database(checkpoint, "controller")
                elif scenario in {"missing-artifact", "missing-evidence"}:
                    connection = sqlite3.connect(checkpoint / "history.sqlite3")
                    try:
                        if scenario == "missing-artifact":
                            connection.execute("DELETE FROM candidate_artifacts")
                        else:
                            connection.execute(
                                "UPDATE experiments SET promotable = 0"
                            )
                        connection.commit()
                    finally:
                        connection.close()
                    _refresh_database(checkpoint, "history")
                else:
                    artifact_relative = (
                        f"state_objects/sha256/{SEED_HASH[:2]}/{SEED_HASH[2:]}"
                    )
                    artifact = checkpoint / artifact_relative
                    artifact.write_bytes(artifact.read_bytes() + b"# corrupt\n")
                    manifest = _read_manifest(checkpoint)
                    _refresh_file_record(checkpoint, manifest, artifact_relative)
                    _write_manifest(checkpoint, manifest)

                with self.assertRaisesRegex(ValueError, message):
                    verify_checkpoint(checkpoint)

    def test_controller_evaluation_links_are_reconciled_against_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._linked_evaluation_checkpoint(
                Path(temporary).resolve() / "source"
            )
            verify_checkpoint(checkpoint)

        cases = ("missing-history-row", "different-uid", "different-artifact")
        for scenario in cases:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                checkpoint = self._linked_evaluation_checkpoint(
                    Path(temporary).resolve() / "source"
                )
                database = checkpoint / "controller.sqlite3"
                connection = sqlite3.connect(database)
                try:
                    if scenario == "missing-history-row":
                        connection.execute(
                            "DROP TRIGGER evaluation_attempts_history_link_guard"
                        )
                        connection.execute(
                            "UPDATE evaluation_attempts SET history_experiment_id = 999"
                        )
                    elif scenario == "different-uid":
                        connection.execute(
                            "UPDATE evaluation_attempts SET experiment_uid = ?",
                            ("sha256:" + "1" * 64,),
                        )
                    else:
                        connection.execute(
                            "UPDATE evaluation_attempts SET candidate_artifact_id = ?",
                            ("source-sha256-v1:" + "2" * 64,),
                        )
                    connection.commit()
                finally:
                    connection.close()
                _refresh_database(checkpoint, "controller")

                with self.assertRaisesRegex(
                    ValueError, "evaluation link differs from trusted History"
                ):
                    verify_checkpoint(checkpoint)

    def test_campaign_checkpoint_requires_database_and_matching_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint, _ = self._campaign_checkpoint(root / "source")
            manifest = _read_manifest(checkpoint)
            databases = manifest["databases"]
            assert isinstance(databases, dict)
            databases.pop("campaign")
            _write_manifest(checkpoint, manifest)
            with self.assertRaisesRegex(
                ValueError, "does not exactly match included databases"
            ):
                verify_checkpoint(checkpoint)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint, _ = self._campaign_checkpoint(root / "source")
            (checkpoint / "campaign.sqlite3").unlink()
            manifest = _read_manifest(checkpoint)
            databases = manifest["databases"]
            rows = manifest["files"]
            assert isinstance(databases, dict)
            assert isinstance(rows, list)
            databases.pop("campaign")
            manifest["files"] = [
                row for row in rows if row["path"] != "campaign.sqlite3"
            ]
            _write_manifest(checkpoint, manifest)
            with self.assertRaisesRegex(ValueError, "Controller ownership"):
                verify_checkpoint(checkpoint)

    def test_campaign_checkpoint_cross_checks_ownership_child_and_baseline(self) -> None:
        cases = (
            ("manifest-campaign", "Controller ownership"),
            ("workflow-campaign", "Controller ownership"),
            ("campaign-namespace", "does not prove the frozen"),
            ("active-baseline", "does not prove the frozen"),
            ("baseline-artifact", "does not prove the frozen"),
            ("child-identity", "does not prove the frozen"),
            ("child-budget", "does not prove the frozen"),
            ("child-status", "does not prove the frozen"),
            ("invalid-child-json", "invalid child/baseline JSON"),
            ("staged-without-evidence", "does not prove the frozen"),
            ("staged-fabricated-evidence", "not both present in History"),
        )
        for scenario, message in cases:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                checkpoint, _ = self._campaign_checkpoint(root / "source")
                if scenario == "manifest-campaign":
                    manifest = _read_manifest(checkpoint)
                    manifest["campaign_id"] = "another-campaign"
                    _write_manifest(checkpoint, manifest)
                elif scenario == "workflow-campaign":
                    database = checkpoint / "controller.sqlite3"
                    connection = sqlite3.connect(database)
                    try:
                        connection.execute(
                            "DROP TRIGGER runs_v3_identity_immutable"
                        )
                        encoded = connection.execute(
                            "SELECT workflow_snapshot_json FROM runs WHERE id = ?",
                            ("recovery-campaign-run",),
                        ).fetchone()[0]
                        workflow = json.loads(encoded)
                        workflow["campaign_id"] = "another-campaign"
                        connection.execute(
                            "UPDATE runs SET workflow_snapshot_json = ? WHERE id = ?",
                            (
                                json.dumps(workflow, sort_keys=True),
                                "recovery-campaign-run",
                            ),
                        )
                        connection.commit()
                    finally:
                        connection.close()
                    _refresh_database(checkpoint, "controller")
                else:
                    database = checkpoint / "campaign.sqlite3"
                    connection = sqlite3.connect(database)
                    try:
                        if scenario == "campaign-namespace":
                            connection.execute(
                                "DROP TRIGGER campaign_identity_immutable"
                            )
                            connection.execute(
                                "UPDATE campaigns SET namespace_id = ?",
                                ("sha256:" + "e" * 64,),
                            )
                        elif scenario == "active-baseline":
                            connection.execute(
                                "UPDATE campaigns SET active_baseline_revision_id = ?",
                                ("wrong-revision",),
                            )
                        elif scenario == "baseline-artifact":
                            connection.execute(
                                "UPDATE baseline_revisions SET artifact_id = ?",
                                ("source-sha256-v1:" + "f" * 64,),
                            )
                        elif scenario == "child-identity":
                            connection.execute(
                                "UPDATE child_runs SET child_index = child_index + 1"
                            )
                        elif scenario == "child-budget":
                            connection.execute(
                                "UPDATE child_runs SET max_candidates = max_candidates - 1"
                            )
                        elif scenario == "child-status":
                            connection.execute(
                                "UPDATE child_runs SET status = 'BUDGET_EXHAUSTED'"
                            )
                        elif scenario == "invalid-child-json":
                            connection.execute(
                                "UPDATE child_runs SET proposer_profile_json = '[]'"
                            )
                        elif scenario == "staged-without-evidence":
                            connection.execute(
                                "UPDATE baseline_revisions SET revision_kind = 'STAGED'"
                            )
                        elif scenario == "staged-fabricated-evidence":
                            connection.execute(
                                """
                                UPDATE baseline_revisions
                                SET revision_kind = 'STAGED',
                                    parent_revision_id = id,
                                    primary_experiment_uid = 'fabricated-primary',
                                    confirmation_experiment_uid = 'fabricated-confirmation'
                                """
                            )
                        connection.commit()
                    finally:
                        connection.close()
                    _refresh_database(checkpoint, "campaign")

                with self.assertRaisesRegex(ValueError, message):
                    verify_checkpoint(checkpoint)

    def test_real_staged_campaign_checkpoint_proves_three_store_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._staged_campaign_checkpoint(
                Path(temporary).resolve() / "source"
            )

            verified = verify_checkpoint(checkpoint)

            self.assertEqual(verified["run_id"], "recovery-staged-child")
            self.assertIn("campaign", verified["databases"])

    def test_campaign_database_cannot_be_grafted_onto_an_unowned_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            checkpoint = self._checkpoint(root / "source")
            campaign_database = checkpoint / "campaign.sqlite3"
            with CampaignStore(campaign_database) as store:
                store.create_campaign(
                    campaign_id="recovery-campaign",
                    namespace_id="sha256:" + "a" * 64,
                    mode="DISCOVERY",
                    snapshot={"profile": "fixture"},
                    budget_limit=BudgetAmount(),
                    initial_artifact_id="source-sha256-v1:" + "b" * 64,
                    initial_policy_snapshot={"policy": "fixture"},
                )
            content = campaign_database.read_bytes()
            manifest_path = checkpoint / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["campaign_id"] = "recovery-campaign"
            manifest["databases"]["campaign"] = _campaign_database_summary(
                campaign_database
            )
            manifest["files"].append(
                {
                    "path": "campaign.sqlite3",
                    "size_bytes": len(content),
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
            )
            manifest["files"].sort(key=lambda item: item["path"])
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )

            destination = root / "must-not-restore-grafted-campaign"
            with self.assertRaisesRegex(ValueError, "non-Campaign"):
                restore_checkpoint(checkpoint, destination)
            self.assertFalse(destination.exists())

    def test_real_campaign_checkpoint_restores_campaign_context(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "source"
            config = _config(source)
            _baseline(config.state_dir)
            baseline = BaselineRef.create(
                namespace=LEGACY_RESEARCH_NAMESPACE,
                artifact_id="source-sha256-v1:" + SEED_HASH,
                source="campaign",
                revision="recovery-campaign-seed",
            )
            proposer_profile = {"kind": "fixture", "revision": "v1"}
            campaign_database = source / "campaign.sqlite3"
            with CampaignStore(campaign_database) as campaign_store:
                campaign = campaign_store.create_campaign(
                    campaign_id="recovery-campaign",
                    namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                    mode="DISCOVERY",
                    snapshot={"profile": "fixture"},
                    budget_limit=BudgetAmount(),
                    initial_baseline_ref=baseline,
                    initial_policy_snapshot={"policy": "fixture"},
                )
                campaign_store.start_campaign("recovery-campaign")
                child = campaign_store.create_child_run(
                    "recovery-campaign",
                    proposer_profile=proposer_profile,
                    controller_run_id="recovery-campaign-run",
                    max_candidates=config.max_candidates,
                    max_wall_seconds=round(config.max_hours * 3600),
                    max_consecutive_failures=config.max_consecutive_failures,
                )
                campaign_store.start_child_run(
                    child["id"], controller_run_id="recovery-campaign-run"
                )

            workflow_snapshot = {
                "schema_version": 2,
                "mode": "DISCOVERY",
                "campaign_id": "recovery-campaign",
                "campaign_snapshot_digest": campaign["snapshot_digest"],
                "campaign_child": {
                    "child_id": child["id"],
                    "child_index": child["child_index"],
                    "controller_run_id": "recovery-campaign-run",
                },
                "proposer_profile": proposer_profile,
            }
            controller = ResearchController(config, evaluator=FakeEvaluator())
            with ControllerStore(controller.controller_db) as store:
                store.create_run(
                    run_id="recovery-campaign-run",
                    deadline_epoch=time.time() + 60,
                    config=config.redacted_dict(),
                    initial_best_hash=SEED_HASH,
                    namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
                    resolved_config_digest="sha256:" + "c" * 64,
                    workflow_snapshot=workflow_snapshot,
                    baseline_ref=baseline.to_dict(),
                    history_cutoff=1,
                )

            checkpoint = Path(
                controller.checkpoint("recovery-campaign-run")["path"]
            )
            verified = verify_checkpoint(checkpoint)
            self.assertEqual(
                verified["manifest"]["campaign_id"], "recovery-campaign"
            )
            self.assertIn("campaign", verified["databases"])

            destination = root / "restored-campaign-runtime"
            report = restore_checkpoint(checkpoint, destination)

            self.assertEqual(report["status"], "READY_FOR_MANUAL_SWITCH")
            self.assertTrue(
                (destination / "campaign" / "campaign.sqlite3").is_file()
            )


if __name__ == "__main__":
    unittest.main()
