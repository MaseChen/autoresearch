from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest

from kernel_research.history import (
    HistoryStore,
    LEGACY_NAMESPACE_ID,
    SCHEMA_VERSION,
    _V2_TO_V3_MIGRATION_CAPABILITY,
)
from kernel_research.platform import CandidateBundle, canonical_json_bytes
from kernel_research.platform import (
    ArtifactId,
    BaselineRef,
    ExperimentIdentity,
    LEGACY_RESEARCH_NAMESPACE,
)


SOURCE = "def run_kernel(*args):\n    return None\n"


def _create_legacy_database(root: Path, version: int) -> Path:
    database = root / f"history-v{version}.sqlite3"
    candidate_hash = hashlib.sha256(SOURCE.encode("utf-8")).hexdigest()
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    artifact_path = artifacts / f"{candidate_hash}.py"
    artifact_path.write_text(SOURCE, encoding="utf-8")
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schema_version INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            candidate_hash TEXT NOT NULL,
            git_commit TEXT,
            backend TEXT NOT NULL,
            suite TEXT NOT NULL,
            status TEXT NOT NULL,
            promotable INTEGER NOT NULL,
            duplicate_of_id INTEGER REFERENCES experiments(id),
            aggregate_score REAL,
            note TEXT NOT NULL,
            environment_json TEXT NOT NULL,
            error_summary TEXT,
            artifact_path TEXT NOT NULL,
            result_json TEXT NOT NULL
        );
        CREATE TABLE case_measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL
                REFERENCES experiments(id) ON DELETE CASCADE,
            case_name TEXT NOT NULL,
            matched_ratio REAL,
            passed INTEGER,
            raw_samples_json TEXT NOT NULL,
            baseline_samples_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            UNIQUE(experiment_id, case_name)
        );
        """
    )
    connection.execute(
        """
        INSERT INTO experiments(
            schema_version, created_at, candidate_hash, git_commit, backend,
            suite, status, promotable, duplicate_of_id, aggregate_score, note,
            environment_json, error_summary, artifact_path, result_json
        ) VALUES (?, '2026-08-01T00:00:00.000Z', ?, NULL, 'c500', 'full',
                  'SUCCESS', 1, NULL, 1.0, 'legacy', '{}', NULL, ?, '{}')
        """,
        (version, candidate_hash, artifact_path.relative_to(root).as_posix()),
    )
    connection.execute(f"PRAGMA user_version = {version}")
    connection.commit()
    connection.close()
    return database


class HistoryV3Tests(unittest.TestCase):
    def test_persisted_version_zero_with_user_objects_fails_before_wal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = _create_legacy_database(root, 2)
            connection = sqlite3.connect(database)
            connection.execute("PRAGMA user_version = 0")
            connection.commit()
            connection.close()
            before = database.read_bytes()

            with self.assertRaisesRegex(ValueError, "version 0 contains user objects"):
                HistoryStore(database, root)

            self.assertEqual(database.read_bytes(), before)
            self.assertFalse(Path(str(database) + "-wal").exists())
            self.assertFalse(Path(str(database) + "-shm").exists())

    def test_persisted_empty_version_zero_database_may_initialize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "history.sqlite3"
            sqlite3.connect(database).close()

            with HistoryStore(database, root) as history:
                self.assertEqual(history.schema_version, SCHEMA_VERSION)

    def test_persisted_legacy_store_open_fails_closed_without_capability(self) -> None:
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                database = _create_legacy_database(root, version)
                before = database.read_bytes()
                with self.assertRaisesRegex(ValueError, "coordinated"):
                    HistoryStore(database, root)
                with self.assertRaisesRegex(ValueError, "coordinated"):
                    HistoryStore(
                        database,
                        root,
                        _v2_to_v3_migration_capability=True,
                    )
                self.assertEqual(database.read_bytes(), before)

    def test_v1_and_v2_migrate_to_stable_legacy_identity(self) -> None:
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                database = _create_legacy_database(root, version)
                with HistoryStore(
                    database,
                    root,
                    _v2_to_v3_migration_capability=(
                        _V2_TO_V3_MIGRATION_CAPABILITY
                    ),
                ) as history:
                    self.assertEqual(history.schema_version, SCHEMA_VERSION)
                    record = history.list_experiments()[0]
                    self.assertEqual(record.namespace_id, LEGACY_NAMESPACE_ID)
                    self.assertTrue(record.experiment_uid.startswith("exp-"))
                    self.assertEqual(
                        record.artifact_id,
                        f"source-sha256-v1:{record.candidate_hash}",
                    )
                    self.assertEqual(record.replicate_kind, "legacy")
                    self.assertEqual(
                        history.get_store_metadata("schema")["schema_version"], 3
                    )
                    tables = {
                        row[0]
                        for row in history._connection.execute(  # noqa: SLF001
                            "SELECT name FROM sqlite_master WHERE type = 'table'"
                        )
                    }
                    self.assertTrue(
                        {
                            "store_metadata",
                            "research_namespaces",
                            "candidate_artifacts",
                            "experiment_relations",
                        }.issubset(tables)
                    )

    def test_v2_migration_rejects_missing_or_tampered_artifacts(self) -> None:
        for failure in ("missing", "tampered"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                database = _create_legacy_database(root, 2)
                artifact = next((root / "artifacts").glob("*.py"))
                if failure == "missing":
                    artifact.unlink()
                else:
                    artifact.write_text("tampered", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "legacy artifact"):
                    HistoryStore(
                        database,
                        root,
                        _v2_to_v3_migration_capability=(
                            _V2_TO_V3_MIGRATION_CAPABILITY
                        ),
                    )
                connection = sqlite3.connect(database)
                try:
                    self.assertEqual(
                        connection.execute("PRAGMA user_version").fetchone()[0], 2
                    )
                    self.assertIsNone(
                        connection.execute(
                            """
                            SELECT name FROM sqlite_master
                            WHERE type = 'table' AND name = 'research_namespaces'
                            """
                        ).fetchone()
                    )
                finally:
                    connection.close()

    def test_namespace_isolation_replicates_and_relations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with HistoryStore(root / "history.sqlite3", root) as history:
                history.ensure_namespace("ns-a", {"profile": "a"})
                history.ensure_namespace("ns-b", {"profile": "b"})
                first = history.record_experiment(
                    candidate_source=SOURCE,
                    backend="c500",
                    suite="full",
                    status="SUCCESS",
                    promotable=True,
                    aggregate_score=1.1,
                    namespace_id="ns-a",
                    experiment_uid="exp-a-primary",
                    replicate_kind="primary",
                )
                other_namespace = history.record_experiment(
                    candidate_source=SOURCE,
                    backend="c500",
                    suite="full",
                    status="SUCCESS",
                    promotable=True,
                    aggregate_score=2.0,
                    namespace_id="ns-b",
                    experiment_uid="exp-b-primary",
                    replicate_kind="primary",
                )
                confirmation = history.record_experiment(
                    candidate_source=SOURCE,
                    backend="c500",
                    suite="full",
                    status="SUCCESS",
                    namespace_id="ns-a",
                    experiment_uid="exp-a-confirmation",
                    replicate_kind="confirmation",
                    replicate_index=1,
                    baseline_experiment_uid=first.experiment_uid,
                )

                self.assertFalse(first.duplicate)
                self.assertFalse(other_namespace.duplicate)
                self.assertFalse(confirmation.duplicate)
                self.assertEqual(history.list_experiments(), [])
                self.assertEqual(len(history.list_experiments_for_namespace("ns-a")), 2)
                self.assertEqual(
                    history.get_best_for_namespace("ns-a").experiment_uid,
                    "exp-a-primary",
                )
                self.assertEqual(
                    history.get_best_for_namespace("ns-b").experiment_uid,
                    "exp-b-primary",
                )
                self.assertTrue(
                    history.has_candidate_in_namespace("ns-a", first.artifact_id)
                )
                self.assertTrue(
                    history.has_candidate_in_namespace("ns-b", first.candidate_hash)
                )
                relations = history.list_experiment_relations(
                    confirmation.experiment_uid, direction="source"
                )
                self.assertEqual(len(relations), 1)
                self.assertEqual(relations[0].relation_type, "baseline")

    def test_experiment_uid_is_idempotent_and_conflicts_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with HistoryStore(root / "history.sqlite3", root) as history:
                history.ensure_namespace("ns", {"profile": "stable"})
                values = {
                    "candidate_source": SOURCE,
                    "backend": "c500",
                    "suite": "quick",
                    "status": "SUCCESS",
                    "namespace_id": "ns",
                    "experiment_uid": "exp-idempotent",
                    "replicate_kind": "noise",
                    "replicate_index": 3,
                }
                first = history.record_experiment(**values)
                replay = history.record_experiment(**values)
                self.assertEqual(replay.id, first.id)
                self.assertEqual(
                    history.get_experiment_by_uid("exp-idempotent").id, first.id
                )
                self.assertEqual(len(history.list_experiments_for_namespace("ns")), 1)
                with self.assertRaisesRegex(ValueError, "different experiment data"):
                    history.record_experiment(**values, note="conflicting replay")

    def test_scientific_cas_checks_ids_and_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with HistoryStore(root / "history.sqlite3", root) as history:
                bundle = CandidateBundle.single_file(content=SOURCE)
                content = canonical_json_bytes(bundle.to_dict(include_content=True))
                artifact_id = str(bundle.artifact_id)
                artifact = history.store_candidate_bundle(bundle)
                replay = history.store_candidate_bundle(bundle)
                self.assertEqual(replay, artifact)
                self.assertTrue(artifact.object_path.startswith("objects/sha256/"))
                self.assertEqual(history.read_candidate_artifact(artifact_id), content)
                with self.assertRaisesRegex(RuntimeError, "collision"):
                    history.store_candidate_artifact(
                        b"different",
                        artifact_id=artifact_id,
                        artifact_kind="source_bundle_v1",
                        manifest=bundle.manifest,
                    )
                (root / artifact.object_path).write_bytes(b"tampered")
                with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                    history.read_candidate_artifact(artifact_id)

    def test_platform_experiment_identity_is_preserved_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = CandidateBundle.single_file(content=SOURCE)
            parent = ArtifactId.for_source_text(SOURCE + "# parent\n")
            baseline = BaselineRef.create(
                namespace=LEGACY_RESEARCH_NAMESPACE,
                artifact_id=parent,
                source="deployment",
                revision="deployment-1",
            )
            identity = ExperimentIdentity.create(
                namespace=LEGACY_RESEARCH_NAMESPACE,
                mode="DISCOVERY",
                candidate_artifact_id=bundle.artifact_id,
                parent_artifact_id=parent,
                baseline=baseline,
                stage="QUICK",
                suite="quick",
                run_id="run-1",
                iteration=1,
                replicate_kind="noise",
                replicate_index=2,
            )
            with HistoryStore(root / "history.sqlite3", root) as history:
                history.store_candidate_bundle(bundle)
                record = history.record_experiment(
                    candidate_source=SOURCE,
                    backend="c500",
                    suite="quick",
                    status="SUCCESS",
                    identity=identity,
                )
                self.assertEqual(record.experiment_uid, identity.experiment_uid)
                self.assertEqual(record.namespace_id, identity.namespace_id)
                self.assertEqual(record.artifact_id, str(bundle.artifact_id))
                self.assertEqual(record.condition_digest, identity.condition_digest)
                self.assertEqual(record.identity, identity.to_dict())


if __name__ == "__main__":
    unittest.main()
