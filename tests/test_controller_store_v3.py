from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
import unittest

from kernel_research.history import (
    LEGACY_NAMESPACE_ID as HISTORY_LEGACY_NAMESPACE_ID,
)
from kernel_research.platform.profiles import LEGACY_RESEARCH_NAMESPACE
from kernel_research.autorun.store import (
    ControllerStore,
    LEGACY_HISTORY_CUTOFF,
    LEGACY_NAMESPACE_ID,
    LEGACY_RESOLVED_CONFIG_DIGEST,
    LEGACY_WORKFLOW_SNAPSHOT,
    SCHEMA_VERSION,
    _V2_TO_V3_MIGRATION_CAPABILITY,
)


SEED_HASH = "a" * 64


def _create_v2_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE runs (
            id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            deadline_epoch REAL NOT NULL,
            valid_candidates INTEGER NOT NULL DEFAULT 0,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            stop_requested INTEGER NOT NULL DEFAULT 0,
            stop_reason TEXT,
            config_json TEXT NOT NULL,
            preflight_json TEXT NOT NULL,
            initial_best_hash TEXT NOT NULL,
            final_best_hash TEXT
        );
        CREATE TABLE iterations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            iteration_index INTEGER NOT NULL,
            status TEXT NOT NULL,
            stage TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            parent_hash TEXT NOT NULL,
            candidate_hash TEXT,
            hypothesis TEXT,
            rationale TEXT,
            prompt_path TEXT,
            raw_output_path TEXT,
            candidate_path TEXT,
            active_container TEXT,
            experiment_ids_json TEXT NOT NULL DEFAULT '{}',
            result_json TEXT NOT NULL DEFAULT '{}',
            error TEXT,
            outcome TEXT,
            UNIQUE(run_id, iteration_index)
        );
        CREATE INDEX iterations_candidate_idx
            ON iterations(candidate_hash, id);
        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
            iteration_id INTEGER REFERENCES iterations(id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            event TEXT NOT NULL,
            details_json TEXT NOT NULL
        );
        CREATE INDEX events_run_iteration_idx
            ON events(run_id, iteration_id, id);
        INSERT INTO runs (
            id, status, created_at, updated_at, deadline_epoch,
            config_json, preflight_json, initial_best_hash
        ) VALUES (
            'legacy-run', 'BUDGET_EXHAUSTED', '2026-01-01T00:00:00Z',
            '2026-01-01T00:00:00Z', 9999999999, '{}', '{}',
            'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        );
        INSERT INTO iterations (
            run_id, iteration_index, status, stage, created_at, updated_at,
            parent_hash
        ) VALUES (
            'legacy-run', 1, 'COMPLETED', 'DONE',
            '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z',
            'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        );
        PRAGMA user_version = 2;
        """
    )
    connection.close()


class ControllerStoreV3Tests(unittest.TestCase):
    def _create_run_and_iteration(
        self, store: ControllerStore, *, run_id: str = "run"
    ) -> tuple[dict, dict]:
        run = store.create_run(
            run_id=run_id,
            deadline_epoch=time.time() + 60,
            config={"fixture": True},
            initial_best_hash=SEED_HASH,
        )
        iteration = store.create_iteration(run_id, 1, SEED_HASH)
        return run, iteration

    def test_persisted_version_zero_with_user_objects_fails_before_wal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "controller.sqlite3"
            _create_v2_database(database)
            connection = sqlite3.connect(database)
            connection.execute("PRAGMA user_version = 0")
            connection.commit()
            connection.close()
            before = database.read_bytes()

            with self.assertRaisesRegex(
                RuntimeError, "version 0 contains user objects"
            ):
                ControllerStore(database)

            self.assertEqual(database.read_bytes(), before)
            self.assertFalse(Path(str(database) + "-wal").exists())
            self.assertFalse(Path(str(database) + "-shm").exists())

    def test_persisted_empty_version_zero_database_may_initialize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "controller.sqlite3"
            sqlite3.connect(database).close()

            with ControllerStore(database) as store:
                self.assertEqual(
                    store.connection.execute("PRAGMA user_version").fetchone()[0],
                    SCHEMA_VERSION,
                )

    def test_persisted_legacy_store_open_fails_closed_without_capability(self) -> None:
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temporary:
                database = Path(temporary) / "controller.sqlite3"
                _create_v2_database(database)
                if version == 1:
                    connection = sqlite3.connect(database)
                    connection.execute("PRAGMA user_version = 1")
                    connection.close()
                before = database.read_bytes()
                with self.assertRaisesRegex(RuntimeError, "coordinated"):
                    ControllerStore(database)
                with self.assertRaisesRegex(RuntimeError, "coordinated"):
                    ControllerStore(
                        database,
                        _v2_to_v3_migration_capability=True,
                    )
                self.assertEqual(database.read_bytes(), before)

    def test_v2_migration_uses_explicit_legacy_unknowns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "controller.sqlite3"
            _create_v2_database(database)

            with ControllerStore(
                database,
                _v2_to_v3_migration_capability=(
                    _V2_TO_V3_MIGRATION_CAPABILITY
                ),
            ) as store:
                self.assertEqual(
                    store.connection.execute("PRAGMA user_version").fetchone()[0],
                    SCHEMA_VERSION,
                )
                run = store.get_run("legacy-run")
                self.assertEqual(run["namespace_id"], LEGACY_NAMESPACE_ID)
                self.assertEqual(
                    run["resolved_config_digest"],
                    LEGACY_RESOLVED_CONFIG_DIGEST,
                )
                self.assertEqual(
                    run["workflow_snapshot"], LEGACY_WORKFLOW_SNAPSHOT
                )
                self.assertEqual(
                    run["baseline_ref"],
                    {
                        "schema_version": 1,
                        "namespace_id": LEGACY_NAMESPACE_ID,
                        "artifact_id": "source-sha256-v1:" + SEED_HASH,
                        "source": "deployment",
                        "revision": "legacy",
                    },
                )
                self.assertEqual(
                    run["history_cutoff"], LEGACY_HISTORY_CUTOFF
                )
                with self.assertRaisesRegex(sqlite3.IntegrityError, "immutable"):
                    store.connection.execute(
                        """
                        UPDATE runs SET namespace_id = 'different'
                        WHERE id = 'legacy-run'
                        """
                    )
                tables = {
                    row[0]
                    for row in store.connection.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                self.assertIn("proposal_attempts", tables)
                self.assertIn("evaluation_attempts", tables)

    def test_v2_migration_rejects_any_nonterminal_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "controller.sqlite3"
            _create_v2_database(database)
            connection = sqlite3.connect(database)
            connection.execute(
                "UPDATE runs SET status = 'RUNNING' WHERE id = 'legacy-run'"
            )
            connection.execute(
                "UPDATE iterations SET status = 'RUNNING', stage = 'PROPOSE'"
            )
            connection.commit()
            connection.close()

            with self.assertRaisesRegex(RuntimeError, "must be terminal"):
                ControllerStore(
                    database,
                    _v2_to_v3_migration_capability=(
                        _V2_TO_V3_MIGRATION_CAPABILITY
                    ),
                )
            connection = sqlite3.connect(database)
            try:
                self.assertEqual(
                    connection.execute("PRAGMA user_version").fetchone()[0], 2
                )
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(runs)")
                }
                self.assertNotIn("namespace_id", columns)
            finally:
                connection.close()

    def test_legacy_namespace_has_one_cross_store_identity(self) -> None:
        self.assertEqual(
            LEGACY_NAMESPACE_ID, LEGACY_RESEARCH_NAMESPACE.namespace_id
        )
        self.assertEqual(LEGACY_NAMESPACE_ID, HISTORY_LEGACY_NAMESPACE_ID)

    def test_v2_migration_rejects_unprovable_baseline_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "controller.sqlite3"
            _create_v2_database(database)
            connection = sqlite3.connect(database)
            connection.execute(
                "UPDATE runs SET initial_best_hash = 'not-a-sha256'"
            )
            connection.commit()
            connection.close()
            with self.assertRaisesRegex(
                RuntimeError, "cannot form a BaselineRef"
            ):
                ControllerStore(
                    database,
                    _v2_to_v3_migration_capability=(
                        _V2_TO_V3_MIGRATION_CAPABILITY
                    ),
                )
            connection = sqlite3.connect(database)
            self.assertEqual(
                connection.execute("PRAGMA user_version").fetchone()[0], 2
            )
            connection.close()

    def test_run_v3_identity_is_returned_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "controller.sqlite3"
            with ControllerStore(database) as store:
                namespace_id = "sha256:" + "b" * 64
                run = store.create_run(
                    run_id="v3-run",
                    deadline_epoch=time.time() + 60,
                    config={"redacted": True},
                    initial_best_hash=SEED_HASH,
                    namespace_id=namespace_id,
                    resolved_config_digest="sha256:" + "c" * 64,
                    workflow_snapshot={"stages": ["SMOKE", "QUICK"]},
                    baseline_ref={
                        "schema_version": 1,
                        "namespace_id": namespace_id,
                        "artifact_id": "source-sha256-v1:" + SEED_HASH,
                        "source": "deployment",
                        "revision": "fixture",
                    },
                    history_cutoff=None,
                )
                self.assertEqual(run["namespace_id"], namespace_id)
                self.assertEqual(
                    run["workflow_snapshot"]["stages"], ["SMOKE", "QUICK"]
                )
                self.assertIsNone(run["history_cutoff"])
                with self.assertRaisesRegex(ValueError, "unsupported run"):
                    store.update_run("v3-run", namespace_id="changed")

    def test_proposal_attempt_terminal_state_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "controller.sqlite3"
            with ControllerStore(database) as store:
                _, iteration = self._create_run_and_iteration(store)
                attempt = store.create_proposal_attempt(
                    run_id="run",
                    iteration_id=iteration["id"],
                    attempt_uid="proposal-uid",
                    proposal_context_id="sha256:context",
                    request={"prompt": "object-id"},
                )
                duplicate = store.create_proposal_attempt(
                    run_id="run",
                    iteration_id=iteration["id"],
                    attempt_uid="proposal-uid",
                    proposal_context_id="sha256:context",
                    request={"prompt": "object-id"},
                )
                self.assertEqual(duplicate["id"], attempt["id"])
                finished = store.finish_proposal_attempt(
                    attempt["id"],
                    status="SUCCEEDED",
                    result={"schema_version": 2},
                    candidate_artifact_id="bundle-sha256-v1:candidate",
                    input_tokens=10,
                    output_tokens=20,
                    latency_ms=12.5,
                )
                retry = store.finish_proposal_attempt(
                    attempt["id"],
                    status="SUCCEEDED",
                    result={"schema_version": 2},
                    candidate_artifact_id="bundle-sha256-v1:candidate",
                    input_tokens=10,
                    output_tokens=20,
                    latency_ms=12.5,
                )
                self.assertEqual(retry, finished)
                with self.assertRaisesRegex(ValueError, "conflicting"):
                    store.finish_proposal_attempt(
                        attempt["id"],
                        status="FAILED",
                        error="late rewrite",
                    )

    def test_evaluation_uid_reconciliation_and_backup_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "controller.sqlite3"
            backup = root / "backup.sqlite3"
            with ControllerStore(database) as store:
                _, iteration = self._create_run_and_iteration(store)
                intent = {
                    "experiment_uid": "experiment-uid",
                    "run_id": "run",
                    "iteration_id": iteration["id"],
                    "stage": "SMOKE",
                    "suite": "smoke",
                    "candidate_artifact_id": "bundle-sha256-v1:candidate",
                    "condition_digest": "sha256:condition",
                    "request": {"identity": "frozen"},
                }
                created = store.create_evaluation_attempt(**intent)
                duplicate = store.create_evaluation_attempt(**intent)
                self.assertEqual(created["id"], duplicate["id"])
                with self.assertRaisesRegex(ValueError, "conflicts"):
                    store.create_evaluation_attempt(
                        **{**intent, "suite": "quick"}
                    )
                store.start_evaluation_attempt("experiment-uid")
                store.finish_evaluation_attempt(
                    "experiment-uid",
                    status="SUCCEEDED",
                    result={"status": "SUCCESS"},
                )
                self.assertEqual(
                    [
                        item["experiment_uid"]
                        for item in store.list_unlinked_evaluation_attempts()
                    ],
                    ["experiment-uid"],
                )
                linked = store.link_history_experiment("experiment-uid", 41)
                retried = store.reconcile_evaluation_attempt(
                    "experiment-uid", 41
                )
                self.assertEqual(linked, retried)
                self.assertEqual(linked["history_experiment_id"], 41)
                self.assertEqual(store.list_unlinked_evaluation_attempts(), [])
                with self.assertRaisesRegex(ValueError, "another History"):
                    store.link_history_experiment("experiment-uid", 42)
                store.backup_to(backup)

            with ControllerStore(backup) as restored:
                attempt = restored.get_evaluation_attempt_by_uid(
                    "experiment-uid"
                )
                self.assertIsNotNone(attempt)
                self.assertEqual(attempt["history_experiment_id"], 41)
                self.assertEqual(
                    restored.get_run("run")["namespace_id"],
                    LEGACY_NAMESPACE_ID,
                )

    def test_unknown_evaluation_outcome_cannot_restart_or_link(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "controller.sqlite3"
            with ControllerStore(database) as store:
                _, iteration = self._create_run_and_iteration(store)
                store.create_evaluation_attempt(
                    experiment_uid="unknown-uid",
                    run_id="run",
                    iteration_id=iteration["id"],
                    stage="FULL_PRIMARY",
                    suite="full",
                )
                store.start_evaluation_attempt("unknown-uid")
                store.finish_evaluation_attempt(
                    "unknown-uid",
                    status="UNKNOWN_OUTCOME",
                    error="host terminated during GPU execution",
                )
                with self.assertRaisesRegex(ValueError, "may not restart"):
                    store.start_evaluation_attempt("unknown-uid")
                with self.assertRaisesRegex(ValueError, "succeeded"):
                    store.link_history_experiment("unknown-uid", 9)
                self.assertEqual(store.list_unlinked_evaluation_attempts(), [])

    def test_attempt_ownership_and_frozen_baseline_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "controller.sqlite3"
            with ControllerStore(database) as store:
                _, first = self._create_run_and_iteration(store)
                _, second = self._create_run_and_iteration(
                    store, run_id="other-run"
                )
                with self.assertRaisesRegex(ValueError, "does not belong"):
                    store.create_proposal_attempt(
                        run_id="run",
                        iteration_id=second["id"],
                    )
                with self.assertRaisesRegex(ValueError, "frozen run baseline"):
                    store.create_evaluation_attempt(
                        experiment_uid="wrong-baseline",
                        run_id="run",
                        iteration_id=first["id"],
                        stage="SMOKE",
                        suite="smoke",
                        baseline_ref={"artifact_id": "different"},
                    )
                store.create_evaluation_attempt(
                    experiment_uid="pending",
                    run_id="run",
                    iteration_id=first["id"],
                    stage="SMOKE",
                    suite="smoke",
                )
                with self.assertRaisesRegex(ValueError, "must be started"):
                    store.finish_evaluation_attempt(
                        "pending", status="FAILED", error="not started"
                    )


if __name__ == "__main__":
    unittest.main()
