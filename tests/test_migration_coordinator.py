from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sqlite3
import subprocess
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import kernel_research.migration as migration_module
from kernel_research import cli as legacy_cli
from kernel_research.autorun import admin as autorun_admin
from kernel_research.autorun import cli as autorun_cli
from kernel_research.autorun.store import ControllerStore
from kernel_research.migration import (
    CoordinatedMigrationError,
    coordinate_v2_to_v3_migration,
    preflight_v2_to_v3_migration,
    production_cli_schema_guard,
    verify_v2_migration_checkpoint,
)

from test_controller_store_v3 import _create_v2_database
from test_history_v3 import _create_legacy_database


def _version(path: Path) -> int:
    connection = sqlite3.connect(path)
    try:
        return int(connection.execute("PRAGMA user_version").fetchone()[0])
    finally:
        connection.close()


class MigrationCoordinatorTests(unittest.TestCase):
    def _pair(self, root: Path) -> tuple[Path, Path, Path, dict[str, object]]:
        state = root / "state"
        controller = root / "controller"
        state.mkdir(parents=True)
        controller.mkdir(parents=True)
        generated_history = _create_legacy_database(state, 2)
        history_db = state / "history.sqlite3"
        generated_history.replace(history_db)
        controller_db = controller / "controller.sqlite3"
        _create_v2_database(controller_db)
        repository = root / "repository"
        repository.mkdir()
        subprocess.run(("git", "init", "--quiet", str(repository)), check=True)
        subprocess.run(
            ("git", "-C", str(repository), "config", "user.email", "test@example.com"),
            check=True,
        )
        subprocess.run(
            ("git", "-C", str(repository), "config", "user.name", "Test"),
            check=True,
        )
        (repository / "kernel.py").write_text("def kernel():\n    pass\n", encoding="utf-8")
        subprocess.run(("git", "-C", str(repository), "add", "kernel.py"), check=True)
        subprocess.run(
            ("git", "-C", str(repository), "commit", "--quiet", "-m", "fixture"),
            check=True,
        )
        commit = subprocess.run(
            ("git", "-C", str(repository), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        config_path = root / "controller-config.json"
        config_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "repository_dir": str(repository),
                    "state_dir": str(state),
                    "controller_dir": str(controller),
                    "deepseek_key_file": "/run/credentials/deepseek.key",
                    "expected_git_commit": commit,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return state, history_db, controller_db, {
            "config_path": config_path,
            "repository": repository,
            "expected_git_commit": commit,
        }

    def test_preflight_rejects_inline_secret_without_echoing_value(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            state, history_db, controller_db, source = self._pair(root)
            secret = "must-never-appear-in-error"
            Path(source["config_path"]).write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "repository_dir": str(source["repository"]),
                        "state_dir": str(state),
                        "controller_dir": str(controller_db.parent),
                        "expected_git_commit": source["expected_git_commit"],
                        "api_key": secret,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "inline secret") as raised:
                preflight_v2_to_v3_migration(
                    history_db=history_db,
                    controller_db=controller_db,
                    state_dir=state,
                    **source,
                )
            self.assertNotIn(secret, str(raised.exception))
            self.assertEqual(_version(history_db), 2)
            self.assertEqual(_version(controller_db), 2)

    def test_preflight_requires_clean_exact_non_symlink_repository(self) -> None:
        for failure in ("dirty", "wrong-head", "symlink"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                state, history_db, controller_db, source = self._pair(root)
                selected = dict(source)
                if failure == "dirty":
                    (Path(source["repository"]) / "untracked.txt").write_text(
                        "dirty", encoding="utf-8"
                    )
                    pattern = "clean"
                elif failure == "wrong-head":
                    selected["expected_git_commit"] = "0" * 40
                    pattern = "does not match"
                else:
                    alias = root / "repository-alias"
                    alias.symlink_to(source["repository"], target_is_directory=True)
                    selected["repository"] = alias
                    pattern = "symlink"
                with self.assertRaisesRegex(ValueError, pattern):
                    preflight_v2_to_v3_migration(
                        history_db=history_db,
                        controller_db=controller_db,
                        state_dir=state,
                        **selected,
                    )
                self.assertEqual(_version(history_db), 2)
                self.assertEqual(_version(controller_db), 2)

    def test_success_publishes_verified_full_checkpoint_before_both_v3_stores(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            state, history_db, controller_db, source = self._pair(root)
            for database in (history_db, controller_db):
                connection = sqlite3.connect(database)
                try:
                    self.assertEqual(
                        connection.execute(
                            "PRAGMA journal_mode = WAL"
                        ).fetchone(),
                        ("wal",),
                    )
                finally:
                    connection.close()
            extra = state / "artifacts" / "manifest.json"
            extra.write_text("preserve me", encoding="utf-8")

            ready = preflight_v2_to_v3_migration(
                history_db=history_db,
                controller_db=controller_db,
                state_dir=state,
                **source,
            )
            self.assertEqual(ready["status"], "READY")
            self.assertEqual(ready["databases"]["history"]["user_version"], 2)
            self.assertEqual(
                ready["databases"]["controller"]["terminal_state_check"],
                "ok",
            )

            result = coordinate_v2_to_v3_migration(
                history_db=history_db,
                controller_db=controller_db,
                state_dir=state,
                checkpoint_root=root / "migration-checkpoints",
                operation_id="production-001",
                **source,
            )

            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(result["versions_after"], {"history": 3, "controller": 3})
            self.assertEqual(_version(history_db), 3)
            self.assertEqual(_version(controller_db), 3)
            checkpoint = Path(result["checkpoint"])
            verified = verify_v2_migration_checkpoint(checkpoint)
            self.assertEqual(verified["status"], "VERIFIED")
            self.assertEqual(list(checkpoint.glob("*.sqlite3-*")), [])
            for database in (
                checkpoint / "history.sqlite3",
                checkpoint / "controller.sqlite3",
            ):
                connection = sqlite3.connect(
                    database.resolve().as_uri() + "?mode=ro",
                    uri=True,
                )
                try:
                    self.assertEqual(
                        connection.execute(
                            "PRAGMA journal_mode"
                        ).fetchone(),
                        ("delete",),
                    )
                finally:
                    connection.close()
            self.assertEqual(
                (checkpoint / "config.json").read_bytes(),
                Path(source["config_path"]).read_bytes(),
            )
            self.assertTrue((checkpoint / "code" / "repository.bundle").is_file())
            self.assertEqual(
                verified["code"]["expected_git_commit"],
                source["expected_git_commit"],
            )
            self.assertEqual(verified["code"]["offline_verify"], "git_bundle_verify_ok")
            self.assertEqual(
                (checkpoint / "artifacts" / extra.name).read_text(encoding="utf-8"),
                "preserve me",
            )
            self.assertFalse(result["single_database_rollback_supported"])
            self.assertEqual(result["rollback_scope"], "FULL_CHECKPOINT_ONLY")
            self.assertFalse(
                (state / ".v2-to-v3-migration-intent.json").exists()
            )
            self.assertFalse(
                (
                    controller_db.parent
                    / ".v2-to-v3-migration-intent.json"
                ).exists()
            )
            self.assertEqual(
                production_cli_schema_guard(
                    history_db=history_db,
                    controller_db=controller_db,
                ),
                {"history": 3, "controller": 3},
            )

    def test_nonterminal_controller_fails_before_checkpoint_or_store_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            state, history_db, controller_db, source = self._pair(root)
            connection = sqlite3.connect(controller_db)
            connection.execute(
                "UPDATE iterations SET status = 'RUNNING', stage = 'QUICK'"
            )
            connection.commit()
            connection.close()
            checkpoint_root = root / "migration-checkpoints"

            with self.assertRaises(CoordinatedMigrationError) as raised:
                coordinate_v2_to_v3_migration(
                    history_db=history_db,
                    controller_db=controller_db,
                    state_dir=state,
                    checkpoint_root=checkpoint_root,
                    operation_id="must-not-start",
                    **source,
                )

            self.assertEqual(raised.exception.phase, "preflight")
            self.assertFalse(raised.exception.live_migration_started)
            self.assertIsNone(raised.exception.checkpoint_path)
            self.assertEqual(_version(history_db), 2)
            self.assertEqual(_version(controller_db), 2)
            self.assertEqual(list(checkpoint_root.glob("v2-to-v3-*")), [])

    def test_partial_live_failure_never_rolls_back_one_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            state, history_db, controller_db, source = self._pair(root)
            real_controller_store = ControllerStore
            calls = 0

            def controller_store(path: Path, **kwargs: object) -> ControllerStore:
                nonlocal calls
                calls += 1
                if calls == 1:  # checkpoint dry-run
                    return real_controller_store(path, **kwargs)
                raise RuntimeError("injected live Controller migration failure")

            with (
                mock.patch(
                    "kernel_research.migration.ControllerStore",
                    side_effect=controller_store,
                ),
                self.assertRaises(CoordinatedMigrationError) as raised,
            ):
                coordinate_v2_to_v3_migration(
                    history_db=history_db,
                    controller_db=controller_db,
                    state_dir=state,
                    checkpoint_root=root / "migration-checkpoints",
                    operation_id="partial-proof",
                    **source,
                )

            failure = raised.exception
            self.assertEqual(failure.phase, "live-controller-migration")
            self.assertTrue(failure.live_migration_started)
            self.assertIn("never roll back one database", str(failure))
            self.assertEqual(
                failure.observed_versions,
                {"history": 3, "controller": 2},
            )
            self.assertEqual(_version(history_db), 3)
            self.assertEqual(_version(controller_db), 2)
            self.assertTrue(
                (state / ".v2-to-v3-migration-intent.json").is_file()
            )
            self.assertTrue(
                (
                    controller_db.parent
                    / ".v2-to-v3-migration-intent.json"
                ).is_file()
            )
            with redirect_stderr(io.StringIO()) as legacy_error:
                self.assertEqual(
                    legacy_cli.main(["history", "--state-dir", str(state)]),
                    2,
                )
            self.assertIn("fenced", legacy_error.getvalue())
            with self.assertRaisesRegex(RuntimeError, "fenced"):
                autorun_admin._active_run_error(  # noqa: SLF001
                    SimpleNamespace(controller_dir=controller_db.parent)
                )
            self.assertIsNotNone(failure.checkpoint_path)
            self.assertEqual(
                verify_v2_migration_checkpoint(failure.checkpoint_path)["status"],
                "VERIFIED",
            )
            with self.assertRaisesRegex(ValueError, "mixed"):
                production_cli_schema_guard(
                    history_db=history_db,
                    controller_db=controller_db,
                )

    def test_source_change_after_checkpoint_is_rejected_before_live_migration(
        self,
    ) -> None:
        for drift in ("history", "config", "code"):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                state, history_db, controller_db, source = self._pair(root)
                dry_run = migration_module._dry_run_checkpoint_migration

                def dry_run_then_mutate(
                    checkpoint: dict, checkpoint_root: Path
                ) -> None:
                    dry_run(checkpoint, checkpoint_root)
                    if drift == "history":
                        connection = sqlite3.connect(history_db)
                        connection.execute(
                            "UPDATE experiments "
                            "SET note = 'changed after checkpoint'"
                        )
                        connection.commit()
                        connection.close()
                    elif drift == "config":
                        config = Path(source["config_path"])
                        config.write_bytes(config.read_bytes() + b" \n")
                    else:
                        (Path(source["repository"]) / "kernel.py").write_text(
                            "def kernel():\n    return 1\n", encoding="utf-8"
                        )

                with (
                    mock.patch.object(
                        migration_module,
                        "_dry_run_checkpoint_migration",
                        side_effect=dry_run_then_mutate,
                    ),
                    self.assertRaises(CoordinatedMigrationError) as raised,
                ):
                    coordinate_v2_to_v3_migration(
                        history_db=history_db,
                        controller_db=controller_db,
                        state_dir=state,
                        checkpoint_root=root / "migration-checkpoints",
                        operation_id=f"source-drift-{drift}",
                        **source,
                    )

                failure = raised.exception
                self.assertEqual(failure.phase, "source-revalidation")
                self.assertFalse(failure.live_migration_started)
                self.assertEqual(_version(history_db), 2)
                self.assertEqual(_version(controller_db), 2)
                self.assertFalse(
                    (state / ".v2-to-v3-migration-intent.json").exists()
                )
                self.assertEqual(
                    verify_v2_migration_checkpoint(failure.checkpoint_path)[
                        "status"
                    ],
                    "VERIFIED",
                )

    def test_checkpoint_tamper_is_detected(self) -> None:
        for target in ("artifact", "config", "code"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                state, history_db, controller_db, source = self._pair(root)
                result = coordinate_v2_to_v3_migration(
                    history_db=history_db,
                    controller_db=controller_db,
                    state_dir=state,
                    checkpoint_root=root / "migration-checkpoints",
                    operation_id=f"tamper-{target}",
                    **source,
                )
                checkpoint = Path(result["checkpoint"])
                if target == "artifact":
                    selected = next((checkpoint / "artifacts").glob("*.py"))
                elif target == "config":
                    selected = checkpoint / "config.json"
                else:
                    selected = checkpoint / "code" / "repository.bundle"
                selected.write_bytes(selected.read_bytes() + b"tampered\n")

                with self.assertRaisesRegex(ValueError, "corrupted"):
                    verify_v2_migration_checkpoint(checkpoint)

    def test_cli_migrates_pair_and_normal_status_cannot_bypass_guard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            state, history_db, controller_db, source = self._pair(root)
            config = SimpleNamespace(
                state_dir=state,
                controller_dir=controller_db.parent,
                repository_dir=source["repository"],
                expected_git_commit=source["expected_git_commit"],
            )
            with (
                mock.patch.object(
                    autorun_cli.ControllerConfig,
                    "load",
                    return_value=config,
                ),
                mock.patch.object(
                    autorun_cli,
                    "ResearchController",
                ) as controller_class,
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()) as error,
            ):
                status_code = autorun_cli.main(
                    ["status", "--config", str(source["config_path"])]
                )
            self.assertEqual(status_code, 2)
            self.assertIn("migrate-v3", error.getvalue())
            controller_class.assert_not_called()
            self.assertEqual(_version(history_db), 2)
            self.assertEqual(_version(controller_db), 2)

            with redirect_stderr(io.StringIO()) as legacy_error:
                legacy_code = legacy_cli.main(
                    ["history", "--state-dir", str(state)]
                )
            self.assertEqual(legacy_code, 2)
            self.assertIn("coordinated", legacy_error.getvalue())
            self.assertEqual(_version(history_db), 2)

            with self.assertRaisesRegex(RuntimeError, "coordinated"):
                autorun_admin._active_run_error(config)  # noqa: SLF001
            self.assertEqual(_version(controller_db), 2)

            with (
                mock.patch.object(
                    autorun_cli.ControllerConfig,
                    "load",
                    return_value=config,
                ),
                redirect_stdout(io.StringIO()) as output,
            ):
                migrate_code = autorun_cli.main(
                    [
                        "migrate-v3",
                        "--config",
                        str(source["config_path"]),
                        "--output",
                        str(root / "migration-checkpoints"),
                        "--operation-id",
                        "cli-proof",
                    ]
                )
            self.assertEqual(migrate_code, 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "SUCCESS")
            self.assertEqual(_version(history_db), 3)
            self.assertEqual(_version(controller_db), 3)

    def test_cli_preserves_structured_full_checkpoint_recovery_evidence(self) -> None:
        config = SimpleNamespace(
            state_dir=Path("/inactive/state"),
            controller_dir=Path("/inactive/controller"),
            repository_dir=Path("/inactive/repository"),
            expected_git_commit="a" * 40,
        )
        failure = CoordinatedMigrationError(
            "injected failure",
            phase="live-controller-migration",
            checkpoint_path=Path("/checkpoint/v2-to-v3-proof"),
            observed_versions={"history": 3, "controller": 2},
            live_migration_started=True,
        )
        with (
            mock.patch.object(
                autorun_cli.ControllerConfig,
                "load",
                return_value=config,
            ),
            mock.patch.object(
                autorun_cli,
                "coordinate_v2_to_v3_migration",
                side_effect=failure,
            ),
            redirect_stderr(io.StringIO()) as error,
        ):
            code = autorun_cli.main(
                [
                    "migrate-v3",
                    "--config",
                    "/fixture/config.json",
                    "--output",
                    "/checkpoint-root",
                ]
            )

        self.assertEqual(code, 2)
        payload = json.loads(error.getvalue())
        self.assertEqual(payload["status"], "FAILED")
        self.assertEqual(payload["rollback_scope"], "FULL_CHECKPOINT_ONLY")
        self.assertEqual(
            payload["checkpoint"], "/checkpoint/v2-to-v3-proof"
        )
        self.assertFalse(payload["single_database_rollback_supported"])


if __name__ == "__main__":
    unittest.main()
