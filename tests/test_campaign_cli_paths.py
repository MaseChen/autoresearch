from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from kernel_research.campaign.cli import main as campaign_main
from kernel_research.campaign.paths import (
    campaign_maintenance_fence,
    conventional_campaign_database,
    conventional_campaign_maintenance_lock,
    validate_production_campaign_database,
    verify_inherited_campaign_maintenance_fence,
)


class CampaignCliPathTests(unittest.TestCase):
    def _root(self, temporary: str) -> Path:
        # macOS spells the temporary root through /var -> /private/var.  The
        # production boundary intentionally requires callers to use the
        # canonical spelling.
        return Path(temporary).resolve()

    def _run(self, argv: list[str]) -> tuple[int, dict, dict]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = campaign_main(argv)
        success = json.loads(stdout.getvalue()) if stdout.getvalue() else {}
        failure = json.loads(stderr.getvalue()) if stderr.getvalue() else {}
        return code, success, failure

    def test_conventional_absolute_path_is_the_only_live_cli_location(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            database = conventional_campaign_database(root)
            code, payload, failure = self._run(
                ["status", "--database", str(database)]
            )

            self.assertEqual((code, failure), (0, {}))
            self.assertEqual(payload["status"], "SUCCESS")
            self.assertEqual(payload["result"]["campaigns"], [])
            self.assertTrue(database.is_file())
            self.assertEqual(
                validate_production_campaign_database(
                    database, runtime_root=root
                ),
                database,
            )

    def test_custom_database_is_rejected_before_every_stateful_command_family(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            custom = root / "campaign" / "custom.sqlite3"
            missing = root / "missing.json"
            artifact_id = "source-sha256-v1:" + "a" * 64
            invocations = {
                "create": [
                    "create",
                    "--database",
                    str(custom),
                    "--campaign-id",
                    "campaign",
                    "--namespace-id",
                    "sha256:" + "b" * 64,
                    "--mode",
                    "DISCOVERY",
                    "--snapshot",
                    str(missing),
                    "--initial-artifact-id",
                    artifact_id,
                    "--initial-policy-snapshot",
                    str(missing),
                ],
                "start": [
                    "start",
                    "--database",
                    str(custom),
                    "--campaign-id",
                    "campaign",
                ],
                "execute": [
                    "child",
                    "execute",
                    "--database",
                    str(custom),
                    "--config",
                    str(missing),
                    "--campaign-id",
                    "campaign",
                    "--controller-run-id",
                    "run",
                    "--proposer-profile",
                    str(missing),
                ],
                "resume": [
                    "resume",
                    "--database",
                    str(custom),
                    "--campaign-id",
                    "campaign",
                ],
                "oj": [
                    "oj",
                    "show",
                    "--database",
                    str(custom),
                    "--nomination-id",
                    "nomination",
                ],
                "soak": [
                    "soak",
                    "status",
                    "--database",
                    str(custom),
                    "--config",
                    str(missing),
                    "--gate-id",
                    "gate",
                ],
            }

            for family, argv in invocations.items():
                with self.subTest(family=family):
                    code, payload, failure = self._run(argv)
                    self.assertEqual((code, payload), (2, {}))
                    self.assertEqual(failure["error_type"], "ValueError")
                    self.assertIn("conventional absolute path", failure["error"])
            self.assertFalse(custom.exists())

    def test_root_direct_relative_and_symlink_spellings_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            with self.assertRaisesRegex(ValueError, "conventional absolute path"):
                validate_production_campaign_database(
                    root / "campaign.sqlite3"
                )
            with self.assertRaisesRegex(ValueError, "absolute path"):
                validate_production_campaign_database(
                    Path("campaign") / "campaign.sqlite3"
                )

            target = root / "objects" / "restored.sqlite3"
            target.parent.mkdir()
            target.write_bytes(b"not opened by the validator")
            database = conventional_campaign_database(root)
            database.parent.mkdir()
            database.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "must be canonical"):
                validate_production_campaign_database(database)

    def test_explicit_runtime_root_binding_cannot_be_redirected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            other = root / "other-runtime"
            other.mkdir()
            database = conventional_campaign_database(root)
            with self.assertRaisesRegex(ValueError, "supplied runtime_root"):
                validate_production_campaign_database(
                    database, runtime_root=other
                )

    def test_filesystem_root_and_unresolvable_paths_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "filesystem root"):
            conventional_campaign_database(Path("/"))

        with patch.object(
            Path, "resolve", side_effect=RuntimeError("symlink loop")
        ):
            with self.assertRaisesRegex(ValueError, "resolved canonically"):
                conventional_campaign_database(Path("/runtime"))

    def test_symlinked_runtime_root_is_not_an_alternate_spelling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = self._root(temporary)
            actual = parent / "actual-runtime"
            actual.mkdir()
            alias = parent / "runtime-alias"
            alias.symlink_to(actual, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "must be canonical"):
                conventional_campaign_database(alias)

    def test_maintenance_fence_is_fixed_nonaliased_and_inheritable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._root(temporary)
            lock = conventional_campaign_maintenance_lock(root)
            with campaign_maintenance_fence(root) as descriptor:
                self.assertEqual(
                    verify_inherited_campaign_maintenance_fence(
                        root, descriptor
                    ),
                    descriptor,
                )
                self.assertEqual(
                    lock.read_text(encoding="ascii").strip(), str(os.getpid())
                )
            unlocked = os.open(lock, os.O_RDWR)
            try:
                with self.assertRaisesRegex(ValueError, "is not locked"):
                    verify_inherited_campaign_maintenance_fence(root, unlocked)
            finally:
                os.close(unlocked)

            alias_root = root / "alias-runtime"
            target = alias_root / "target.lock"
            target.parent.mkdir()
            target.write_text("", encoding="ascii")
            alias_lock = conventional_campaign_maintenance_lock(alias_root)
            alias_lock.parent.mkdir(exist_ok=True)
            alias_lock.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "canonical"):
                with campaign_maintenance_fence(alias_root):
                    pass

            hardlink_root = root / "hardlink-runtime"
            hardlink_target = hardlink_root / "target.lock"
            hardlink_target.parent.mkdir()
            hardlink_target.write_text("", encoding="ascii")
            hardlink = conventional_campaign_maintenance_lock(hardlink_root)
            hardlink.parent.mkdir(exist_ok=True)
            os.link(hardlink_target, hardlink)
            with self.assertRaisesRegex(ValueError, "filesystem aliases"):
                with campaign_maintenance_fence(hardlink_root):
                    pass


if __name__ == "__main__":
    unittest.main()
