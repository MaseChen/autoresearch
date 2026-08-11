from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from kernel_research.autorun import admin
from kernel_research.autorun.errors import ControlledRuntimeError
from kernel_research.campaign import BudgetAmount, CampaignStore
from kernel_research.campaign import cli as campaign_cli
from kernel_research.campaign.paths import conventional_campaign_database


class AdminCampaignGuardTests(unittest.TestCase):
    def _campaign(self, root: Path) -> CampaignStore:
        store = CampaignStore(root / "campaign" / "campaign.sqlite3")
        store.create_campaign(
            campaign_id="active-campaign",
            namespace_id="sha256:" + "a" * 64,
            mode="DISCOVERY",
            snapshot={},
            budget_limit=BudgetAmount(candidates=1),
            initial_artifact_id="source-sha256-v1:" + "b" * 64,
            initial_policy_snapshot={},
        )
        return store

    def test_guard_treats_created_running_and_paused_as_active(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with self._campaign(root) as store:
                self.assertIn("CREATED", admin._active_campaign_error(root) or "")
                store.start_campaign("active-campaign")
                store.pause_campaign(
                    "active-campaign", status="PAUSED_OPERATOR", reason="manual"
                )
                self.assertIn(
                    "PAUSED_OPERATOR", admin._active_campaign_error(root) or ""
                )
                store.finish_campaign(
                    "active-campaign", cancelled=True, reason="finished"
                )
                self.assertIsNone(admin._active_campaign_error(root))

    def test_update_and_adopt_fail_before_repository_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest = mock.Mock(runtime_root=root)
            with self._campaign(root):
                with mock.patch.object(admin, "_repo_state") as repository_state:
                    with self.assertRaisesRegex(
                        ControlledRuntimeError, "campaigns are active"
                    ):
                        admin.update(manifest, doctor=False)
                    repository_state.assert_not_called()
                    with self.assertRaisesRegex(
                        ControlledRuntimeError, "campaigns are active"
                    ):
                        admin.adopt_baseline(
                            manifest, candidate_hash="c" * 64, doctor=False
                        )
                    repository_state.assert_not_called()
                    with self.assertRaisesRegex(
                        ControlledRuntimeError, "campaigns are active"
                    ):
                        admin.sync(manifest)
                    repository_state.assert_not_called()
                    with self.assertRaisesRegex(
                        ControlledRuntimeError, "campaigns are active"
                    ):
                        admin._post_update(manifest, doctor=False)
                    repository_state.assert_not_called()

    def test_guard_fails_closed_for_symlinked_or_nonregular_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()

            broken_root = base / "broken"
            (broken_root / "campaign").mkdir(parents=True)
            (broken_root / "campaign" / "campaign.sqlite3").symlink_to(
                broken_root / "missing.sqlite3"
            )
            with self.assertRaisesRegex(
                ControlledRuntimeError, "non-symlink boundary"
            ):
                admin._active_campaign_error(broken_root)

            linked_root = base / "linked"
            linked_root.mkdir()
            real_campaign = base / "real-campaign"
            real_campaign.mkdir()
            (linked_root / "campaign").symlink_to(
                real_campaign, target_is_directory=True
            )
            with self.assertRaisesRegex(
                ControlledRuntimeError, "non-symlink boundary"
            ):
                admin._active_campaign_error(linked_root)

            nonregular_root = base / "nonregular"
            (nonregular_root / "campaign" / "campaign.sqlite3").mkdir(
                parents=True
            )
            with self.assertRaisesRegex(
                ControlledRuntimeError, "not a regular file"
            ):
                admin._active_campaign_error(nonregular_root)

    def test_admin_guard_blocks_campaign_create_until_publication_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            database = conventional_campaign_database(root)
            snapshot = root / "snapshot.json"
            policy = root / "policy.json"
            snapshot.write_text("{}", encoding="utf-8")
            policy.write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                database=str(database),
                campaign_id="created-after-publication",
                namespace_id="sha256:" + "a" * 64,
                mode="DISCOVERY",
                snapshot=str(snapshot),
                initial_baseline_ref=None,
                initial_artifact_id="source-sha256-v1:" + "b" * 64,
                initial_policy_snapshot=str(policy),
                max_candidates=1,
                max_wall_ms=0,
                max_gpu_ms=0,
                max_tokens=0,
                max_cost_microusd=0,
                allow_staged_lineage=False,
            )
            manifest = mock.Mock(runtime_root=root)
            guard_entered = threading.Event()
            permit_publish = threading.Event()
            create_done = threading.Event()
            order: list[str] = []
            failures: list[BaseException] = []

            def publish() -> None:
                try:
                    with admin._deployment_mutation_guard(manifest):
                        guard_entered.set()
                        if not permit_publish.wait(5):
                            raise AssertionError("test publication was not released")
                        admin._require_no_active_campaign(root)
                        order.append("published")
                except BaseException as exc:  # pragma: no cover - diagnostic
                    failures.append(exc)

            def create() -> None:
                try:
                    campaign_cli._create(args)
                    order.append("created")
                except BaseException as exc:  # pragma: no cover - diagnostic
                    failures.append(exc)
                finally:
                    create_done.set()

            publisher = threading.Thread(target=publish)
            creator = threading.Thread(target=create)
            with mock.patch.object(campaign_cli, "_success", return_value=0):
                publisher.start()
                self.assertTrue(guard_entered.wait(5))
                creator.start()
                try:
                    self.assertFalse(create_done.wait(0.15))
                finally:
                    permit_publish.set()
                publisher.join(5)
                creator.join(5)
            self.assertFalse(publisher.is_alive())
            self.assertFalse(creator.is_alive())
            self.assertEqual(failures, [])
            self.assertEqual(order, ["published", "created"])
            with CampaignStore(database) as store:
                self.assertEqual(
                    store.get_campaign("created-after-publication")["status"],
                    "CREATED",
                )

    def test_campaign_start_commits_before_admin_can_recheck(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            database = conventional_campaign_database(root)
            with self._campaign(root):
                pass
            args = argparse.Namespace(
                database=str(database), campaign_id="active-campaign"
            )
            manifest = mock.Mock(runtime_root=root)
            start_committed = threading.Event()
            release_start = threading.Event()
            admin_done = threading.Event()
            admin_entered = threading.Event()
            start_failures: list[BaseException] = []
            admin_failures: list[BaseException] = []
            original_start = CampaignStore.start_campaign

            def held_start(store: CampaignStore, campaign_id: str):
                result = original_start(store, campaign_id)
                start_committed.set()
                if not release_start.wait(5):
                    raise AssertionError("test Campaign start was not released")
                return result

            def start() -> None:
                try:
                    campaign_cli._start(args)
                except BaseException as exc:  # pragma: no cover - diagnostic
                    start_failures.append(exc)

            def administer() -> None:
                try:
                    with admin._deployment_mutation_guard(manifest):
                        admin_entered.set()
                except BaseException as exc:
                    admin_failures.append(exc)
                finally:
                    admin_done.set()

            starter = threading.Thread(target=start)
            administrator = threading.Thread(target=administer)
            with (
                mock.patch.object(
                    CampaignStore, "start_campaign", new=held_start
                ),
                mock.patch.object(campaign_cli, "_success", return_value=0),
            ):
                starter.start()
                self.assertTrue(start_committed.wait(5))
                administrator.start()
                try:
                    self.assertFalse(admin_done.wait(0.15))
                finally:
                    release_start.set()
                starter.join(5)
                administrator.join(5)
            self.assertFalse(starter.is_alive())
            self.assertFalse(administrator.is_alive())
            self.assertEqual(start_failures, [])
            self.assertFalse(admin_entered.is_set())
            self.assertEqual(len(admin_failures), 1)
            self.assertIsInstance(admin_failures[0], ControlledRuntimeError)
            self.assertIn("campaigns are active", str(admin_failures[0]))

    def test_every_admin_publisher_enters_the_shared_guard(self) -> None:
        manifest = mock.Mock(runtime_root=Path("/tmp/runtime"))
        blocked = ControlledRuntimeError("maintenance sentinel")
        publishers = (
            lambda: admin.sync(manifest),
            lambda: admin.update(manifest, doctor=False),
            lambda: admin.adopt_baseline(
                manifest, candidate_hash="c" * 64, doctor=False
            ),
            lambda: admin._post_update(manifest, doctor=False),
        )
        for publish in publishers:
            with self.subTest(publisher=publish):
                with (
                    mock.patch.object(
                        admin,
                        "_deployment_mutation_guard",
                        side_effect=blocked,
                    ) as guard,
                    self.assertRaisesRegex(
                        ControlledRuntimeError, "maintenance sentinel"
                    ),
                ):
                    publish()
                guard.assert_called_once_with(manifest)


if __name__ == "__main__":
    unittest.main()
