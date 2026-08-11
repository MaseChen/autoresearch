from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest

from kernel_research.campaign.cli import build_parser
from kernel_research.campaign.soak import (
    SOAK_MAX_HEARTBEAT_GAP_SECONDS,
    SOAK_STAGES,
    SoakGate,
)
from kernel_research.campaign.soak_collector import (
    COUNT_FIELDS,
    SoakObservation,
)
from kernel_research.campaign.store import CampaignStore
from kernel_research.platform.canonical import canonical_sha256


HOUR = 60 * 60


class FakeClock:
    def __init__(self, epoch: float = 1_000.0) -> None:
        self.epoch = epoch

    def __call__(self) -> float:
        return self.epoch

    def advance(self, seconds: float) -> None:
        self.epoch += seconds


class FakeTrustedCollector:
    def __init__(self) -> None:
        self.invariant = {"code": "commit-a", "profiles": "registry-a"}
        self.counts = {field: 0 for field in COUNT_FIELDS}
        self.unavailable_reasons: tuple[str, ...] = ()
        self.emit_activity = True
        self.emit_terminal_activity = True
        self.emit_staged_activity = True
        self.source_cursor: object = {}

    def collect(
        self, *, interval_start_epoch: float, interval_end_epoch: float
    ) -> SoakObservation:
        status = "UNAVAILABLE" if self.unavailable_reasons else "AVAILABLE"
        marker = int(interval_end_epoch)
        run_id = f"run-{marker}"
        primary = f"primary-{marker}"
        confirmation = f"confirmation-{marker}"
        return SoakObservation.unsafe_create_for_tests(
            status=status,
            invariant_snapshot=self.invariant,
            interval_start_epoch=interval_start_epoch,
            interval_end_epoch=interval_end_epoch,
            counts=self.counts,
            sources={
                "campaign": {
                    "status": status,
                    "cursor": self.source_cursor,
                    "summary": {
                        "activity": {
                            "terminal_children": ([
                                {
                                    "child_id": marker + 1,
                                    "controller_run_id": run_id,
                                    "status": "BUDGET_EXHAUSTED",
                                }
                            ] if (
                                self.emit_activity
                                and self.emit_terminal_activity
                            ) else []),
                            "staged_revisions": ([
                                {
                                    "revision_id": f"revision-{marker}",
                                    "primary_experiment_uid": primary,
                                    "confirmation_experiment_uid": confirmation,
                                }
                            ] if (
                                self.emit_activity
                                and self.emit_staged_activity
                            ) else []),
                        }
                    },
                },
                "controller": {
                    "status": status,
                    "cursor": self.source_cursor,
                    "summary": {
                        "activity": {
                            "controller_run_ids": (
                                [run_id] if self.emit_activity else []
                            ),
                            "confirmed_experiment_uids": ([
                                primary,
                                confirmation,
                            ] if self.emit_activity else []),
                        }
                    },
                },
                "docker": {"status": status, "cursor": self.source_cursor},
            },
            unavailable_reasons=self.unavailable_reasons,
        )


class CampaignSoakTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "campaign.sqlite3"
        self.store = CampaignStore(self.path)
        # Long qualification tests execute thousands of tiny transactions.
        # Durability itself is covered by production pragmas and restart tests.
        self.store.connection.execute("PRAGMA synchronous = OFF")
        self.clock = FakeClock()
        self.collector = FakeTrustedCollector()
        self.gate = SoakGate(
            self.store,
            gate_id="release-v2",
            collector=self.collector,
            clock=self.clock,
        )

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def _complete_active(self) -> None:
        active = self.store.get_active_soak_generation("release-v2")
        assert active is not None
        remaining = int(
            float(active["required_seconds"])
            - float(active["accumulated_seconds"])
        )
        result = None
        while remaining:
            step = min(SOAK_MAX_HEARTBEAT_GAP_SECONDS, remaining)
            self.clock.advance(step)
            result = self.gate.heartbeat()
            remaining -= step
        assert result is not None
        self.assertEqual(result["outcome"], "COMPLETED")

    def test_fast_forward_cannot_complete_but_continuous_clean_can(self) -> None:
        self.assertEqual(
            [(stage.stage_id, stage.required_seconds) for stage in SOAK_STAGES],
            [
                ("MVP_24H", 24 * HOUR),
                ("STAGED_LINEAGE_72H", 72 * HOUR),
                ("LONG_CAMPAIGN_168H", 168 * HOUR),
            ],
        )
        started = self.gate.start()
        self.assertEqual(started["outcome"], "STARTED")
        self.assertEqual(started["generation"]["accumulated_seconds"], 0)

        self.clock.advance(24 * HOUR)
        jumped = self.gate.heartbeat()
        self.assertEqual(jumped["outcome"], "VIOLATION")
        self.assertEqual(jumped["violation"]["reason_code"], "HEARTBEAT_GAP")
        self.assertEqual(jumped["generation"]["accumulated_seconds"], 0)

        self._complete_active()
        digest = self.collector.collect(
            interval_start_epoch=self.clock(), interval_end_epoch=self.clock()
        ).invariant_snapshot_digest
        self.assertEqual(
            self.store.completed_soak_stages("release-v2", digest),
            ("MVP_24H",),
        )

    def test_gate_inputs_clock_and_stage_order_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "gate_id"):
            SoakGate(
                self.store,
                gate_id="",
                collector=self.collector,
                clock=self.clock,
            )
        with self.assertRaisesRegex(ValueError, "clock"):
            SoakGate(
                self.store,
                gate_id="gate",
                collector=self.collector,
                clock=None,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(TypeError, "collector"):
            SoakGate(
                self.store,
                gate_id="gate",
                collector=object(),  # type: ignore[arg-type]
                clock=self.clock,
            )
        for value in (True, -1, math.nan, math.inf):
            invalid_clock = SoakGate(
                self.store,
                gate_id="clock-gate",
                collector=self.collector,
                clock=lambda value=value: value,
            )
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "finite non-negative"
            ):
                invalid_clock.status()

        with self.assertRaisesRegex(ValueError, "no active generation"):
            self.gate.heartbeat()
        with self.assertRaisesRegex(ValueError, "must run in order"):
            self.gate.start(stage="STAGED_LINEAGE_72H")

    def test_replay_active_restart_and_backwards_clock_are_safe(self) -> None:
        started = self.gate.start()
        self.assertEqual(started["outcome"], "STARTED")
        self.assertEqual(self.gate.heartbeat()["outcome"], "REPLAYED")
        self.clock.advance(1)
        with self.assertRaisesRegex(ValueError, "already active"):
            self.gate.start()

        self.clock.advance(-2)
        backwards = self.gate.heartbeat()
        self.assertEqual(backwards["outcome"], "VIOLATION")
        self.assertEqual(
            backwards["violation"]["reason_code"], "HEARTBEAT_GAP"
        )

    def test_unavailable_start_and_invalid_cursor_earn_no_credit(self) -> None:
        self.collector.unavailable_reasons = ("CAMPAIGN_DB_UNAVAILABLE",)
        unavailable = self.gate.start()
        self.assertEqual(unavailable["outcome"], "VIOLATION")
        self.assertEqual(
            unavailable["violation"]["reason_code"], "SOURCE_UNAVAILABLE"
        )
        self.assertEqual(unavailable["generation"]["accumulated_seconds"], 0)

        invalid_collector = FakeTrustedCollector()
        invalid_collector.source_cursor = {"offset": math.nan}
        invalid_gate = SoakGate(
            self.store,
            gate_id="invalid-cursor",
            collector=invalid_collector,
            clock=self.clock,
        )
        with self.assertRaisesRegex(ValueError, "strict JSON"):
            invalid_gate.start()

    def test_empty_runtime_reaches_duration_but_cannot_pass_activity_gate(self) -> None:
        self.collector.emit_activity = False
        self.gate.start()
        last = None
        for _ in range((24 * HOUR) // SOAK_MAX_HEARTBEAT_GAP_SECONDS):
            self.clock.advance(SOAK_MAX_HEARTBEAT_GAP_SECONDS)
            last = self.gate.heartbeat()
        assert last is not None
        self.assertEqual(last["outcome"], "PENDING_ACTIVITY")
        self.assertTrue(last["duration_satisfied"])
        self.assertFalse(last["activity_satisfied"])
        self.assertEqual(last["generation"]["status"], "ACTIVE")
        self.assertEqual(last["generation"]["accumulated_seconds"], 24 * HOUR)
        self.assertFalse(self.gate.profiling_allowed())

        self.collector.emit_activity = True
        self.clock.advance(1)
        completed = self.gate.heartbeat()
        self.assertEqual(completed["outcome"], "COMPLETED")

    def test_lineage_stage_requires_new_confirmed_staged_revision(self) -> None:
        self.gate.start(stage="MVP_24H")
        self._complete_active()
        self.collector.emit_staged_activity = False
        self.gate.start(stage="STAGED_LINEAGE_72H")
        last = None
        for _ in range((72 * HOUR) // SOAK_MAX_HEARTBEAT_GAP_SECONDS):
            self.clock.advance(SOAK_MAX_HEARTBEAT_GAP_SECONDS)
            last = self.gate.heartbeat()
        assert last is not None
        self.assertEqual(last["outcome"], "PENDING_ACTIVITY")
        self.assertFalse(last["activity_satisfied"])

        self.collector.emit_staged_activity = True
        self.clock.advance(1)
        self.assertEqual(self.gate.heartbeat()["outcome"], "COMPLETED")

    def test_unavailable_gap_and_each_nonzero_counter_reset_atomically(self) -> None:
        self.gate.start()
        self.clock.advance(300)
        self.gate.heartbeat()
        self.assertEqual(
            self.store.get_active_soak_generation("release-v2")[
                "accumulated_seconds"
            ],
            300,
        )

        self.clock.advance(1)
        self.collector.unavailable_reasons = ("DOCKER_INVENTORY_UNAVAILABLE",)
        unavailable = self.gate.heartbeat()
        self.assertEqual(unavailable["outcome"], "VIOLATION")
        self.assertEqual(
            unavailable["violation"]["reason_code"], "SOURCE_UNAVAILABLE"
        )
        self.assertEqual(unavailable["generation"]["accumulated_seconds"], 0)
        self.collector.unavailable_reasons = ()
        self.clock.advance(1)
        reanchored = self.gate.heartbeat()
        self.assertEqual(
            reanchored["violation"]["reason_code"],
            "SOURCE_RECOVERED_REANCHOR",
        )

        for field in COUNT_FIELDS:
            self.clock.advance(1)
            self.collector.counts[field] = 1
            result = self.gate.heartbeat()
            self.assertEqual(result["outcome"], "VIOLATION")
            self.assertEqual(
                result["violation"]["reason_code"], "INVARIANT_VIOLATION"
            )
            self.assertEqual(result["violation"][field], 1)
            self.assertEqual(result["generation"]["accumulated_seconds"], 0)
            self.collector.counts[field] = 0

        self.clock.advance(SOAK_MAX_HEARTBEAT_GAP_SECONDS + 1)
        gap = self.gate.heartbeat()
        self.assertEqual(gap["violation"]["reason_code"], "HEARTBEAT_GAP")
        self.assertEqual(len(self.store.list_soak_violations("release-v2")), 7)

    def test_invariant_change_returns_to_mvp_and_invalidates_profiling(self) -> None:
        for stage in SOAK_STAGES:
            self.gate.start(stage=stage.stage_id)
            self._complete_active()
        self.assertTrue(self.gate.profiling_allowed())
        with self.assertRaisesRegex(ValueError, "already complete"):
            self.gate.start()
        self.clock.advance(1)
        monitored = self.gate.heartbeat()
        self.assertEqual(monitored["outcome"], "QUALIFIED_MONITORED")
        self.collector.counts["orphan_container_count"] = 1
        self.assertFalse(self.gate.profiling_allowed())
        self.assertFalse(self.gate.status()["profiling_allowed"])
        self.clock.advance(1)
        post_completion_violation = self.gate.heartbeat()
        self.assertEqual(post_completion_violation["outcome"], "VIOLATION")
        self.assertEqual(
            post_completion_violation["generation"]["stage"], "MVP_24H"
        )
        self.collector.counts["orphan_container_count"] = 0
        self.assertFalse(self.gate.profiling_allowed())
        old_status = self.gate.status()
        self.assertEqual(old_status["completed_stages"], [])

        self.collector.invariant = {
            "code": "commit-b",
            "profiles": "registry-a",
        }
        changed = self.gate.status()
        self.assertEqual(changed["completed_stages"], [])
        self.assertEqual(changed["pending_stage"], "MVP_24H")
        self.assertFalse(changed["profiling_allowed"])
        restarted = self.gate.start()
        self.assertEqual(restarted["generation"]["stage"], "MVP_24H")
        self.assertEqual(restarted["generation"]["accumulated_seconds"], 0)

        self.clock.advance(1)
        self.collector.invariant = {
            "code": "commit-c",
            "profiles": "registry-a",
        }
        active_change = self.gate.heartbeat()
        self.assertEqual(active_change["outcome"], "VIOLATION")
        self.assertEqual(
            active_change["violation"]["reason_code"], "INVARIANT_CHANGED"
        )
        self.assertEqual(active_change["generation"]["stage"], "MVP_24H")

    def test_observations_and_violations_are_immutable_and_raw_input_is_rejected(self) -> None:
        self.gate.start()
        self.clock.advance(1)
        self.collector.counts["budget_leak_count"] = 1
        result = self.gate.heartbeat()
        evidence_id = result["observation"]["evidence_id"]
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.connection.execute(
                "UPDATE soak_observations SET status = 'AVAILABLE' "
                "WHERE gate_id = ? AND evidence_id = ?",
                ("release-v2", evidence_id),
            )
        self.store.connection.rollback()
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.connection.execute(
                "DELETE FROM soak_violations WHERE id = ?",
                (result["violation"]["id"],),
            )
        self.store.connection.rollback()
        with self.assertRaisesRegex(TypeError, "trusted SoakObservation"):
            self.store.record_soak_heartbeat(
                gate_id="release-v2", observation={}  # type: ignore[arg-type]
            )

    def test_progress_survives_store_restart(self) -> None:
        started = self.gate.start()
        for _ in range(3):
            self.clock.advance(300)
            self.gate.heartbeat()
        self.store.close()
        self.store = CampaignStore(self.path)
        self.gate = SoakGate(
            self.store,
            gate_id="release-v2",
            collector=self.collector,
            clock=self.clock,
        )
        active = self.store.get_active_soak_generation("release-v2")
        self.assertEqual(active["id"], started["generation"]["id"])
        self.assertEqual(active["accumulated_seconds"], 900)

    def test_existing_campaign_v1_adds_trusted_soak_schema_without_version_bump(self) -> None:
        self.store.close()
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("DROP TABLE soak_observations")
        connection.execute("DROP TABLE soak_violations")
        connection.execute("DROP TABLE soak_generations")
        connection.executescript(
            """
            CREATE TABLE soak_generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, gate_id TEXT NOT NULL,
                stage TEXT NOT NULL, generation_index INTEGER NOT NULL,
                status TEXT NOT NULL, required_seconds REAL NOT NULL,
                invariant_snapshot_digest TEXT NOT NULL,
                invariant_snapshot_json TEXT NOT NULL,
                started_epoch REAL NOT NULL, last_heartbeat_epoch REAL NOT NULL,
                accumulated_seconds REAL NOT NULL, completed_epoch REAL,
                violated_epoch REAL
            );
            CREATE TABLE soak_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, gate_id TEXT NOT NULL,
                stage TEXT NOT NULL, generation_id INTEGER NOT NULL,
                generation_index INTEGER NOT NULL,
                invariant_snapshot_digest TEXT NOT NULL,
                observed_epoch REAL NOT NULL, lease_overlap_count INTEGER NOT NULL,
                orphan_container_count INTEGER NOT NULL,
                hard_fault_retry_count INTEGER NOT NULL,
                budget_leak_count INTEGER NOT NULL, evidence_json TEXT NOT NULL
            );
            """
        )
        connection.commit()
        self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 1)
        connection.close()

        self.store = CampaignStore(self.path)
        generation_columns = {
            row[1]
            for row in self.store.connection.execute(
                "PRAGMA table_info(soak_generations)"
            )
        }
        violation_columns = {
            row[1]
            for row in self.store.connection.execute(
                "PRAGMA table_info(soak_violations)"
            )
        }
        self.assertTrue(
            {"initial_evidence_id", "last_evidence_id"}.issubset(
                generation_columns
            )
        )
        self.assertTrue(
            {"evidence_id", "reason_code"}.issubset(violation_columns)
        )
        self.assertEqual(
            self.store.connection.execute("PRAGMA user_version").fetchone()[0],
            1,
        )

    def test_legacy_untrusted_active_credit_is_discarded(self) -> None:
        snapshot = dict(self.collector.invariant)
        digest = canonical_sha256(snapshot)
        self.store.connection.execute(
            """
            INSERT INTO soak_generations(
                gate_id, stage, generation_index, status, required_seconds,
                invariant_snapshot_digest, invariant_snapshot_json,
                started_epoch, last_heartbeat_epoch, accumulated_seconds,
                initial_evidence_id, last_evidence_id
            ) VALUES (
                'legacy-gate', 'LONG_CAMPAIGN_168H', 1, 'ACTIVE', 604800,
                ?, ?, 1, ?, 604799, NULL, NULL
            )
            """,
            (digest, json.dumps(snapshot, sort_keys=True), self.clock()),
        )
        self.store.connection.commit()
        legacy_gate = SoakGate(
            self.store,
            gate_id="legacy-gate",
            collector=self.collector,
            clock=self.clock,
        )
        result = legacy_gate.start(stage="LONG_CAMPAIGN_168H")
        self.assertEqual(result["outcome"], "VIOLATION")
        self.assertEqual(
            result["violation"]["reason_code"], "UNTRUSTED_LEGACY_STATE"
        )
        self.assertEqual(result["generation"]["stage"], "MVP_24H")
        self.assertEqual(result["generation"]["accumulated_seconds"], 0)

    def test_cli_has_no_path_for_raw_counts_or_evidence(self) -> None:
        parser = build_parser()
        valid = parser.parse_args(
            [
                "soak", "heartbeat", "--database", str(self.path),
                "--config", "/trusted/config.json", "--gate-id", "gate",
            ]
        )
        for field in (*COUNT_FIELDS, "evidence", "invariant_snapshot"):
            self.assertFalse(hasattr(valid, field))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "soak", "heartbeat", "--database", str(self.path),
                    "--config", "/trusted/config.json", "--gate-id", "gate",
                    "--budget-leak-count", "0",
                ]
            )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "soak", "start", "--database", str(self.path),
                    "--config", "/trusted/config.json", "--gate-id", "gate",
                    "--invariant-snapshot", "/untrusted.json",
                ]
            )


if __name__ == "__main__":
    unittest.main()
