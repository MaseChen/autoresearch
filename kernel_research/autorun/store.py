"""Durable controller state, separate from evaluator history."""

from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping

from .states import (
    IterationStatus,
    RunStatus,
    Stage,
    TERMINAL_RUN_STATUSES,
    validate_stage_transition,
)
from .errors import ControlledRuntimeError

SCHEMA_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _state_value(value: Any) -> str:
    return str(value.value) if hasattr(value, "value") else str(value)


class ControllerStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._initialize()

    def _initialize(self) -> None:
        version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise ControlledRuntimeError(
                f"controller schema {version} is newer than supported "
                f"version {SCHEMA_VERSION}"
            )
        if version == 0:
            self.connection.executescript(
                """
            BEGIN IMMEDIATE;
            CREATE TABLE runs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deadline_epoch REAL NOT NULL,
                valid_candidates INTEGER NOT NULL DEFAULT 0,
                consecutive_failures INTEGER NOT NULL DEFAULT 0,
                stop_requested INTEGER NOT NULL DEFAULT 0
                    CHECK (stop_requested IN (0, 1)),
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
            PRAGMA user_version = 1;
            COMMIT;
            """
            )
            version = 1
        if version == 1:
            self._migrate_v1_to_v2()

    def _migrate_v1_to_v2(self) -> None:
        allowed_run_statuses = tuple(status.value for status in RunStatus)
        allowed_iteration_statuses = tuple(
            status.value for status in IterationStatus
        )
        allowed_stages = tuple(stage.value for stage in Stage)
        invalid_runs = self.connection.execute(
            "SELECT DISTINCT status FROM runs "
            f"WHERE status NOT IN ({','.join('?' for _ in allowed_run_statuses)})",
            allowed_run_statuses,
        ).fetchall()
        invalid_iteration_statuses = self.connection.execute(
            "SELECT DISTINCT status FROM iterations "
            f"WHERE status NOT IN "
            f"({','.join('?' for _ in allowed_iteration_statuses)})",
            allowed_iteration_statuses,
        ).fetchall()
        invalid_stages = self.connection.execute(
            "SELECT DISTINCT stage FROM iterations "
            f"WHERE stage NOT IN ({','.join('?' for _ in allowed_stages)})",
            allowed_stages,
        ).fetchall()
        invalid_pairs = self.connection.execute(
            """
            SELECT DISTINCT status, stage FROM iterations
            WHERE (status = 'COMPLETED' AND stage <> 'DONE')
               OR (status <> 'COMPLETED' AND stage = 'DONE')
            """
        ).fetchall()
        invalid = [
            *(f"run status {row[0]!r}" for row in invalid_runs),
            *(
                f"iteration status {row[0]!r}"
                for row in invalid_iteration_statuses
            ),
            *(f"iteration stage {row[0]!r}" for row in invalid_stages),
            *(
                f"iteration state pair {row[0]!r}/{row[1]!r}"
                for row in invalid_pairs
            ),
        ]
        if invalid:
            raise ControlledRuntimeError(
                "controller schema v1 contains unknown state values: "
                + ", ".join(invalid)
            )
        run_values = ", ".join(f"'{value}'" for value in allowed_run_statuses)
        iteration_values = ", ".join(
            f"'{value}'" for value in allowed_iteration_statuses
        )
        stage_values = ", ".join(f"'{value}'" for value in allowed_stages)
        self.connection.executescript(
            f"""
            BEGIN IMMEDIATE;
            CREATE TRIGGER runs_status_insert_guard
            BEFORE INSERT ON runs
            WHEN NEW.status NOT IN ({run_values})
            BEGIN
                SELECT RAISE(ABORT, 'invalid run status');
            END;
            CREATE TRIGGER runs_status_update_guard
            BEFORE UPDATE OF status ON runs
            WHEN NEW.status NOT IN ({run_values})
            BEGIN
                SELECT RAISE(ABORT, 'invalid run status');
            END;
            CREATE TRIGGER iterations_state_insert_guard
            BEFORE INSERT ON iterations
            WHEN NEW.status NOT IN ({iteration_values})
              OR NEW.stage NOT IN ({stage_values})
              OR (NEW.status = 'COMPLETED' AND NEW.stage <> 'DONE')
              OR (NEW.status <> 'COMPLETED' AND NEW.stage = 'DONE')
            BEGIN
                SELECT RAISE(ABORT, 'invalid iteration state');
            END;
            CREATE TRIGGER iterations_state_update_guard
            BEFORE UPDATE OF status, stage ON iterations
            WHEN NEW.status NOT IN ({iteration_values})
              OR NEW.stage NOT IN ({stage_values})
              OR (NEW.status = 'COMPLETED' AND NEW.stage <> 'DONE')
              OR (NEW.status <> 'COMPLETED' AND NEW.stage = 'DONE')
            BEGIN
                SELECT RAISE(ABORT, 'invalid iteration state');
            END;
            CREATE INDEX IF NOT EXISTS events_run_iteration_idx
                ON events(run_id, iteration_id, id);
            PRAGMA user_version = 2;
            COMMIT;
            """
        )

    @contextmanager
    def _transaction(self):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def _add_event(
        self,
        run_id: str,
        event: str,
        details: Mapping[str, Any],
        *,
        iteration_id: int | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO events (
                run_id, iteration_id, created_at, event, details_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, iteration_id, utc_now(), event, _json(details)),
        )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "ControllerStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def create_run(
        self,
        *,
        run_id: str,
        deadline_epoch: float,
        config: Mapping[str, Any],
        initial_best_hash: str,
        preflight: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self._transaction():
            self.connection.execute(
                """
                INSERT INTO runs (
                    id, status, created_at, updated_at, deadline_epoch,
                    config_json, preflight_json, initial_best_hash
                ) VALUES (?, 'RUNNING', ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    now,
                    now,
                    deadline_epoch,
                    _json(config),
                    _json(preflight or {}),
                    initial_best_hash,
                ),
            )
            self._add_event(
                run_id, "RUN_CREATED", {"deadline_epoch": deadline_epoch}
            )
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown run id: {run_id}")
        result = dict(row)
        result["stop_requested"] = bool(result["stop_requested"])
        result["config"] = json.loads(result.pop("config_json"))
        result["preflight"] = json.loads(result.pop("preflight_json"))
        return result

    def latest_run(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT id FROM runs ORDER BY created_at DESC, id DESC LIMIT 1"
        ).fetchone()
        return None if row is None else self.get_run(str(row["id"]))

    def update_run(self, run_id: str, **values: Any) -> dict[str, Any]:
        allowed = {
            "status",
            "valid_candidates",
            "consecutive_failures",
            "stop_requested",
            "stop_reason",
            "final_best_hash",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unsupported run fields: {sorted(unknown)}")
        if not values:
            return self.get_run(run_id)
        if "status" in values:
            status_value = _state_value(values["status"])
            RunStatus(status_value)
            current = self.get_run(run_id)
            if (
                current["status"] in TERMINAL_RUN_STATUSES
                and status_value != current["status"]
            ):
                raise ValueError("terminal run status may not transition")
            values["status"] = status_value
        values["updated_at"] = utc_now()
        columns = ", ".join(f"{key} = ?" for key in values)
        parameters = [
            int(value) if key == "stop_requested" else value
            for key, value in values.items()
        ]
        parameters.append(run_id)
        self.connection.execute(
            f"UPDATE runs SET {columns} WHERE id = ?", parameters
        )
        return self.get_run(run_id)

    def request_stop(self, run_id: str) -> dict[str, Any]:
        now = utc_now()
        with self._transaction():
            self.connection.execute(
                "UPDATE runs SET stop_requested = 1, updated_at = ? WHERE id = ?",
                (now, run_id),
            )
            self._add_event(run_id, "STOP_REQUESTED", {})
        return self.get_run(run_id)

    def update_run_with_event(
        self,
        run_id: str,
        event: str,
        details: Mapping[str, Any],
        **values: Any,
    ) -> dict[str, Any]:
        with self._transaction():
            self._update_run_values(run_id, values)
            self._add_event(run_id, event, details)
        return self.get_run(run_id)

    def _update_run_values(
        self, run_id: str, values: Mapping[str, Any]
    ) -> None:
        allowed = {
            "status",
            "valid_candidates",
            "consecutive_failures",
            "stop_requested",
            "stop_reason",
            "final_best_hash",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unsupported run fields: {sorted(unknown)}")
        if not values:
            return
        prepared = dict(values)
        if "status" in prepared:
            status_value = _state_value(prepared["status"])
            RunStatus(status_value)
            row = self.connection.execute(
                "SELECT status FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"unknown run id: {run_id}")
            if (
                str(row["status"]) in TERMINAL_RUN_STATUSES
                and status_value != str(row["status"])
            ):
                raise ValueError("terminal run status may not transition")
            prepared["status"] = status_value
        prepared["updated_at"] = utc_now()
        columns = ", ".join(f"{key} = ?" for key in prepared)
        parameters = [
            int(value) if key == "stop_requested" else value
            for key, value in prepared.items()
        ]
        parameters.append(run_id)
        self.connection.execute(
            f"UPDATE runs SET {columns} WHERE id = ?", parameters
        )

    def create_iteration(
        self, run_id: str, iteration_index: int, parent_hash: str
    ) -> dict[str, Any]:
        now = utc_now()
        with self._transaction():
            cursor = self.connection.execute(
                """
                INSERT INTO iterations (
                    run_id, iteration_index, status, stage, created_at, updated_at,
                    parent_hash
                ) VALUES (?, ?, 'RUNNING', 'PROPOSE', ?, ?, ?)
                """,
                (run_id, iteration_index, now, now, parent_hash),
            )
            iteration_id = int(cursor.lastrowid)
            self._add_event(
                run_id,
                "ITERATION_CREATED",
                {"iteration_index": iteration_index},
                iteration_id=iteration_id,
            )
        return self.get_iteration(iteration_id)

    def get_iteration(self, iteration_id: int) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM iterations WHERE id = ?", (iteration_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown iteration id: {iteration_id}")
        result = dict(row)
        result["experiment_ids"] = json.loads(
            result.pop("experiment_ids_json")
        )
        result["result"] = json.loads(result.pop("result_json"))
        return result

    def get_iteration_by_index(
        self, run_id: str, iteration_index: int
    ) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT id FROM iterations WHERE run_id = ? AND iteration_index = ?",
            (run_id, iteration_index),
        ).fetchone()
        return None if row is None else self.get_iteration(int(row["id"]))

    def latest_iteration(self, run_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT id FROM iterations
            WHERE run_id = ?
            ORDER BY iteration_index DESC
            LIMIT 1
            """,
            (run_id,),
        ).fetchone()
        return None if row is None else self.get_iteration(int(row["id"]))

    def list_iterations(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT id FROM iterations
            WHERE run_id = ?
            ORDER BY iteration_index
            """,
            (run_id,),
        ).fetchall()
        return [self.get_iteration(int(row["id"])) for row in rows]

    def update_iteration(self, iteration_id: int, **values: Any) -> dict[str, Any]:
        allowed = {
            "status",
            "stage",
            "candidate_hash",
            "hypothesis",
            "rationale",
            "prompt_path",
            "raw_output_path",
            "candidate_path",
            "active_container",
            "experiment_ids",
            "result",
            "error",
            "outcome",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unsupported iteration fields: {sorted(unknown)}")
        if not values:
            return self.get_iteration(iteration_id)
        current = self.get_iteration(iteration_id)
        if "status" in values:
            status_value = _state_value(values["status"])
            IterationStatus(status_value)
            values["status"] = status_value
        if "stage" in values:
            stage_value = _state_value(values["stage"])
            validate_stage_transition(current["stage"], stage_value)
            values["stage"] = stage_value
        if (
            values.get("stage") == Stage.DONE.value
            and values.get("status", current["status"])
            != IterationStatus.COMPLETED.value
        ):
            raise ValueError("DONE iteration must be COMPLETED")
        if (
            values.get("status") == IterationStatus.COMPLETED.value
            and values.get("stage", current["stage"]) != Stage.DONE.value
        ):
            raise ValueError("COMPLETED iteration must be DONE")
        prepared: dict[str, Any] = {}
        for key, value in values.items():
            if key in {"experiment_ids", "result"}:
                prepared[f"{key}_json"] = _json(value)
            else:
                prepared[key] = value
        prepared["updated_at"] = utc_now()
        columns = ", ".join(f"{key} = ?" for key in prepared)
        parameters = [*prepared.values(), iteration_id]
        self.connection.execute(
            f"UPDATE iterations SET {columns} WHERE id = ?", parameters
        )
        return self.get_iteration(iteration_id)

    def update_iteration_with_event(
        self,
        iteration_id: int,
        event: str,
        details: Mapping[str, Any],
        **values: Any,
    ) -> dict[str, Any]:
        current = self.get_iteration(iteration_id)
        with self._transaction():
            self._update_iteration_values(iteration_id, current, values)
            self._add_event(
                str(current["run_id"]),
                event,
                details,
                iteration_id=iteration_id,
            )
        return self.get_iteration(iteration_id)

    def accept_candidate(
        self,
        iteration_id: int,
        *,
        candidate_hash: str,
        hypothesis: str,
        rationale: str,
        candidate_path: str,
    ) -> dict[str, Any]:
        current = self.get_iteration(iteration_id)
        if current["stage"] != Stage.PROPOSE.value:
            raise ValueError("candidate may only be accepted from PROPOSE")
        run_id = str(current["run_id"])
        with self._transaction():
            self.connection.execute(
                """
                UPDATE runs
                SET valid_candidates = valid_candidates + 1, updated_at = ?
                WHERE id = ?
                """,
                (utc_now(), run_id),
            )
            self._update_iteration_values(
                iteration_id,
                current,
                {
                    "stage": Stage.POLICY.value,
                    "candidate_hash": candidate_hash,
                    "hypothesis": hypothesis,
                    "rationale": rationale,
                    "candidate_path": candidate_path,
                    "active_container": None,
                },
            )
            self._add_event(
                run_id,
                "CANDIDATE_ACCEPTED",
                {"candidate_hash": candidate_hash},
                iteration_id=iteration_id,
            )
        return self.get_iteration(iteration_id)

    def _update_iteration_values(
        self,
        iteration_id: int,
        current: Mapping[str, Any],
        values: Mapping[str, Any],
    ) -> None:
        allowed = {
            "status",
            "stage",
            "candidate_hash",
            "hypothesis",
            "rationale",
            "prompt_path",
            "raw_output_path",
            "candidate_path",
            "active_container",
            "experiment_ids",
            "result",
            "error",
            "outcome",
        }
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"unsupported iteration fields: {sorted(unknown)}")
        prepared: dict[str, Any] = {}
        for key, value in values.items():
            if key == "status":
                status_value = _state_value(value)
                IterationStatus(status_value)
                prepared[key] = status_value
            elif key == "stage":
                stage_value = _state_value(value)
                validate_stage_transition(
                    str(current["stage"]), stage_value
                )
                prepared[key] = stage_value
            elif key in {"experiment_ids", "result"}:
                prepared[f"{key}_json"] = _json(value)
            else:
                prepared[key] = value
        target_stage = prepared.get("stage", current["stage"])
        target_status = prepared.get("status", current["status"])
        if (
            target_stage == Stage.DONE.value
            and target_status != IterationStatus.COMPLETED.value
        ):
            raise ValueError("DONE iteration must be COMPLETED")
        if (
            target_status == IterationStatus.COMPLETED.value
            and target_stage != Stage.DONE.value
        ):
            raise ValueError("COMPLETED iteration must be DONE")
        if not prepared:
            return
        prepared["updated_at"] = utc_now()
        columns = ", ".join(f"{key} = ?" for key in prepared)
        self.connection.execute(
            f"UPDATE iterations SET {columns} WHERE id = ?",
            [*prepared.values(), iteration_id],
        )

    def candidate_seen(self, candidate_hash: str) -> bool:
        row = self.connection.execute(
            """
            SELECT 1 FROM iterations
            WHERE candidate_hash = ?
              AND (outcome IS NULL OR outcome <> 'PROPOSAL_VALIDATED')
            LIMIT 1
            """,
            (candidate_hash,),
        ).fetchone()
        return row is not None

    def add_event(
        self,
        run_id: str,
        event: str,
        details: Mapping[str, Any],
        *,
        iteration_id: int | None = None,
    ) -> None:
        self._add_event(
            run_id, event, details, iteration_id=iteration_id
        )

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return [
            {
                **dict(row),
                "details": json.loads(row["details_json"]),
            }
            for row in rows
        ]

    def backup_to(self, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(destination)
        try:
            self.connection.backup(target)
        finally:
            target.close()
