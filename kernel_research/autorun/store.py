"""Durable controller state, separate from evaluator history."""

from __future__ import annotations

from datetime import datetime, timezone
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping

from ..platform.identity import BaselineRef
from ..platform.profiles import LEGACY_RESEARCH_NAMESPACE
from .states import (
    IterationStatus,
    RunStatus,
    Stage,
    TERMINAL_RUN_STATUSES,
    validate_stage_transition,
)
from .errors import ControlledRuntimeError

SCHEMA_VERSION = 3

# See HistoryStore's matching capability.  This is intentionally an opaque,
# identity-checked object instead of a bool that could be propagated by mistake.
_V2_TO_V3_MIGRATION_CAPABILITY = object()
_V2_TO_V3_MIGRATION_INTENT = ".v2-to-v3-migration-intent.json"

LEGACY_NAMESPACE_SNAPSHOT = LEGACY_RESEARCH_NAMESPACE.to_dict()
LEGACY_NAMESPACE_ID = LEGACY_RESEARCH_NAMESPACE.namespace_id
LEGACY_RESOLVED_CONFIG_DIGEST = "legacy_unknown"
LEGACY_WORKFLOW_SNAPSHOT = {
    "schema_version": 0,
    "status": "legacy_unknown",
}
LEGACY_HISTORY_CUTOFF = {
    "kind": "legacy_unknown",
}

PROPOSAL_ATTEMPT_STATUSES = frozenset(
    {"RUNNING", "SUCCEEDED", "FAILED", "UNKNOWN_OUTCOME"}
)
EVALUATION_ATTEMPT_STATUSES = frozenset(
    {"PENDING", "RUNNING", "SUCCEEDED", "FAILED", "UNKNOWN_OUTCOME"}
)
TERMINAL_ATTEMPT_STATUSES = frozenset(
    {"SUCCEEDED", "FAILED", "UNKNOWN_OUTCOME"}
)

_UNSET = object()


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


def _legacy_baseline_ref(initial_best_hash: str) -> dict[str, Any]:
    try:
        return BaselineRef.create(
            namespace=LEGACY_RESEARCH_NAMESPACE,
            artifact_id=f"source-sha256-v1:{initial_best_hash}",
            source="deployment",
            revision="legacy",
        ).to_dict()
    except ValueError as exc:
        raise ControlledRuntimeError(
            "legacy initial_best_hash cannot form a BaselineRef"
        ) from exc


class ControllerStore:
    def __init__(
        self,
        path: Path,
        *,
        _v2_to_v3_migration_capability: object | None = None,
    ) -> None:
        self.path = path
        self._authorize_persisted_legacy_schema(
            _v2_to_v3_migration_capability
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 5000")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self._initialize()

    def _authorize_persisted_legacy_schema(self, capability: object | None) -> None:
        """Reject implicit legacy migration before journal-affecting pragmas."""

        intent = self.path.parent / _V2_TO_V3_MIGRATION_INTENT
        if intent.exists() or intent.is_symlink():
            if capability is not _V2_TO_V3_MIGRATION_CAPABILITY:
                raise ControlledRuntimeError(
                    "coordinated V2-to-V3 migration intent is active; "
                    "Controller Store access is fenced"
                )
        if not self.path.exists():
            return
        if self.path.is_symlink() or not self.path.is_file():
            raise ControlledRuntimeError(
                "controller database must be a regular non-symlink file"
            )
        connection = sqlite3.connect(
            self.path.resolve().as_uri() + "?mode=ro", uri=True
        )
        user_objects: list[tuple[Any, ...]] = []
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 0:
                user_objects = connection.execute(
                    """
                    SELECT type, name FROM sqlite_master
                    WHERE name NOT LIKE 'sqlite_%'
                    ORDER BY type, name
                    """
                ).fetchall()
        finally:
            connection.close()
        if version == 0 and user_objects:
            rendered = ", ".join(
                f"{row[0]} {row[1]}" for row in user_objects[:8]
            )
            if len(user_objects) > 8:
                rendered += ", ..."
            raise ControlledRuntimeError(
                "persisted Controller schema version 0 contains user objects; "
                "refusing implicit initialization or migration: " + rendered
            )
        if version in {1, 2} and capability is not _V2_TO_V3_MIGRATION_CAPABILITY:
            raise ControlledRuntimeError(
                "persisted Controller schema V1/V2 requires the coordinated "
                "V2-to-V3 migration capability; direct Store migration is disabled"
            )

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
            version = 2
        if version == 2:
            self._migrate_v2_to_v3()

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

    def _migrate_v2_to_v3(self) -> None:
        active_runs = self.connection.execute(
            "SELECT id, status FROM runs WHERE status NOT IN ("
            + ",".join("?" for _ in TERMINAL_RUN_STATUSES)
            + ") ORDER BY id",
            tuple(sorted(TERMINAL_RUN_STATUSES)),
        ).fetchall()
        active_iterations = self.connection.execute(
            "SELECT run_id, iteration_index FROM iterations "
            "WHERE status <> 'COMPLETED' ORDER BY run_id, iteration_index"
        ).fetchall()
        if active_runs or active_iterations:
            identifiers = [
                f"run {row['id']} ({row['status']})" for row in active_runs
            ] + [
                f"iteration {row['run_id']}:{row['iteration_index']}"
                for row in active_iterations
            ]
            raise ControlledRuntimeError(
                "all runs and iterations must be terminal before controller "
                "v2 migration: " + ", ".join(identifiers)
            )
        proposal_statuses = ", ".join(
            f"'{status}'" for status in sorted(PROPOSAL_ATTEMPT_STATUSES)
        )
        evaluation_statuses = ", ".join(
            f"'{status}'" for status in sorted(EVALUATION_ATTEMPT_STATUSES)
        )
        terminal_statuses = ", ".join(
            f"'{status}'" for status in sorted(TERMINAL_ATTEMPT_STATUSES)
        )
        legacy_workflow = _json(LEGACY_WORKFLOW_SNAPSHOT).replace("'", "''")
        baseline_marker = "0" * 64
        baseline_template = _json(_legacy_baseline_ref(baseline_marker))
        baseline_prefix, baseline_suffix = baseline_template.split(
            baseline_marker
        )
        baseline_prefix = baseline_prefix.replace("'", "''")
        baseline_suffix = baseline_suffix.replace("'", "''")
        for row in self.connection.execute(
            "SELECT initial_best_hash FROM runs"
        ).fetchall():
            _legacy_baseline_ref(str(row["initial_best_hash"]))
        legacy_cutoff = _json(LEGACY_HISTORY_CUTOFF).replace("'", "''")
        self.connection.executescript(
            f"""
            BEGIN IMMEDIATE;
            ALTER TABLE runs ADD COLUMN namespace_id TEXT NOT NULL
                DEFAULT '{LEGACY_NAMESPACE_ID}';
            ALTER TABLE runs ADD COLUMN resolved_config_digest TEXT NOT NULL
                DEFAULT '{LEGACY_RESOLVED_CONFIG_DIGEST}';
            ALTER TABLE runs ADD COLUMN workflow_snapshot_json TEXT NOT NULL
                DEFAULT '{legacy_workflow}';
            ALTER TABLE runs ADD COLUMN baseline_ref_json TEXT NOT NULL
                DEFAULT '{{}}';
            ALTER TABLE runs ADD COLUMN history_cutoff_json TEXT NOT NULL
                DEFAULT '{legacy_cutoff}';
            UPDATE runs
            SET baseline_ref_json =
                '{baseline_prefix}' || initial_best_hash || '{baseline_suffix}';

            CREATE TRIGGER runs_v3_identity_immutable
            BEFORE UPDATE OF namespace_id, resolved_config_digest,
                workflow_snapshot_json, baseline_ref_json, history_cutoff_json
            ON runs
            WHEN NEW.namespace_id IS NOT OLD.namespace_id
              OR NEW.resolved_config_digest IS NOT OLD.resolved_config_digest
              OR NEW.workflow_snapshot_json IS NOT OLD.workflow_snapshot_json
              OR NEW.baseline_ref_json IS NOT OLD.baseline_ref_json
              OR NEW.history_cutoff_json IS NOT OLD.history_cutoff_json
            BEGIN
                SELECT RAISE(ABORT, 'run research identity is immutable');
            END;

            CREATE TABLE proposal_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_uid TEXT UNIQUE,
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                iteration_id INTEGER NOT NULL
                    REFERENCES iterations(id) ON DELETE CASCADE,
                attempt_index INTEGER NOT NULL,
                status TEXT NOT NULL CHECK (status IN ({proposal_statuses})),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                proposal_context_id TEXT,
                parent_artifact_id TEXT,
                proposer_profile_json TEXT NOT NULL DEFAULT '{{}}',
                prompt_protocol_json TEXT NOT NULL DEFAULT '{{}}',
                request_json TEXT NOT NULL DEFAULT '{{}}',
                result_json TEXT NOT NULL DEFAULT '{{}}',
                prompt_object_id TEXT,
                raw_output_object_id TEXT,
                candidate_artifact_id TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost_usd REAL,
                latency_ms REAL,
                error TEXT,
                UNIQUE(iteration_id, attempt_index)
            );
            CREATE INDEX proposal_attempts_run_iteration_idx
                ON proposal_attempts(run_id, iteration_id, attempt_index);
            CREATE TRIGGER proposal_attempts_owner_insert_guard
            BEFORE INSERT ON proposal_attempts
            WHEN NOT EXISTS (
                SELECT 1 FROM iterations
                WHERE id = NEW.iteration_id AND run_id = NEW.run_id
            )
            BEGIN
                SELECT RAISE(ABORT, 'proposal attempt iteration/run mismatch');
            END;
            CREATE TRIGGER proposal_attempts_owner_update_guard
            BEFORE UPDATE OF run_id, iteration_id ON proposal_attempts
            WHEN NOT EXISTS (
                SELECT 1 FROM iterations
                WHERE id = NEW.iteration_id AND run_id = NEW.run_id
            )
            BEGIN
                SELECT RAISE(ABORT, 'proposal attempt iteration/run mismatch');
            END;
            CREATE TRIGGER proposal_attempts_terminal_guard
            BEFORE UPDATE OF status ON proposal_attempts
            WHEN OLD.status IN ({terminal_statuses})
             AND NEW.status <> OLD.status
            BEGIN
                SELECT RAISE(ABORT, 'terminal proposal attempt may not transition');
            END;
            CREATE TRIGGER proposal_attempts_terminal_payload_guard
            BEFORE UPDATE OF result_json, raw_output_object_id,
                candidate_artifact_id, input_tokens, output_tokens, cost_usd,
                latency_ms, error
            ON proposal_attempts
            WHEN OLD.status IN ({terminal_statuses})
             AND (
                    NEW.result_json IS NOT OLD.result_json
                 OR NEW.raw_output_object_id IS NOT OLD.raw_output_object_id
                 OR NEW.candidate_artifact_id IS NOT OLD.candidate_artifact_id
                 OR NEW.input_tokens IS NOT OLD.input_tokens
                 OR NEW.output_tokens IS NOT OLD.output_tokens
                 OR NEW.cost_usd IS NOT OLD.cost_usd
                 OR NEW.latency_ms IS NOT OLD.latency_ms
                 OR NEW.error IS NOT OLD.error
             )
            BEGIN
                SELECT RAISE(ABORT, 'terminal proposal attempt is immutable');
            END;

            CREATE TABLE evaluation_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_uid TEXT NOT NULL UNIQUE,
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                iteration_id INTEGER NOT NULL
                    REFERENCES iterations(id) ON DELETE CASCADE,
                status TEXT NOT NULL CHECK (status IN ({evaluation_statuses})),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                history_linked_at TEXT,
                stage TEXT NOT NULL,
                suite TEXT NOT NULL,
                replicate_kind TEXT NOT NULL,
                replicate_index INTEGER NOT NULL CHECK (replicate_index >= 0),
                candidate_artifact_id TEXT,
                parent_artifact_id TEXT,
                baseline_ref_json TEXT NOT NULL DEFAULT '{{}}',
                condition_digest TEXT,
                request_json TEXT NOT NULL DEFAULT '{{}}',
                result_json TEXT NOT NULL DEFAULT '{{}}',
                history_experiment_id INTEGER,
                error TEXT
            );
            CREATE INDEX evaluation_attempts_run_iteration_idx
                ON evaluation_attempts(run_id, iteration_id, id);
            CREATE INDEX evaluation_attempts_unlinked_idx
                ON evaluation_attempts(status, history_experiment_id, id);
            CREATE UNIQUE INDEX evaluation_attempts_history_experiment_uidx
                ON evaluation_attempts(history_experiment_id)
                WHERE history_experiment_id IS NOT NULL;
            CREATE TRIGGER evaluation_attempts_owner_insert_guard
            BEFORE INSERT ON evaluation_attempts
            WHEN NOT EXISTS (
                SELECT 1 FROM iterations
                WHERE id = NEW.iteration_id AND run_id = NEW.run_id
            )
            BEGIN
                SELECT RAISE(ABORT, 'evaluation attempt iteration/run mismatch');
            END;
            CREATE TRIGGER evaluation_attempts_owner_update_guard
            BEFORE UPDATE OF run_id, iteration_id ON evaluation_attempts
            WHEN NOT EXISTS (
                SELECT 1 FROM iterations
                WHERE id = NEW.iteration_id AND run_id = NEW.run_id
            )
            BEGIN
                SELECT RAISE(ABORT, 'evaluation attempt iteration/run mismatch');
            END;
            CREATE TRIGGER evaluation_attempts_transition_guard
            BEFORE UPDATE OF status ON evaluation_attempts
            WHEN (OLD.status = 'PENDING' AND NEW.status NOT IN ('PENDING', 'RUNNING'))
              OR (OLD.status = 'RUNNING'
                  AND NEW.status NOT IN (
                      'RUNNING', 'SUCCEEDED', 'FAILED', 'UNKNOWN_OUTCOME'
                  ))
              OR (OLD.status IN ({terminal_statuses}) AND NEW.status <> OLD.status)
            BEGIN
                SELECT RAISE(ABORT, 'invalid evaluation attempt transition');
            END;
            CREATE TRIGGER evaluation_attempts_history_link_guard
            BEFORE UPDATE OF history_experiment_id ON evaluation_attempts
            WHEN OLD.history_experiment_id IS NOT NULL
             AND NEW.history_experiment_id IS NOT OLD.history_experiment_id
            BEGIN
                SELECT RAISE(ABORT, 'history experiment link is immutable');
            END;
            CREATE TRIGGER evaluation_attempts_history_status_insert_guard
            BEFORE INSERT ON evaluation_attempts
            WHEN NEW.history_experiment_id IS NOT NULL
             AND NEW.status <> 'SUCCEEDED'
            BEGIN
                SELECT RAISE(ABORT, 'only succeeded evaluation may link History');
            END;
            CREATE TRIGGER evaluation_attempts_history_status_update_guard
            BEFORE UPDATE OF history_experiment_id ON evaluation_attempts
            WHEN NEW.history_experiment_id IS NOT NULL
             AND NEW.status <> 'SUCCEEDED'
            BEGIN
                SELECT RAISE(ABORT, 'only succeeded evaluation may link History');
            END;
            CREATE TRIGGER evaluation_attempts_terminal_payload_guard
            BEFORE UPDATE OF result_json, finished_at, error
            ON evaluation_attempts
            WHEN OLD.status IN ({terminal_statuses})
             AND (
                    NEW.result_json IS NOT OLD.result_json
                 OR NEW.finished_at IS NOT OLD.finished_at
                 OR NEW.error IS NOT OLD.error
             )
            BEGIN
                SELECT RAISE(ABORT, 'terminal evaluation attempt is immutable');
            END;

            PRAGMA user_version = 3;
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
        namespace_id: str = LEGACY_NAMESPACE_ID,
        resolved_config_digest: str = LEGACY_RESOLVED_CONFIG_DIGEST,
        workflow_snapshot: Mapping[str, Any] | None = None,
        baseline_ref: Mapping[str, Any] | None = None,
        history_cutoff: Any = _UNSET,
    ) -> dict[str, Any]:
        if not isinstance(namespace_id, str) or not namespace_id:
            raise ValueError("namespace_id must be a non-empty string")
        if (
            not isinstance(resolved_config_digest, str)
            or not resolved_config_digest
        ):
            raise ValueError(
                "resolved_config_digest must be a non-empty string"
            )
        stored_workflow = (
            LEGACY_WORKFLOW_SNAPSHOT
            if workflow_snapshot is None
            else workflow_snapshot
        )
        if baseline_ref is None:
            if namespace_id != LEGACY_NAMESPACE_ID:
                raise ValueError(
                    "non-legacy runs require an explicit baseline_ref"
                )
            stored_baseline = _legacy_baseline_ref(initial_best_hash)
        else:
            stored_baseline = BaselineRef.from_value(
                dict(baseline_ref)
            ).to_dict()
            if stored_baseline["namespace_id"] != namespace_id:
                raise ValueError(
                    "baseline_ref must belong to the run research namespace"
                )
        stored_cutoff = (
            LEGACY_HISTORY_CUTOFF
            if history_cutoff is _UNSET
            else history_cutoff
        )
        now = utc_now()
        with self._transaction():
            self.connection.execute(
                """
                INSERT INTO runs (
                    id, status, created_at, updated_at, deadline_epoch,
                    config_json, preflight_json, initial_best_hash,
                    namespace_id, resolved_config_digest,
                    workflow_snapshot_json, baseline_ref_json,
                    history_cutoff_json
                ) VALUES (
                    ?, 'RUNNING', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    run_id,
                    now,
                    now,
                    deadline_epoch,
                    _json(config),
                    _json(preflight or {}),
                    initial_best_hash,
                    namespace_id,
                    resolved_config_digest,
                    _json(stored_workflow),
                    _json(stored_baseline),
                    _json(stored_cutoff),
                ),
            )
            self._add_event(
                run_id,
                "RUN_CREATED",
                {
                    "deadline_epoch": deadline_epoch,
                    "namespace_id": namespace_id,
                    "resolved_config_digest": resolved_config_digest,
                },
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
        result["workflow_snapshot"] = json.loads(
            result.pop("workflow_snapshot_json")
        )
        result["baseline_ref"] = json.loads(
            result.pop("baseline_ref_json")
        )
        result["history_cutoff"] = json.loads(
            result.pop("history_cutoff_json")
        )
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

    def list_recent_scientific_iterations(
        self,
        *,
        exclude_run_id: str | None,
        limit: int,
        namespace_id: str | None = None,
        history_cutoff: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return newest completed candidate iterations from earlier runs."""

        if limit < 0:
            raise ValueError("limit must be non-negative")
        if namespace_id is not None and (
            not isinstance(namespace_id, str) or not namespace_id
        ):
            raise ValueError("namespace_id must be non-empty when provided")
        if history_cutoff is not None and (
            type(history_cutoff) is not int or history_cutoff < 0
        ):
            raise ValueError("history_cutoff must be a non-negative integer")
        rows = self.connection.execute(
            """
            SELECT MAX(iteration.id) AS id
            FROM iterations AS iteration
            JOIN runs AS run ON run.id = iteration.run_id
            WHERE (? IS NULL OR iteration.run_id <> ?)
              AND (? IS NULL OR run.namespace_id = ?)
              AND (
                    ? IS NULL
                    OR COALESCE(
                        (
                            SELECT MAX(attempt.history_experiment_id)
                            FROM evaluation_attempts AS attempt
                            WHERE attempt.iteration_id = iteration.id
                              AND attempt.status = 'SUCCEEDED'
                        ),
                        9223372036854775807
                    ) <= ?
              )
              AND iteration.status = 'COMPLETED'
              AND iteration.candidate_hash IS NOT NULL
              AND iteration.outcome NOT IN ('PROPOSAL_VALIDATED', 'DUPLICATE')
            GROUP BY iteration.candidate_hash
            ORDER BY MAX(iteration.id) DESC
            LIMIT ?
            """,
            (
                exclude_run_id,
                exclude_run_id,
                namespace_id,
                namespace_id,
                history_cutoff,
                history_cutoff,
                limit,
            ),
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

    def _require_iteration_owner(
        self, run_id: str, iteration_id: int
    ) -> None:
        row = self.connection.execute(
            "SELECT run_id FROM iterations WHERE id = ?", (iteration_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown iteration id: {iteration_id}")
        if str(row["run_id"]) != run_id:
            raise ValueError("attempt iteration does not belong to run")

    @staticmethod
    def _proposal_attempt_from_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["proposer_profile"] = json.loads(
            result.pop("proposer_profile_json")
        )
        result["prompt_protocol"] = json.loads(
            result.pop("prompt_protocol_json")
        )
        result["request"] = json.loads(result.pop("request_json"))
        result["result"] = json.loads(result.pop("result_json"))
        return result

    def get_proposal_attempt(self, attempt_id: int) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM proposal_attempts WHERE id = ?", (attempt_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown proposal attempt id: {attempt_id}")
        return self._proposal_attempt_from_row(row)

    def list_proposal_attempts(
        self, run_id: str, *, iteration_id: int | None = None
    ) -> list[dict[str, Any]]:
        if iteration_id is None:
            rows = self.connection.execute(
                """
                SELECT * FROM proposal_attempts
                WHERE run_id = ?
                ORDER BY iteration_id, attempt_index
                """,
                (run_id,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT * FROM proposal_attempts
                WHERE run_id = ? AND iteration_id = ?
                ORDER BY attempt_index
                """,
                (run_id, iteration_id),
            ).fetchall()
        return [self._proposal_attempt_from_row(row) for row in rows]

    def create_proposal_attempt(
        self,
        *,
        run_id: str,
        iteration_id: int,
        attempt_index: int | None = None,
        attempt_uid: str | None = None,
        proposal_context_id: str | None = None,
        parent_artifact_id: str | None = None,
        proposer_profile: Mapping[str, Any] | None = None,
        prompt_protocol: Mapping[str, Any] | None = None,
        request: Mapping[str, Any] | None = None,
        prompt_object_id: str | None = None,
    ) -> dict[str, Any]:
        if attempt_uid is not None and not attempt_uid:
            raise ValueError("attempt_uid must be non-empty when provided")
        if attempt_index is not None and attempt_index < 1:
            raise ValueError("attempt_index must be positive")
        proposer_json = _json(proposer_profile or {})
        prompt_protocol_json = _json(prompt_protocol or {})
        request_json = _json(request or {})
        with self._transaction():
            self._require_iteration_owner(run_id, iteration_id)
            existing = None
            if attempt_uid is not None:
                existing = self.connection.execute(
                    "SELECT * FROM proposal_attempts WHERE attempt_uid = ?",
                    (attempt_uid,),
                ).fetchone()
            if existing is not None:
                expected = {
                    "run_id": run_id,
                    "iteration_id": iteration_id,
                    "proposal_context_id": proposal_context_id,
                    "parent_artifact_id": parent_artifact_id,
                    "proposer_profile_json": proposer_json,
                    "prompt_protocol_json": prompt_protocol_json,
                    "request_json": request_json,
                    "prompt_object_id": prompt_object_id,
                }
                if attempt_index is not None:
                    expected["attempt_index"] = attempt_index
                conflicts = [
                    key
                    for key, value in expected.items()
                    if existing[key] != value
                ]
                if conflicts:
                    raise ValueError(
                        "proposal attempt_uid conflicts with existing intent: "
                        + ", ".join(sorted(conflicts))
                    )
                attempt_id = int(existing["id"])
            else:
                if attempt_index is None:
                    row = self.connection.execute(
                        """
                        SELECT COALESCE(MAX(attempt_index), 0) + 1
                        FROM proposal_attempts WHERE iteration_id = ?
                        """,
                        (iteration_id,),
                    ).fetchone()
                    attempt_index = int(row[0])
                now = utc_now()
                cursor = self.connection.execute(
                    """
                    INSERT INTO proposal_attempts (
                        attempt_uid, run_id, iteration_id, attempt_index,
                        status, created_at, updated_at, proposal_context_id,
                        parent_artifact_id, proposer_profile_json,
                        prompt_protocol_json, request_json, prompt_object_id
                    ) VALUES (
                        ?, ?, ?, ?, 'RUNNING', ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        attempt_uid,
                        run_id,
                        iteration_id,
                        attempt_index,
                        now,
                        now,
                        proposal_context_id,
                        parent_artifact_id,
                        proposer_json,
                        prompt_protocol_json,
                        request_json,
                        prompt_object_id,
                    ),
                )
                attempt_id = int(cursor.lastrowid)
                self._add_event(
                    run_id,
                    "PROPOSAL_ATTEMPT_CREATED",
                    {
                        "attempt_id": attempt_id,
                        "attempt_index": attempt_index,
                        "attempt_uid": attempt_uid,
                    },
                    iteration_id=iteration_id,
                )
        return self.get_proposal_attempt(attempt_id)

    def finish_proposal_attempt(
        self,
        attempt_id: int,
        *,
        status: str,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
        raw_output_object_id: str | None = None,
        candidate_artifact_id: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
        latency_ms: float | None = None,
    ) -> dict[str, Any]:
        status_value = _state_value(status)
        if status_value not in TERMINAL_ATTEMPT_STATUSES:
            raise ValueError("proposal attempt must finish in a terminal status")
        for name, value in (
            ("input_tokens", input_tokens),
            ("output_tokens", output_tokens),
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")
        for name, value in (("cost_usd", cost_usd), ("latency_ms", latency_ms)):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        result_json = _json(result or {})
        target = {
            "status": status_value,
            "result_json": result_json,
            "error": error,
            "raw_output_object_id": raw_output_object_id,
            "candidate_artifact_id": candidate_artifact_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
            "latency_ms": latency_ms,
        }
        with self._transaction():
            current = self.connection.execute(
                "SELECT * FROM proposal_attempts WHERE id = ?", (attempt_id,)
            ).fetchone()
            if current is None:
                raise ValueError(f"unknown proposal attempt id: {attempt_id}")
            if str(current["status"]) in TERMINAL_ATTEMPT_STATUSES:
                conflicts = [
                    key
                    for key, value in target.items()
                    if current[key] != value
                ]
                if conflicts:
                    raise ValueError(
                        "terminal proposal attempt has conflicting result: "
                        + ", ".join(sorted(conflicts))
                    )
            else:
                now = utc_now()
                assignments = ", ".join(f"{key} = ?" for key in target)
                self.connection.execute(
                    f"""
                    UPDATE proposal_attempts
                    SET {assignments}, updated_at = ?
                    WHERE id = ?
                    """,
                    (*target.values(), now, attempt_id),
                )
                self._add_event(
                    str(current["run_id"]),
                    "PROPOSAL_ATTEMPT_FINISHED",
                    {"attempt_id": attempt_id, "status": status_value},
                    iteration_id=int(current["iteration_id"]),
                )
        return self.get_proposal_attempt(attempt_id)

    @staticmethod
    def _evaluation_attempt_from_row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["baseline_ref"] = json.loads(
            result.pop("baseline_ref_json")
        )
        result["request"] = json.loads(result.pop("request_json"))
        result["result"] = json.loads(result.pop("result_json"))
        return result

    def get_evaluation_attempt_by_uid(
        self, experiment_uid: str
    ) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM evaluation_attempts WHERE experiment_uid = ?",
            (experiment_uid,),
        ).fetchone()
        return None if row is None else self._evaluation_attempt_from_row(row)

    def list_evaluation_attempts(
        self, run_id: str, *, iteration_id: int | None = None
    ) -> list[dict[str, Any]]:
        if iteration_id is None:
            rows = self.connection.execute(
                """
                SELECT * FROM evaluation_attempts
                WHERE run_id = ? ORDER BY id
                """,
                (run_id,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT * FROM evaluation_attempts
                WHERE run_id = ? AND iteration_id = ? ORDER BY id
                """,
                (run_id, iteration_id),
            ).fetchall()
        return [self._evaluation_attempt_from_row(row) for row in rows]

    def list_evaluation_attempts_by_replicate_kind(
        self, replicate_kind: str
    ) -> list[dict[str, Any]]:
        """Read the durable cross-Run ledger for one evidence-only kind."""

        if not isinstance(replicate_kind, str) or not replicate_kind:
            raise ValueError("replicate_kind must be a non-empty string")
        rows = self.connection.execute(
            """
            SELECT * FROM evaluation_attempts
            WHERE replicate_kind = ? ORDER BY id
            """,
            (replicate_kind,),
        ).fetchall()
        return [self._evaluation_attempt_from_row(row) for row in rows]

    def create_evaluation_attempt(
        self,
        *,
        experiment_uid: str,
        run_id: str,
        iteration_id: int,
        stage: str,
        suite: str,
        replicate_kind: str = "primary",
        replicate_index: int = 0,
        candidate_artifact_id: str | None = None,
        parent_artifact_id: str | None = None,
        baseline_ref: Mapping[str, Any] | None = None,
        condition_digest: str | None = None,
        request: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        for name, value in (
            ("experiment_uid", experiment_uid),
            ("stage", stage),
            ("suite", suite),
            ("replicate_kind", replicate_kind),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if (
            type(replicate_index) is not int
            or replicate_index < 0
            or replicate_index > 9223372036854775807
        ):
            raise ValueError(
                "replicate_index must fit a non-negative SQLite INTEGER"
            )
        request_json = _json(request or {})
        with self._transaction():
            self._require_iteration_owner(run_id, iteration_id)
            run = self.connection.execute(
                "SELECT baseline_ref_json FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise ValueError(f"unknown run id: {run_id}")
            run_baseline_json = str(run["baseline_ref_json"])
            baseline_json = (
                run_baseline_json if baseline_ref is None else _json(baseline_ref)
            )
            if baseline_json != run_baseline_json:
                raise ValueError(
                    "evaluation baseline_ref must match the frozen run baseline"
                )
            intent = {
                "run_id": run_id,
                "iteration_id": iteration_id,
                "stage": stage,
                "suite": suite,
                "replicate_kind": replicate_kind,
                "replicate_index": replicate_index,
                "candidate_artifact_id": candidate_artifact_id,
                "parent_artifact_id": parent_artifact_id,
                "baseline_ref_json": baseline_json,
                "condition_digest": condition_digest,
                "request_json": request_json,
            }
            existing = self.connection.execute(
                "SELECT * FROM evaluation_attempts WHERE experiment_uid = ?",
                (experiment_uid,),
            ).fetchone()
            if existing is not None:
                conflicts = [
                    key
                    for key, value in intent.items()
                    if existing[key] != value
                ]
                if conflicts:
                    raise ValueError(
                        "experiment_uid conflicts with existing intent: "
                        + ", ".join(sorted(conflicts))
                    )
            else:
                now = utc_now()
                columns = ", ".join(intent)
                placeholders = ", ".join("?" for _ in intent)
                cursor = self.connection.execute(
                    f"""
                    INSERT INTO evaluation_attempts (
                        experiment_uid, status, created_at, updated_at,
                        {columns}
                    ) VALUES (?, 'PENDING', ?, ?, {placeholders})
                    """,
                    (experiment_uid, now, now, *intent.values()),
                )
                attempt_id = int(cursor.lastrowid)
                self._add_event(
                    run_id,
                    "EVALUATION_ATTEMPT_CREATED",
                    {
                        "attempt_id": attempt_id,
                        "experiment_uid": experiment_uid,
                        "stage": stage,
                    },
                    iteration_id=iteration_id,
                )
        attempt = self.get_evaluation_attempt_by_uid(experiment_uid)
        if attempt is None:  # pragma: no cover - protected by the transaction
            raise RuntimeError("evaluation attempt disappeared after creation")
        return attempt

    def start_evaluation_attempt(
        self, experiment_uid: str
    ) -> dict[str, Any]:
        with self._transaction():
            current = self.connection.execute(
                "SELECT * FROM evaluation_attempts WHERE experiment_uid = ?",
                (experiment_uid,),
            ).fetchone()
            if current is None:
                raise ValueError(f"unknown experiment_uid: {experiment_uid}")
            status = str(current["status"])
            if status == "PENDING":
                now = utc_now()
                self.connection.execute(
                    """
                    UPDATE evaluation_attempts
                    SET status = 'RUNNING', started_at = ?, updated_at = ?
                    WHERE experiment_uid = ?
                    """,
                    (now, now, experiment_uid),
                )
                self._add_event(
                    str(current["run_id"]),
                    "EVALUATION_ATTEMPT_STARTED",
                    {"experiment_uid": experiment_uid},
                    iteration_id=int(current["iteration_id"]),
                )
            elif status != "RUNNING":
                raise ValueError(
                    f"terminal evaluation attempt may not restart: {status}"
                )
        attempt = self.get_evaluation_attempt_by_uid(experiment_uid)
        if attempt is None:  # pragma: no cover - protected by the transaction
            raise RuntimeError("evaluation attempt disappeared after start")
        return attempt

    def finish_evaluation_attempt(
        self,
        experiment_uid: str,
        *,
        status: str,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        status_value = _state_value(status)
        if status_value not in TERMINAL_ATTEMPT_STATUSES:
            raise ValueError("evaluation attempt must finish in a terminal status")
        result_json = _json(result or {})
        with self._transaction():
            current = self.connection.execute(
                "SELECT * FROM evaluation_attempts WHERE experiment_uid = ?",
                (experiment_uid,),
            ).fetchone()
            if current is None:
                raise ValueError(f"unknown experiment_uid: {experiment_uid}")
            current_status = str(current["status"])
            if current_status == "PENDING":
                raise ValueError("evaluation attempt must be started before finish")
            if current_status in TERMINAL_ATTEMPT_STATUSES:
                if (
                    current_status != status_value
                    or current["result_json"] != result_json
                    or current["error"] != error
                ):
                    raise ValueError(
                        "terminal evaluation attempt has conflicting result"
                    )
            else:
                now = utc_now()
                self.connection.execute(
                    """
                    UPDATE evaluation_attempts
                    SET status = ?, result_json = ?, error = ?,
                        finished_at = ?, updated_at = ?
                    WHERE experiment_uid = ?
                    """,
                    (
                        status_value,
                        result_json,
                        error,
                        now,
                        now,
                        experiment_uid,
                    ),
                )
                self._add_event(
                    str(current["run_id"]),
                    "EVALUATION_ATTEMPT_FINISHED",
                    {
                        "experiment_uid": experiment_uid,
                        "status": status_value,
                    },
                    iteration_id=int(current["iteration_id"]),
                )
        attempt = self.get_evaluation_attempt_by_uid(experiment_uid)
        if attempt is None:  # pragma: no cover - protected by the transaction
            raise RuntimeError("evaluation attempt disappeared after finish")
        return attempt

    def link_history_experiment(
        self, experiment_uid: str, history_experiment_id: int
    ) -> dict[str, Any]:
        if type(history_experiment_id) is not int or history_experiment_id < 1:
            raise ValueError("history_experiment_id must be a positive integer")
        with self._transaction():
            current = self.connection.execute(
                "SELECT * FROM evaluation_attempts WHERE experiment_uid = ?",
                (experiment_uid,),
            ).fetchone()
            if current is None:
                raise ValueError(f"unknown experiment_uid: {experiment_uid}")
            if str(current["status"]) != "SUCCEEDED":
                raise ValueError(
                    "only a succeeded evaluation attempt may link History"
                )
            owner = self.connection.execute(
                """
                SELECT experiment_uid FROM evaluation_attempts
                WHERE history_experiment_id = ?
                """,
                (history_experiment_id,),
            ).fetchone()
            if owner is not None and str(owner["experiment_uid"]) != experiment_uid:
                raise ValueError(
                    "History experiment is already linked to another attempt"
                )
            linked = current["history_experiment_id"]
            if linked is None:
                now = utc_now()
                self.connection.execute(
                    """
                    UPDATE evaluation_attempts
                    SET history_experiment_id = ?, history_linked_at = ?,
                        updated_at = ?
                    WHERE experiment_uid = ?
                    """,
                    (history_experiment_id, now, now, experiment_uid),
                )
                self._add_event(
                    str(current["run_id"]),
                    "HISTORY_EXPERIMENT_LINKED",
                    {
                        "experiment_uid": experiment_uid,
                        "history_experiment_id": history_experiment_id,
                    },
                    iteration_id=int(current["iteration_id"]),
                )
            elif int(linked) != history_experiment_id:
                raise ValueError(
                    "experiment_uid is already linked to another History record"
                )
        attempt = self.get_evaluation_attempt_by_uid(experiment_uid)
        if attempt is None:  # pragma: no cover - protected by the transaction
            raise RuntimeError("evaluation attempt disappeared after History link")
        return attempt

    def reconcile_evaluation_attempt(
        self, experiment_uid: str, history_experiment_id: int
    ) -> dict[str, Any]:
        """Idempotently connect a persisted History row to its controller intent."""

        return self.link_history_experiment(
            experiment_uid, history_experiment_id
        )

    def list_unlinked_evaluation_attempts(
        self, *, run_id: str | None = None
    ) -> list[dict[str, Any]]:
        if run_id is None:
            rows = self.connection.execute(
                """
                SELECT * FROM evaluation_attempts
                WHERE status = 'SUCCEEDED'
                  AND history_experiment_id IS NULL
                ORDER BY id
                """
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT * FROM evaluation_attempts
                WHERE status = 'SUCCEEDED'
                  AND history_experiment_id IS NULL
                  AND run_id = ?
                ORDER BY id
                """,
                (run_id,),
            ).fetchall()
        return [self._evaluation_attempt_from_row(row) for row in rows]

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

    def candidate_seen_in_namespace(
        self, namespace_id: str, candidate_hash: str
    ) -> bool:
        """Return whether this scientific namespace already proposed bytes.

        Candidate bytes are deliberately reusable across namespaces.  The
        controller database therefore scopes duplicate suppression through
        the owning run instead of treating ``candidate_hash`` as a global
        identity.
        """

        if not isinstance(namespace_id, str) or not namespace_id:
            raise ValueError("namespace_id must be a non-empty string")
        row = self.connection.execute(
            """
            SELECT 1
            FROM iterations AS iteration
            JOIN runs AS run ON run.id = iteration.run_id
            WHERE run.namespace_id = ?
              AND iteration.candidate_hash = ?
              AND (
                  iteration.outcome IS NULL
                  OR iteration.outcome <> 'PROPOSAL_VALIDATED'
              )
            LIMIT 1
            """,
            (namespace_id, candidate_hash),
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
