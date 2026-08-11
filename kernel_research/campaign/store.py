"""SQLite v1 store for bounded, crash-recoverable campaigns."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Iterator, Mapping, Sequence
import uuid

from ..platform.identity import BaselineRef, ExecutionEnvironmentDigest
from .models import (
    BudgetAmount,
    CampaignMode,
    CampaignStatus,
    ChildRunStatus,
    MAX_CHILD_CANDIDATES,
    MAX_CHILD_CONSECUTIVE_FAILURES,
    MAX_CHILD_WALL_SECONDS,
    ResourceLease,
    TERMINAL_CAMPAIGN_STATUSES,
    TERMINAL_CHILD_STATUSES,
)
from .soak_collector import COUNT_FIELDS, SoakObservation


SCHEMA_VERSION = 1
SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ARTIFACT_ID_RE = re.compile(
    r"^(?:source|bundle)-sha256-v1:[0-9a-f]{64}$"
)
OJ_TYPES = frozenset({"PRIMARY_SUBMISSION", "EXPLORATORY_SUBMISSION"})
OJ_VERDICTS = frozenset(
    {
        "ACCEPTED",
        "WRONG_ANSWER",
        "TIME_LIMIT",
        "RUNTIME_ERROR",
        "COMPILE_ERROR",
        "SYSTEM_ERROR",
        "OTHER",
    }
)
SOAK_STAGE_ORDER = (
    "MVP_24H",
    "STAGED_LINEAGE_72H",
    "LONG_CAMPAIGN_168H",
)
SOAK_STAGE_REQUIRED_SECONDS = {
    "MVP_24H": 24 * 60 * 60,
    "STAGED_LINEAGE_72H": 72 * 60 * 60,
    "LONG_CAMPAIGN_168H": 168 * 60 * 60,
}
SOAK_GENERATION_STATUSES = frozenset({"ACTIVE", "COMPLETED", "VIOLATED"})
SOAK_MAX_HEARTBEAT_GAP_SECONDS = 300


def _utc_now() -> str:
    milliseconds = int((time.time() % 1) * 1000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{milliseconds:03d}Z"


def _json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _token(value: str, name: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be a non-empty bounded string")
    if any(ord(character) < 0x20 for character in value):
        raise ValueError(f"{name} contains control characters")
    return value


def _digest(value: str, name: str) -> str:
    if not isinstance(value, str) or not SHA256_DIGEST_RE.fullmatch(value):
        raise ValueError(f"{name} must be sha256:<64 lowercase hex digits>")
    return value


def _artifact_id(value: str) -> str:
    if not isinstance(value, str) or not ARTIFACT_ID_RE.fullmatch(value):
        raise ValueError("artifact_id must be a tagged V1 source or bundle digest")
    return value


def _finite_epoch(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def _non_negative_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _utc_epoch(value: str, name: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be an RFC3339 UTC timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    epoch = parsed.astimezone(timezone.utc).timestamp()
    if not math.isfinite(epoch) or epoch < 0:
        raise ValueError(f"{name} must be a finite non-negative timestamp")
    return epoch


def _baseline_ref(value: object, name: str) -> BaselineRef:
    try:
        selected = (
            value
            if isinstance(value, BaselineRef)
            else BaselineRef.from_value(
                dict(value) if isinstance(value, Mapping) else value
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid BaselineRef") from exc
    if selected.source != "campaign":
        raise ValueError(f"{name} must be Campaign-owned")
    return selected


class CampaignStore:
    """Single-writer campaign store with explicit recovery boundaries."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self._initialize()

    def __enter__(self) -> "CampaignStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def _initialize(self) -> None:
        version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if version not in (0, SCHEMA_VERSION):
            raise ValueError(
                f"unsupported campaign schema version {version}; expected {SCHEMA_VERSION}"
            )
        if version == 0:
            self.connection.executescript(
                """
            BEGIN IMMEDIATE;
            CREATE TABLE campaigns (
                id TEXT PRIMARY KEY,
                namespace_id TEXT NOT NULL,
                mode TEXT NOT NULL CHECK (mode IN ('BENCHMARK', 'DISCOVERY')),
                status TEXT NOT NULL CHECK (status IN (
                    'CREATED', 'RUNNING', 'PAUSED_OPERATOR', 'PAUSED_BUDGET',
                    'PAUSED_UNKNOWN_OUTCOME', 'PAUSED_HARD_FAILURE',
                    'PAUSED_DATA_INTEGRITY', 'COMPLETED', 'CANCELLED'
                )),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                snapshot_digest TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                max_candidates INTEGER NOT NULL CHECK (max_candidates >= 0),
                max_wall_ms INTEGER NOT NULL CHECK (max_wall_ms >= 0),
                max_gpu_ms INTEGER NOT NULL CHECK (max_gpu_ms >= 0),
                max_tokens INTEGER NOT NULL CHECK (max_tokens >= 0),
                max_cost_microusd INTEGER NOT NULL CHECK (max_cost_microusd >= 0),
                active_baseline_revision_id TEXT,
                allow_staged_lineage INTEGER NOT NULL CHECK (allow_staged_lineage IN (0, 1)),
                stop_reason TEXT,
                pause_evidence_digest TEXT
            );
            CREATE TRIGGER campaign_identity_immutable
            BEFORE UPDATE OF namespace_id, mode, snapshot_digest, snapshot_json,
                max_candidates, max_wall_ms, max_gpu_ms, max_tokens,
                max_cost_microusd, allow_staged_lineage
            ON campaigns
            WHEN NEW.namespace_id IS NOT OLD.namespace_id
              OR NEW.mode IS NOT OLD.mode
              OR NEW.snapshot_digest IS NOT OLD.snapshot_digest
              OR NEW.snapshot_json IS NOT OLD.snapshot_json
              OR NEW.max_candidates IS NOT OLD.max_candidates
              OR NEW.max_wall_ms IS NOT OLD.max_wall_ms
              OR NEW.max_gpu_ms IS NOT OLD.max_gpu_ms
              OR NEW.max_tokens IS NOT OLD.max_tokens
              OR NEW.max_cost_microusd IS NOT OLD.max_cost_microusd
              OR NEW.allow_staged_lineage IS NOT OLD.allow_staged_lineage
            BEGIN
                SELECT RAISE(ABORT, 'campaign identity and limits are immutable');
            END;

            CREATE TABLE baseline_revisions (
                id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL REFERENCES campaigns(id) ON DELETE RESTRICT,
                revision_index INTEGER NOT NULL CHECK (revision_index >= 0),
                namespace_id TEXT NOT NULL,
                parent_revision_id TEXT REFERENCES baseline_revisions(id),
                artifact_id TEXT NOT NULL,
                baseline_ref_json TEXT,
                revision_kind TEXT NOT NULL CHECK (revision_kind IN ('DEPLOYMENT_SEED', 'STAGED')),
                primary_experiment_uid TEXT,
                confirmation_experiment_uid TEXT,
                policy_snapshot_json TEXT NOT NULL,
                idempotency_key TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(campaign_id, revision_index),
                UNIQUE(campaign_id, idempotency_key)
            );
            CREATE INDEX baseline_revisions_parent_idx
                ON baseline_revisions(campaign_id, parent_revision_id);

            CREATE TABLE child_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL REFERENCES campaigns(id) ON DELETE RESTRICT,
                child_index INTEGER NOT NULL CHECK (child_index > 0),
                controller_run_id TEXT UNIQUE,
                status TEXT NOT NULL CHECK (status IN (
                    'PENDING', 'RUNNING', 'PROMOTED', 'BUDGET_EXHAUSTED',
                    'STOPPED', 'FAILED', 'HARD_FAILED', 'UNKNOWN_GPU_OUTCOME'
                )),
                baseline_revision_id TEXT NOT NULL REFERENCES baseline_revisions(id),
                proposer_profile_json TEXT NOT NULL,
                max_candidates INTEGER NOT NULL CHECK (max_candidates BETWEEN 1 AND 5),
                max_wall_seconds INTEGER NOT NULL CHECK (max_wall_seconds BETWEEN 1 AND 21600),
                max_consecutive_failures INTEGER NOT NULL CHECK (max_consecutive_failures BETWEEN 1 AND 3),
                stop_after_promotion INTEGER NOT NULL CHECK (stop_after_promotion = 1),
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                result_json TEXT NOT NULL DEFAULT '{}',
                UNIQUE(campaign_id, child_index)
            );
            CREATE UNIQUE INDEX child_runs_one_active_idx
                ON child_runs(campaign_id)
                WHERE status IN ('PENDING', 'RUNNING');

            CREATE TABLE budget_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL REFERENCES campaigns(id) ON DELETE RESTRICT,
                idempotency_key TEXT NOT NULL,
                action_kind TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('RESERVED', 'SETTLED', 'CANCELLED')),
                reserved_candidates INTEGER NOT NULL CHECK (reserved_candidates >= 0),
                reserved_wall_ms INTEGER NOT NULL CHECK (reserved_wall_ms >= 0),
                reserved_gpu_ms INTEGER NOT NULL CHECK (reserved_gpu_ms >= 0),
                reserved_tokens INTEGER NOT NULL CHECK (reserved_tokens >= 0),
                reserved_cost_microusd INTEGER NOT NULL CHECK (reserved_cost_microusd >= 0),
                actual_candidates INTEGER CHECK (actual_candidates >= 0),
                actual_wall_ms INTEGER CHECK (actual_wall_ms >= 0),
                actual_gpu_ms INTEGER CHECK (actual_gpu_ms >= 0),
                actual_tokens INTEGER CHECK (actual_tokens >= 0),
                actual_cost_microusd INTEGER CHECK (actual_cost_microusd >= 0),
                created_at TEXT NOT NULL,
                settled_at TEXT,
                UNIQUE(campaign_id, idempotency_key)
            );

            CREATE TABLE resource_leases (
                resource_id TEXT NOT NULL,
                fencing_epoch INTEGER NOT NULL CHECK (fencing_epoch > 0),
                campaign_id TEXT NOT NULL REFERENCES campaigns(id) ON DELETE RESTRICT,
                status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'RELEASED', 'EXPIRED', 'QUARANTINED')),
                acquired_at TEXT NOT NULL,
                expires_epoch REAL NOT NULL,
                released_at TEXT,
                reason TEXT,
                PRIMARY KEY(resource_id, fencing_epoch)
            );
            CREATE UNIQUE INDEX resource_one_active_lease_idx
                ON resource_leases(resource_id)
                WHERE status = 'ACTIVE';

            CREATE TABLE oj_nominations (
                id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL REFERENCES campaigns(id) ON DELETE RESTRICT,
                namespace_id TEXT NOT NULL,
                nomination_type TEXT NOT NULL CHECK (nomination_type IN ('PRIMARY_SUBMISSION', 'EXPLORATORY_SUBMISSION')),
                artifact_id TEXT NOT NULL,
                baseline_revision_id TEXT REFERENCES baseline_revisions(id),
                bundle_object_id TEXT NOT NULL,
                manifest_digest TEXT NOT NULL,
                local_metrics_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('NOMINATED', 'EXPORTED', 'SUBMITTED', 'FEEDBACK_RECORDED')),
                created_at TEXT NOT NULL,
                exported_at TEXT,
                submission_id TEXT,
                verdict TEXT,
                score REAL,
                note TEXT,
                feedback_at TEXT,
                UNIQUE(campaign_id, manifest_digest)
            );

            CREATE TABLE outbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL REFERENCES campaigns(id) ON DELETE RESTRICT,
                idempotency_key TEXT NOT NULL UNIQUE,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                delivered_at TEXT
            );
            PRAGMA user_version = 1;
            COMMIT;
                """
            )
        # Soak gates are part of the first published campaign schema.  Early
        # V1 databases may predate these additive tables, so V1 initialization
        # deliberately remains idempotent instead of bumping user_version.
        self._initialize_soak_schema()
        self._initialize_scientific_identity_schema()

    def _initialize_scientific_identity_schema(self) -> None:
        """Add fail-closed V2 baseline and trusted-doctor evidence storage.

        Early Campaign V1 databases have no environment identity.  Their rows
        remain NULL deliberately: callers can inspect them, but no code path
        may infer an environment or use them for staged lineage.
        """

        columns = {
            str(row[1])
            for row in self.connection.execute(
                "PRAGMA table_info(baseline_revisions)"
            )
        }
        if "baseline_ref_json" not in columns:
            self.connection.execute(
                "ALTER TABLE baseline_revisions ADD COLUMN baseline_ref_json TEXT"
            )
        self.connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TRIGGER IF NOT EXISTS baseline_revision_ref_immutable
            BEFORE UPDATE OF baseline_ref_json ON baseline_revisions
            BEGIN
                SELECT RAISE(ABORT, 'baseline revision identity is immutable');
            END;

            CREATE TABLE IF NOT EXISTS campaign_doctor_evidence (
                digest TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL REFERENCES campaigns(id) ON DELETE RESTRICT,
                resource_id TEXT NOT NULL,
                quarantine_fencing_epoch INTEGER NOT NULL,
                observed_epoch REAL NOT NULL,
                execution_environment_digest TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS campaign_doctor_evidence_campaign_idx
                ON campaign_doctor_evidence(campaign_id, observed_epoch);
            CREATE TRIGGER IF NOT EXISTS campaign_doctor_evidence_immutable_update
            BEFORE UPDATE ON campaign_doctor_evidence
            BEGIN
                SELECT RAISE(ABORT, 'campaign doctor evidence is immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS campaign_doctor_evidence_immutable_delete
            BEFORE DELETE ON campaign_doctor_evidence
            BEGIN
                SELECT RAISE(ABORT, 'campaign doctor evidence is immutable');
            END;
            COMMIT;
            """
        )
        doctor_columns = {
            str(row[1])
            for row in self.connection.execute(
                "PRAGMA table_info(campaign_doctor_evidence)"
            )
        }
        if "quarantine_fencing_epoch" not in doctor_columns:
            # Any row created by the earlier additive draft lacks a provable
            # quarantine identity.  NULL is intentionally fail-closed.
            self.connection.execute(
                """
                ALTER TABLE campaign_doctor_evidence
                ADD COLUMN quarantine_fencing_epoch INTEGER
                """
            )
            self.connection.commit()

    def _initialize_soak_schema(self) -> None:
        self.connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS soak_generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gate_id TEXT NOT NULL,
                stage TEXT NOT NULL CHECK (stage IN (
                    'MVP_24H', 'STAGED_LINEAGE_72H', 'LONG_CAMPAIGN_168H'
                )),
                generation_index INTEGER NOT NULL CHECK (generation_index > 0),
                status TEXT NOT NULL CHECK (status IN (
                    'ACTIVE', 'COMPLETED', 'VIOLATED'
                )),
                required_seconds REAL NOT NULL CHECK (
                    (stage = 'MVP_24H' AND required_seconds = 86400)
                    OR (stage = 'STAGED_LINEAGE_72H' AND required_seconds = 259200)
                    OR (stage = 'LONG_CAMPAIGN_168H' AND required_seconds = 604800)
                ),
                invariant_snapshot_digest TEXT NOT NULL,
                invariant_snapshot_json TEXT NOT NULL,
                started_epoch REAL NOT NULL,
                last_heartbeat_epoch REAL NOT NULL,
                accumulated_seconds REAL NOT NULL CHECK (accumulated_seconds >= 0),
                completed_epoch REAL,
                violated_epoch REAL,
                initial_evidence_id TEXT,
                last_evidence_id TEXT,
                UNIQUE(gate_id, stage, generation_index)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS soak_one_active_generation_idx
                ON soak_generations(gate_id) WHERE status = 'ACTIVE';
            CREATE INDEX IF NOT EXISTS soak_generations_gate_stage_idx
                ON soak_generations(gate_id, stage, generation_index);
            CREATE TRIGGER IF NOT EXISTS soak_generation_identity_immutable
            BEFORE UPDATE OF gate_id, stage, generation_index, required_seconds,
                invariant_snapshot_digest, invariant_snapshot_json, started_epoch
            ON soak_generations
            WHEN NEW.gate_id IS NOT OLD.gate_id
              OR NEW.stage IS NOT OLD.stage
              OR NEW.generation_index IS NOT OLD.generation_index
              OR NEW.required_seconds IS NOT OLD.required_seconds
              OR NEW.invariant_snapshot_digest IS NOT OLD.invariant_snapshot_digest
              OR NEW.invariant_snapshot_json IS NOT OLD.invariant_snapshot_json
              OR NEW.started_epoch IS NOT OLD.started_epoch
            BEGIN
                SELECT RAISE(ABORT, 'soak generation identity is immutable');
            END;

            CREATE TABLE IF NOT EXISTS soak_violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                gate_id TEXT NOT NULL,
                stage TEXT NOT NULL CHECK (stage IN (
                    'MVP_24H', 'STAGED_LINEAGE_72H', 'LONG_CAMPAIGN_168H'
                )),
                generation_id INTEGER NOT NULL
                    REFERENCES soak_generations(id) ON DELETE RESTRICT,
                generation_index INTEGER NOT NULL CHECK (generation_index > 0),
                invariant_snapshot_digest TEXT NOT NULL,
                observed_epoch REAL NOT NULL,
                lease_overlap_count INTEGER NOT NULL CHECK (lease_overlap_count >= 0),
                orphan_container_count INTEGER NOT NULL CHECK (orphan_container_count >= 0),
                hard_fault_retry_count INTEGER NOT NULL CHECK (hard_fault_retry_count >= 0),
                budget_leak_count INTEGER NOT NULL CHECK (budget_leak_count >= 0),
                evidence_json TEXT NOT NULL,
                evidence_id TEXT,
                reason_code TEXT
            );
            CREATE INDEX IF NOT EXISTS soak_violations_gate_idx
                ON soak_violations(gate_id, id);
            CREATE TRIGGER IF NOT EXISTS soak_violation_immutable_update
            BEFORE UPDATE ON soak_violations
            BEGIN
                SELECT RAISE(ABORT, 'soak violations are immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS soak_violation_immutable_delete
            BEFORE DELETE ON soak_violations
            BEGIN
                SELECT RAISE(ABORT, 'soak violations are immutable');
            END;

            CREATE TABLE IF NOT EXISTS soak_observations (
                gate_id TEXT NOT NULL,
                evidence_id TEXT NOT NULL,
                generation_id INTEGER NOT NULL
                    REFERENCES soak_generations(id) ON DELETE RESTRICT,
                collector_revision TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('AVAILABLE', 'UNAVAILABLE')),
                interval_start_epoch REAL NOT NULL,
                interval_end_epoch REAL NOT NULL,
                invariant_snapshot_digest TEXT NOT NULL,
                observation_json TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                PRIMARY KEY(gate_id, evidence_id)
            );
            CREATE INDEX IF NOT EXISTS soak_observations_generation_idx
                ON soak_observations(generation_id, interval_end_epoch);
            CREATE TRIGGER IF NOT EXISTS soak_observation_immutable_update
            BEFORE UPDATE ON soak_observations
            BEGIN
                SELECT RAISE(ABORT, 'soak observations are immutable');
            END;
            CREATE TRIGGER IF NOT EXISTS soak_observation_immutable_delete
            BEFORE DELETE ON soak_observations
            BEGIN
                SELECT RAISE(ABORT, 'soak observations are immutable');
            END;
            COMMIT;
            """
        )
        generation_columns = {
            str(row[1])
            for row in self.connection.execute(
                "PRAGMA table_info(soak_generations)"
            )
        }
        for column in ("initial_evidence_id", "last_evidence_id"):
            if column not in generation_columns:
                self.connection.execute(
                    f"ALTER TABLE soak_generations ADD COLUMN {column} TEXT"
                )
        violation_columns = {
            str(row[1])
            for row in self.connection.execute(
                "PRAGMA table_info(soak_violations)"
            )
        }
        for column in ("evidence_id", "reason_code"):
            if column not in violation_columns:
                self.connection.execute(
                    f"ALTER TABLE soak_violations ADD COLUMN {column} TEXT"
                )
        self.connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TRIGGER IF NOT EXISTS soak_generation_initial_evidence_immutable
            BEFORE UPDATE OF initial_evidence_id ON soak_generations
            WHEN NEW.initial_evidence_id IS NOT OLD.initial_evidence_id
            BEGIN
                SELECT RAISE(ABORT, 'soak generation initial evidence is immutable');
            END;
            COMMIT;
            """
        )

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for field in (
            "snapshot_json",
            "policy_snapshot_json",
            "baseline_ref_json",
            "proposer_profile_json",
            "result_json",
            "local_metrics_json",
            "payload_json",
            "invariant_snapshot_json",
            "evidence_json",
            "observation_json",
        ):
            if field in result:
                encoded = result.pop(field)
                result[field.removesuffix("_json")] = (
                    None if encoded is None else json.loads(encoded)
                )
        for field in ("allow_staged_lineage", "stop_after_promotion"):
            if field in result:
                result[field] = bool(result[field])
        return result

    def create_campaign(
        self,
        *,
        campaign_id: str,
        namespace_id: str,
        mode: str,
        snapshot: Mapping[str, Any],
        budget_limit: BudgetAmount,
        initial_artifact_id: str | None = None,
        initial_baseline_ref: Mapping[str, Any] | BaselineRef | None = None,
        initial_policy_snapshot: Mapping[str, Any],
        allow_staged_lineage: bool = False,
    ) -> dict[str, Any]:
        _token(campaign_id, "campaign_id")
        _digest(namespace_id, "namespace_id")
        selected_mode = CampaignMode(mode)
        if initial_baseline_ref is not None and initial_artifact_id is not None:
            raise ValueError(
                "provide initial_baseline_ref or legacy initial_artifact_id, not both"
            )
        revision_id = "baseline-" + uuid.uuid4().hex
        if initial_baseline_ref is None:
            if initial_artifact_id is None:
                raise ValueError("initial_baseline_ref is required")
            # Compatibility for the published V1 store API.  The resulting
            # reference is explicit but scientifically unknown, so it can run
            # only in a non-advancing Campaign.
            _artifact_id(initial_artifact_id)
            baseline_ref = BaselineRef.create(
                namespace=namespace_id,
                artifact_id=initial_artifact_id,
                source="campaign",
                revision=revision_id,
            )
        else:
            baseline_ref = _baseline_ref(
                initial_baseline_ref, "initial_baseline_ref"
            )
            if baseline_ref.namespace_id != namespace_id:
                raise ValueError("initial baseline belongs to another namespace")
            initial_artifact_id = str(baseline_ref.artifact_id)
            # The imported BaselineRef revision is a scientific coordinate in
            # its source lineage.  This Campaign owns a distinct local row ID;
            # active pointers and child FKs always use that local identity.
        if selected_mode is CampaignMode.BENCHMARK and allow_staged_lineage:
            raise ValueError("benchmark campaigns may not enable staged lineage")
        if allow_staged_lineage and not baseline_ref.is_scientifically_comparable:
            raise ValueError(
                "staged lineage requires a V2 seed with a resolved execution environment"
            )
        snapshot_value = dict(snapshot)
        snapshot_digest = _canonical_digest(snapshot_value)
        now = _utc_now()
        with self._transaction():
            self.connection.execute(
                """
                INSERT INTO campaigns(
                    id, namespace_id, mode, status, created_at, updated_at,
                    snapshot_digest, snapshot_json, max_candidates, max_wall_ms,
                    max_gpu_ms, max_tokens, max_cost_microusd,
                    active_baseline_revision_id, allow_staged_lineage
                ) VALUES (?, ?, ?, 'CREATED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    namespace_id,
                    selected_mode.value,
                    now,
                    now,
                    snapshot_digest,
                    _json(snapshot_value),
                    budget_limit.candidates,
                    budget_limit.wall_ms,
                    budget_limit.gpu_ms,
                    budget_limit.tokens,
                    budget_limit.cost_microusd,
                    revision_id,
                    int(allow_staged_lineage),
                ),
            )
            self.connection.execute(
                """
                INSERT INTO baseline_revisions(
                    id, campaign_id, revision_index, namespace_id,
                    parent_revision_id, artifact_id, baseline_ref_json,
                    revision_kind,
                    policy_snapshot_json, created_at
                ) VALUES (?, ?, 0, ?, NULL, ?, ?, 'DEPLOYMENT_SEED', ?, ?)
                """,
                (
                    revision_id,
                    campaign_id,
                    namespace_id,
                    initial_artifact_id,
                    _json(baseline_ref.to_dict()),
                    _json(dict(initial_policy_snapshot)),
                    now,
                ),
            )
            self._insert_outbox(
                campaign_id,
                f"campaign-created:{campaign_id}",
                "CAMPAIGN_CREATED",
                {"campaign_id": campaign_id, "namespace_id": namespace_id},
                now=now,
            )
        return self.get_campaign(campaign_id)

    def get_campaign(self, campaign_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown campaign: {campaign_id}")
        result = self._row(row)
        result["budget"] = self.budget_status(campaign_id)
        return result

    def list_campaigns(self, *, active_only: bool = False) -> list[dict[str, Any]]:
        query = "SELECT * FROM campaigns"
        parameters: tuple[Any, ...] = ()
        if active_only:
            placeholders = ",".join("?" for _ in TERMINAL_CAMPAIGN_STATUSES)
            query += f" WHERE status NOT IN ({placeholders})"
            parameters = tuple(sorted(TERMINAL_CAMPAIGN_STATUSES))
        query += " ORDER BY created_at, id"
        return [self._row(row) for row in self.connection.execute(query, parameters)]

    def begin_soak_generation(
        self,
        *,
        gate_id: str,
        stage: str,
        observation: SoakObservation,
    ) -> dict[str, Any]:
        """Begin an ordered gate from a zero-length trusted observation."""

        gate_id = _token(gate_id, "gate_id")
        if stage not in SOAK_STAGE_ORDER:
            raise ValueError("unknown soak stage")
        trusted = self._trusted_soak_observation(observation)
        if trusted.interval_start_epoch != trusted.interval_end_epoch:
            raise ValueError("a soak start observation must have a zero interval")
        with self._transaction():
            active = self.connection.execute(
                """
                SELECT * FROM soak_generations
                WHERE gate_id = ? AND status = 'ACTIVE'
                """,
                (gate_id,),
            ).fetchone()
            if active is not None:
                if active["initial_evidence_id"] is None:
                    return self._violate_soak_generation(
                        gate_id=gate_id,
                        active=active,
                        observation=trusted,
                        reason_code="UNTRUSTED_LEGACY_STATE",
                    )
                if active["last_evidence_id"] == trusted.evidence_id:
                    return {
                        "outcome": "REPLAYED",
                        "generation": self._row(active),
                        "violation": None,
                        "observation": trusted.to_dict(),
                    }
                if (
                    active["invariant_snapshot_digest"]
                    != trusted.invariant_snapshot_digest
                ):
                    return self._violate_soak_generation(
                        gate_id=gate_id,
                        active=active,
                        observation=trusted,
                        reason_code="INVARIANT_CHANGED",
                    )
                raise ValueError("soak gate is already active; use heartbeat")

            completed = {
                str(row["stage"])
                for row in self.connection.execute(
                    """
                    SELECT DISTINCT stage FROM soak_generations
                    WHERE gate_id = ? AND status = 'COMPLETED'
                      AND invariant_snapshot_digest = ?
                      AND initial_evidence_id IS NOT NULL
                    """,
                    (gate_id, trusted.invariant_snapshot_digest),
                )
            }
            expected_stage = next(
                (item for item in SOAK_STAGE_ORDER if item not in completed),
                None,
            )
            if expected_stage is None:
                raise ValueError("all soak stages are already complete")
            if stage != expected_stage:
                raise ValueError(
                    f"soak stages must run in order; expected {expected_stage}"
                )
            generation = self._insert_soak_generation(
                gate_id=gate_id,
                stage=stage,
                observation=trusted,
                started_epoch=trusted.interval_end_epoch,
            )
            self._insert_soak_observation(
                gate_id=gate_id,
                generation_id=int(generation["id"]),
                observation=trusted,
            )
            reason = self._soak_observation_reason(trusted)
            if reason is not None:
                return self._violate_soak_generation(
                    gate_id=gate_id,
                    active=generation,
                    observation=trusted,
                    reason_code=reason,
                    observation_already_inserted=True,
                )
            return {
                "outcome": "STARTED",
                "generation": self._row(generation),
                "violation": None,
                "observation": trusted.to_dict(),
            }

    @staticmethod
    def _trusted_soak_observation(
        observation: SoakObservation,
    ) -> SoakObservation:
        if type(observation) is not SoakObservation:
            raise TypeError(
                "soak state accepts only a trusted SoakObservation object"
            )
        # Reconstruct behind the collector's private seal to detect mutation of
        # caller-visible JSON mappings after the original validation.
        return observation._validated_copy()

    def _insert_soak_generation(
        self,
        *,
        gate_id: str,
        stage: str,
        observation: SoakObservation,
        started_epoch: float,
    ) -> sqlite3.Row:
        generation_index = int(
            self.connection.execute(
                """
                SELECT COALESCE(MAX(generation_index), 0) + 1
                FROM soak_generations WHERE gate_id = ? AND stage = ?
                """,
                (gate_id, stage),
            ).fetchone()[0]
        )
        cursor = self.connection.execute(
            """
            INSERT INTO soak_generations(
                gate_id, stage, generation_index, status, required_seconds,
                invariant_snapshot_digest, invariant_snapshot_json,
                started_epoch, last_heartbeat_epoch, accumulated_seconds,
                initial_evidence_id, last_evidence_id
            ) VALUES (?, ?, ?, 'ACTIVE', ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                gate_id,
                stage,
                generation_index,
                SOAK_STAGE_REQUIRED_SECONDS[stage],
                observation.invariant_snapshot_digest,
                _json(observation.invariant_snapshot),
                started_epoch,
                started_epoch,
                observation.evidence_id,
                observation.evidence_id,
            ),
        )
        row = self.connection.execute(
            "SELECT * FROM soak_generations WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
        assert row is not None
        return row

    def _insert_soak_observation(
        self,
        *,
        gate_id: str,
        generation_id: int,
        observation: SoakObservation,
    ) -> None:
        encoded = _json(observation.to_dict())
        existing = self.connection.execute(
            """
            SELECT generation_id, observation_json FROM soak_observations
            WHERE gate_id = ? AND evidence_id = ?
            """,
            (gate_id, observation.evidence_id),
        ).fetchone()
        if existing is not None:
            if (
                int(existing["generation_id"]) != generation_id
                or existing["observation_json"] != encoded
            ):
                raise ValueError("soak evidence replay disagrees with stored evidence")
            return
        self.connection.execute(
            """
            INSERT INTO soak_observations(
                gate_id, evidence_id, generation_id, collector_revision,
                status, interval_start_epoch, interval_end_epoch,
                invariant_snapshot_digest, observation_json, recorded_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                gate_id,
                observation.evidence_id,
                generation_id,
                observation.collector_revision,
                observation.status,
                observation.interval_start_epoch,
                observation.interval_end_epoch,
                observation.invariant_snapshot_digest,
                encoded,
                _utc_now(),
            ),
        )

    @staticmethod
    def _soak_observation_reason(
        observation: SoakObservation,
    ) -> str | None:
        if observation.status == "UNAVAILABLE":
            return "SOURCE_UNAVAILABLE"
        if any(observation.counts[field] for field in COUNT_FIELDS):
            return "INVARIANT_VIOLATION"
        return None

    def _stored_soak_observation(
        self, *, gate_id: str, evidence_id: str
    ) -> dict[str, Any]:
        row = self.connection.execute(
            """
            SELECT observation_json FROM soak_observations
            WHERE gate_id = ? AND evidence_id = ?
            """,
            (gate_id, evidence_id),
        ).fetchone()
        if row is None:
            raise ValueError("soak generation has no trusted initial evidence")
        value = json.loads(str(row["observation_json"]))
        if not isinstance(value, dict):
            raise ValueError("stored soak observation is invalid")
        return value

    @staticmethod
    def _soak_activity(value: Mapping[str, Any]) -> dict[str, Any]:
        try:
            sources = value["sources"]
            campaign = sources["campaign"]
            controller = sources["controller"]
            if (
                campaign["status"] != "AVAILABLE"
                or controller["status"] != "AVAILABLE"
            ):
                raise ValueError("soak activity source is unavailable")
            campaign_activity = campaign["summary"]["activity"]
            controller_activity = controller["summary"]["activity"]
            terminal_values = campaign_activity["terminal_children"]
            revision_values = campaign_activity["staged_revisions"]
            run_values = controller_activity["controller_run_ids"]
            experiment_values = controller_activity[
                "confirmed_experiment_uids"
            ]
        except (KeyError, TypeError) as exc:
            raise ValueError("soak activity evidence is incomplete") from exc
        if not all(
            isinstance(items, list)
            for items in (
                terminal_values,
                revision_values,
                run_values,
                experiment_values,
            )
        ):
            raise ValueError("soak activity evidence has invalid fields")
        terminal_children: dict[int, str] = {}
        for child in terminal_values:
            if not isinstance(child, dict):
                raise ValueError("terminal child activity is invalid")
            child_id = child.get("child_id")
            run_id = child.get("controller_run_id")
            status = child.get("status")
            if (
                type(child_id) is not int
                or child_id <= 0
                or not isinstance(run_id, str)
                or not run_id
                or not isinstance(status, str)
                or status in {"PENDING", "RUNNING"}
            ):
                raise ValueError("terminal child activity is invalid")
            if child_id in terminal_children:
                raise ValueError("terminal child activity is duplicated")
            terminal_children[child_id] = run_id
        revisions: dict[str, tuple[str, str]] = {}
        for revision in revision_values:
            if not isinstance(revision, dict):
                raise ValueError("staged lineage activity is invalid")
            revision_id = revision.get("revision_id")
            primary = revision.get("primary_experiment_uid")
            confirmation = revision.get("confirmation_experiment_uid")
            if not all(
                isinstance(item, str) and item
                for item in (revision_id, primary, confirmation)
            ):
                raise ValueError("staged lineage activity is invalid")
            if revision_id in revisions:
                raise ValueError("staged lineage activity is duplicated")
            revisions[revision_id] = (primary, confirmation)
        if not all(isinstance(item, str) and item for item in run_values):
            raise ValueError("Controller run activity is invalid")
        if not all(
            isinstance(item, str) and item for item in experiment_values
        ):
            raise ValueError("Controller experiment activity is invalid")
        return {
            "terminal_children": terminal_children,
            "staged_revisions": revisions,
            "controller_run_ids": set(run_values),
            "confirmed_experiment_uids": set(experiment_values),
        }

    def _soak_stage_activity_satisfied(
        self,
        *,
        gate_id: str,
        generation: sqlite3.Row,
        observation: SoakObservation,
    ) -> bool:
        initial_id = generation["initial_evidence_id"]
        if not isinstance(initial_id, str) or not initial_id:
            return False
        initial = self._soak_activity(
            self._stored_soak_observation(
                gate_id=gate_id, evidence_id=initial_id
            )
        )
        final = self._soak_activity(observation.to_dict())
        stage = str(generation["stage"])
        if stage in {"MVP_24H", "LONG_CAMPAIGN_168H"}:
            new_child_ids = (
                set(final["terminal_children"])
                - set(initial["terminal_children"])
            )
            return any(
                final["terminal_children"][child_id]
                in final["controller_run_ids"]
                for child_id in new_child_ids
            )
        if stage == "STAGED_LINEAGE_72H":
            new_revision_ids = (
                set(final["staged_revisions"])
                - set(initial["staged_revisions"])
            )
            confirmed = final["confirmed_experiment_uids"]
            return any(
                set(final["staged_revisions"][revision_id]).issubset(
                    confirmed
                )
                for revision_id in new_revision_ids
            )
        raise ValueError("unknown soak stage")

    def _violate_soak_generation(
        self,
        *,
        gate_id: str,
        active: sqlite3.Row,
        observation: SoakObservation,
        reason_code: str,
        observation_already_inserted: bool = False,
    ) -> dict[str, Any]:
        allowed_reasons = {
            "SOURCE_UNAVAILABLE",
            "HEARTBEAT_GAP",
            "INVARIANT_VIOLATION",
            "INVARIANT_CHANGED",
            "UNTRUSTED_LEGACY_STATE",
            "SOURCE_RECOVERED_REANCHOR",
        }
        if reason_code not in allowed_reasons:
            raise ValueError("unknown soak violation reason")
        observed = max(
            float(active["last_heartbeat_epoch"]),
            observation.interval_end_epoch,
        )
        if not observation_already_inserted:
            self._insert_soak_observation(
                gate_id=gate_id,
                generation_id=int(active["id"]),
                observation=observation,
            )
        updated = self.connection.execute(
            """
            UPDATE soak_generations
            SET status = 'VIOLATED', last_heartbeat_epoch = ?,
                last_evidence_id = ?, violated_epoch = ?
            WHERE id = ? AND status = 'ACTIVE'
            """,
            (observed, observation.evidence_id, observed, active["id"]),
        )
        if updated.rowcount != 1:
            raise ValueError("soak generation is no longer active")
        counts = observation.counts
        violation_cursor = self.connection.execute(
            """
            INSERT INTO soak_violations(
                gate_id, stage, generation_id, generation_index,
                invariant_snapshot_digest, observed_epoch,
                lease_overlap_count, orphan_container_count,
                hard_fault_retry_count, budget_leak_count, evidence_json,
                evidence_id, reason_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                gate_id,
                active["stage"],
                active["id"],
                active["generation_index"],
                active["invariant_snapshot_digest"],
                observed,
                counts["lease_overlap_count"],
                counts["orphan_container_count"],
                counts["hard_fault_retry_count"],
                counts["budget_leak_count"],
                _json(observation.to_dict()),
                observation.evidence_id,
                reason_code,
            ),
        )
        restart_stage = (
            SOAK_STAGE_ORDER[0]
            if reason_code
            in {"INVARIANT_CHANGED", "UNTRUSTED_LEGACY_STATE"}
            else str(active["stage"])
        )
        restart_observation = observation
        generation = self._insert_soak_generation(
            gate_id=gate_id,
            stage=restart_stage,
            observation=restart_observation,
            started_epoch=observed,
        )
        violation = self.connection.execute(
            "SELECT * FROM soak_violations WHERE id = ?",
            (violation_cursor.lastrowid,),
        ).fetchone()
        assert violation is not None
        return {
            "outcome": "VIOLATION",
            "generation": self._row(generation),
            "violation": self._row(violation),
            "observation": observation.to_dict(),
        }

    def get_active_soak_generation(self, gate_id: str) -> dict[str, Any] | None:
        gate_id = _token(gate_id, "gate_id")
        row = self.connection.execute(
            """
            SELECT * FROM soak_generations
            WHERE gate_id = ? AND status = 'ACTIVE'
            """,
            (gate_id,),
        ).fetchone()
        return None if row is None else self._row(row)

    def get_soak_heartbeat_anchor(self, gate_id: str) -> dict[str, Any] | None:
        """Return the active generation, or the newest completed monitor head."""

        gate_id = _token(gate_id, "gate_id")
        row = self.connection.execute(
            """
            SELECT * FROM soak_generations
            WHERE gate_id = ? AND status IN ('ACTIVE', 'COMPLETED')
            ORDER BY CASE status WHEN 'ACTIVE' THEN 0 ELSE 1 END,
                     COALESCE(completed_epoch, last_heartbeat_epoch) DESC,
                     id DESC
            LIMIT 1
            """,
            (gate_id,),
        ).fetchone()
        return None if row is None else self._row(row)

    def list_soak_generations(self, gate_id: str) -> list[dict[str, Any]]:
        gate_id = _token(gate_id, "gate_id")
        return [
            self._row(row)
            for row in self.connection.execute(
                """
                SELECT * FROM soak_generations
                WHERE gate_id = ?
                ORDER BY CASE stage
                    WHEN 'MVP_24H' THEN 1
                    WHEN 'STAGED_LINEAGE_72H' THEN 2
                    ELSE 3 END,
                    generation_index
                """,
                (gate_id,),
            )
        ]

    def list_soak_violations(self, gate_id: str) -> list[dict[str, Any]]:
        gate_id = _token(gate_id, "gate_id")
        return [
            self._row(row)
            for row in self.connection.execute(
                """
                SELECT * FROM soak_violations
                WHERE gate_id = ? ORDER BY id
                """,
                (gate_id,),
            )
        ]

    def list_soak_observations(self, gate_id: str) -> list[dict[str, Any]]:
        gate_id = _token(gate_id, "gate_id")
        return [
            self._row(row)
            for row in self.connection.execute(
                """
                SELECT * FROM soak_observations
                WHERE gate_id = ? ORDER BY interval_end_epoch, evidence_id
                """,
                (gate_id,),
            )
        ]

    def _record_completed_soak_monitor(
        self,
        *,
        gate_id: str,
        completed: sqlite3.Row,
        observation: SoakObservation,
    ) -> dict[str, Any]:
        if completed["initial_evidence_id"] is None:
            reason = "UNTRUSTED_LEGACY_STATE"
        elif (
            completed["invariant_snapshot_digest"]
            != observation.invariant_snapshot_digest
        ):
            reason = "INVARIANT_CHANGED"
        else:
            previous = float(completed["last_heartbeat_epoch"])
            elapsed = observation.interval_end_epoch - previous
            if (
                observation.interval_start_epoch != previous
                or elapsed < 0
                or elapsed > SOAK_MAX_HEARTBEAT_GAP_SECONDS
            ):
                reason = "HEARTBEAT_GAP"
            else:
                reason = self._soak_observation_reason(observation)
        if reason is None:
            self._insert_soak_observation(
                gate_id=gate_id,
                generation_id=int(completed["id"]),
                observation=observation,
            )
            self.connection.execute(
                """
                UPDATE soak_generations
                SET last_heartbeat_epoch = ?, last_evidence_id = ?
                WHERE id = ? AND status = 'COMPLETED'
                """,
                (
                    observation.interval_end_epoch,
                    observation.evidence_id,
                    completed["id"],
                ),
            )
            monitored = self.connection.execute(
                "SELECT * FROM soak_generations WHERE id = ?",
                (completed["id"],),
            ).fetchone()
            assert monitored is not None
            return {
                "outcome": "QUALIFIED_MONITORED",
                "generation": self._row(monitored),
                "violation": None,
                "observation": observation.to_dict(),
            }

        observed = max(
            float(completed["last_heartbeat_epoch"]),
            observation.interval_end_epoch,
        )
        self._insert_soak_observation(
            gate_id=gate_id,
            generation_id=int(completed["id"]),
            observation=observation,
        )
        self.connection.execute(
            """
            UPDATE soak_generations
            SET status = 'VIOLATED', violated_epoch = ?
            WHERE gate_id = ? AND status = 'COMPLETED'
              AND invariant_snapshot_digest = ?
            """,
            (observed, gate_id, completed["invariant_snapshot_digest"]),
        )
        self.connection.execute(
            """
            UPDATE soak_generations
            SET last_heartbeat_epoch = ?, last_evidence_id = ?
            WHERE id = ?
            """,
            (observed, observation.evidence_id, completed["id"]),
        )
        counts = observation.counts
        violation_cursor = self.connection.execute(
            """
            INSERT INTO soak_violations(
                gate_id, stage, generation_id, generation_index,
                invariant_snapshot_digest, observed_epoch,
                lease_overlap_count, orphan_container_count,
                hard_fault_retry_count, budget_leak_count, evidence_json,
                evidence_id, reason_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                gate_id,
                completed["stage"],
                completed["id"],
                completed["generation_index"],
                completed["invariant_snapshot_digest"],
                observed,
                counts["lease_overlap_count"],
                counts["orphan_container_count"],
                counts["hard_fault_retry_count"],
                counts["budget_leak_count"],
                _json(observation.to_dict()),
                observation.evidence_id,
                reason,
            ),
        )
        generation = self._insert_soak_generation(
            gate_id=gate_id,
            stage=SOAK_STAGE_ORDER[0],
            observation=observation,
            started_epoch=observed,
        )
        violation = self.connection.execute(
            "SELECT * FROM soak_violations WHERE id = ?",
            (violation_cursor.lastrowid,),
        ).fetchone()
        assert violation is not None
        return {
            "outcome": "VIOLATION",
            "generation": self._row(generation),
            "violation": self._row(violation),
            "observation": observation.to_dict(),
        }

    def record_soak_heartbeat(
        self,
        *,
        gate_id: str,
        observation: SoakObservation,
    ) -> dict[str, Any]:
        """Apply one trusted observation atomically to the active generation."""

        gate_id = _token(gate_id, "gate_id")
        trusted = self._trusted_soak_observation(observation)
        with self._transaction():
            active = self.connection.execute(
                """
                SELECT * FROM soak_generations
                WHERE gate_id = ? AND status = 'ACTIVE'
                """,
                (gate_id,),
            ).fetchone()
            if active is None:
                completed = self.connection.execute(
                    """
                    SELECT * FROM soak_generations
                    WHERE gate_id = ? AND status = 'COMPLETED'
                    ORDER BY completed_epoch DESC, id DESC LIMIT 1
                    """,
                    (gate_id,),
                ).fetchone()
                if completed is None:
                    raise ValueError("soak gate has no active generation")
                return self._record_completed_soak_monitor(
                    gate_id=gate_id,
                    completed=completed,
                    observation=trusted,
                )
            if active["initial_evidence_id"] is None:
                return self._violate_soak_generation(
                    gate_id=gate_id,
                    active=active,
                    observation=trusted,
                    reason_code="UNTRUSTED_LEGACY_STATE",
                )
            if active["last_evidence_id"] == trusted.evidence_id:
                return {
                    "outcome": "REPLAYED",
                    "generation": self._row(active),
                    "violation": None,
                    "observation": trusted.to_dict(),
                }
            previous_heartbeat = float(active["last_heartbeat_epoch"])
            if (
                active["invariant_snapshot_digest"]
                != trusted.invariant_snapshot_digest
            ):
                return self._violate_soak_generation(
                    gate_id=gate_id,
                    active=active,
                    observation=trusted,
                    reason_code="INVARIANT_CHANGED",
                )
            initial_observation = self._stored_soak_observation(
                gate_id=gate_id,
                evidence_id=str(active["initial_evidence_id"]),
            )
            if (
                initial_observation.get("status") == "UNAVAILABLE"
                and trusted.status == "AVAILABLE"
            ):
                return self._violate_soak_generation(
                    gate_id=gate_id,
                    active=active,
                    observation=trusted,
                    reason_code="SOURCE_RECOVERED_REANCHOR",
                )
            if initial_observation.get("status") != "UNAVAILABLE":
                try:
                    self._soak_activity(initial_observation)
                except ValueError:
                    return self._violate_soak_generation(
                        gate_id=gate_id,
                        active=active,
                        observation=trusted,
                        reason_code="UNTRUSTED_LEGACY_STATE",
                    )
            elapsed = trusted.interval_end_epoch - previous_heartbeat
            if (
                trusted.interval_start_epoch != previous_heartbeat
                or elapsed < 0
                or elapsed > SOAK_MAX_HEARTBEAT_GAP_SECONDS
            ):
                return self._violate_soak_generation(
                    gate_id=gate_id,
                    active=active,
                    observation=trusted,
                    reason_code="HEARTBEAT_GAP",
                )
            reason = self._soak_observation_reason(trusted)
            if reason is not None:
                return self._violate_soak_generation(
                    gate_id=gate_id,
                    active=active,
                    observation=trusted,
                    reason_code=reason,
                )
            self._insert_soak_observation(
                gate_id=gate_id,
                generation_id=int(active["id"]),
                observation=trusted,
            )
            accumulated = min(
                float(active["required_seconds"]),
                float(active["accumulated_seconds"]) + elapsed,
            )
            duration_satisfied = accumulated >= float(
                active["required_seconds"]
            )
            activity_satisfied = self._soak_stage_activity_satisfied(
                gate_id=gate_id,
                generation=active,
                observation=trusted,
            )
            completed = duration_satisfied and activity_satisfied
            self.connection.execute(
                """
                UPDATE soak_generations
                SET status = ?, last_heartbeat_epoch = ?,
                    last_evidence_id = ?, accumulated_seconds = ?,
                    completed_epoch = ?
                WHERE id = ? AND status = 'ACTIVE'
                """,
                (
                    "COMPLETED" if completed else "ACTIVE",
                    trusted.interval_end_epoch,
                    trusted.evidence_id,
                    accumulated,
                    trusted.interval_end_epoch if completed else None,
                    active["id"],
                ),
            )
            generation = self.connection.execute(
                "SELECT * FROM soak_generations WHERE id = ?",
                (active["id"],),
            ).fetchone()
            assert generation is not None
            return {
                "outcome": (
                    "COMPLETED"
                    if completed
                    else (
                        "PENDING_ACTIVITY"
                        if duration_satisfied
                        else "ACCUMULATED"
                    )
                ),
                "generation": self._row(generation),
                "violation": None,
                "observation": trusted.to_dict(),
                "duration_satisfied": duration_satisfied,
                "activity_satisfied": activity_satisfied,
            }

    def completed_soak_stages(
        self, gate_id: str, invariant_snapshot_digest: str
    ) -> tuple[str, ...]:
        gate_id = _token(gate_id, "gate_id")
        digest = _digest(invariant_snapshot_digest, "invariant_snapshot_digest")
        completed = {
            str(row["stage"])
            for row in self.connection.execute(
                """
                SELECT DISTINCT stage FROM soak_generations
                WHERE gate_id = ? AND status = 'COMPLETED'
                  AND invariant_snapshot_digest = ?
                  AND initial_evidence_id IS NOT NULL
                """,
                (gate_id, digest),
            )
        }
        return tuple(stage for stage in SOAK_STAGE_ORDER if stage in completed)

    def soak_profiling_allowed(
        self, gate_id: str, invariant_snapshot_digest: str
    ) -> bool:
        return (
            self.completed_soak_stages(gate_id, invariant_snapshot_digest)
            == SOAK_STAGE_ORDER
        )

    def start_campaign(self, campaign_id: str) -> dict[str, Any]:
        return self._change_status(
            campaign_id,
            expected={CampaignStatus.CREATED.value, CampaignStatus.PAUSED_OPERATOR.value},
            target=CampaignStatus.RUNNING.value,
            reason=None,
        )

    def pause_campaign(
        self, campaign_id: str, *, status: str, reason: str
    ) -> dict[str, Any]:
        target = CampaignStatus(status)
        allowed = {
            CampaignStatus.PAUSED_OPERATOR,
            CampaignStatus.PAUSED_BUDGET,
            CampaignStatus.PAUSED_UNKNOWN_OUTCOME,
            CampaignStatus.PAUSED_HARD_FAILURE,
            CampaignStatus.PAUSED_DATA_INTEGRITY,
        }
        if target not in allowed:
            raise ValueError("target is not a paused campaign status")
        expected = {CampaignStatus.RUNNING.value}
        if target is CampaignStatus.PAUSED_DATA_INTEGRITY:
            # Integrity failures dominate operational pauses.  Once evidence
            # is inconsistent the campaign must not remain resumable merely
            # because it had already paused for promotion, budget, or GPU
            # diagnosis.
            expected.update(status.value for status in allowed)
        return self._change_status(
            campaign_id,
            expected=expected,
            target=target.value,
            reason=_token(reason, "reason", maximum=2000),
        )

    def resume_campaign(
        self, campaign_id: str, *, doctor_evidence_digest: str | None = None
    ) -> dict[str, Any]:
        current = self.get_campaign(campaign_id)
        if current["status"] == CampaignStatus.PAUSED_DATA_INTEGRITY.value:
            raise ValueError("data-integrity pauses require restore into a new runtime")
        if current["status"] in {
            CampaignStatus.PAUSED_HARD_FAILURE.value,
            CampaignStatus.PAUSED_UNKNOWN_OUTCOME.value,
        }:
            if doctor_evidence_digest is None:
                raise ValueError("GPU pause requires fresh doctor evidence")
            _digest(doctor_evidence_digest, "doctor_evidence_digest")
        elif current["status"] not in {
            CampaignStatus.PAUSED_OPERATOR.value,
            CampaignStatus.PAUSED_BUDGET.value,
        }:
            raise ValueError(f"campaign cannot resume from {current['status']}")
        with self._transaction():
            authoritative = self.connection.execute(
                "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
            ).fetchone()
            if authoritative is None:
                raise KeyError(f"unknown campaign: {campaign_id}")
            if authoritative["status"] != current["status"]:
                raise RuntimeError("campaign resume compare-and-swap failed")
            if doctor_evidence_digest is not None:
                evidence_row = self.connection.execute(
                    """
                    SELECT * FROM campaign_doctor_evidence
                    WHERE digest = ? AND campaign_id = ?
                    """,
                    (doctor_evidence_digest, campaign_id),
                ).fetchone()
                if evidence_row is None:
                    raise ValueError(
                        "GPU pause requires trusted doctor evidence recorded by this Campaign"
                    )
                evidence = self._row(evidence_row)["evidence"]
                if (
                    evidence.get("kind") != "CAMPAIGN_RESUME_DOCTOR"
                    or evidence.get("status") != "SUCCESS"
                    or evidence.get("campaign_id") != campaign_id
                    or evidence.get("namespace_id") != authoritative["namespace_id"]
                    or evidence.get("resource_id") != evidence_row["resource_id"]
                ):
                    raise ValueError("trusted doctor evidence identity is invalid")
                quarantine_epoch = evidence.get("quarantine_fencing_epoch")
                if (
                    type(quarantine_epoch) is not int
                    or quarantine_epoch <= 0
                    or quarantine_epoch
                    != evidence_row["quarantine_fencing_epoch"]
                ):
                    raise ValueError(
                        "trusted doctor evidence has no quarantine fencing identity"
                    )
                latest_quarantine = self.connection.execute(
                    """
                    SELECT resource_id, fencing_epoch FROM resource_leases
                    WHERE campaign_id = ? AND status = 'QUARANTINED'
                    ORDER BY rowid DESC LIMIT 1
                    """,
                    (campaign_id,),
                ).fetchone()
                if (
                    latest_quarantine is None
                    or latest_quarantine["resource_id"]
                    != evidence.get("resource_id")
                    or int(latest_quarantine["fencing_epoch"])
                    != quarantine_epoch
                ):
                    raise ValueError(
                        "trusted doctor evidence does not bind the latest Campaign quarantine"
                    )
                environment = ExecutionEnvironmentDigest.from_value(
                    evidence.get("execution_environment")
                )
                if not environment.is_resolved:
                    raise ValueError(
                        "trusted doctor evidence must bind a resolved environment"
                    )
                if environment.digest != evidence_row["execution_environment_digest"]:
                    raise ValueError(
                        "trusted doctor evidence environment column is inconsistent"
                    )
                if float(evidence_row["observed_epoch"]) < _utc_epoch(
                    authoritative["updated_at"], "campaign pause timestamp"
                ):
                    raise ValueError("trusted doctor evidence predates the GPU pause")
                active_ref_row = self.connection.execute(
                    """
                    SELECT baseline_ref_json FROM baseline_revisions
                    WHERE id = ? AND campaign_id = ?
                    """,
                    (authoritative["active_baseline_revision_id"], campaign_id),
                ).fetchone()
                if active_ref_row is None:
                    raise ValueError("Campaign active baseline revision is missing")
                if active_ref_row["baseline_ref_json"] is not None:
                    active_ref = _baseline_ref(
                        json.loads(active_ref_row["baseline_ref_json"]),
                        "active baseline reference",
                    )
                    if active_ref.execution_environment.is_resolved:
                        active_ref.require_environment(environment)
            updated = self.connection.execute(
                """
                UPDATE campaigns
                SET status = 'RUNNING', updated_at = ?, stop_reason = NULL,
                    pause_evidence_digest = ?
                WHERE id = ? AND status = ?
                """,
                (
                    _utc_now(),
                    doctor_evidence_digest,
                    campaign_id,
                    authoritative["status"],
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("campaign resume compare-and-swap failed")
        return self.get_campaign(campaign_id)

    def record_doctor_evidence(
        self, campaign_id: str, *, evidence: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Persist one canonical SUCCESS result from the trusted doctor path."""

        _token(campaign_id, "campaign_id")
        if not isinstance(evidence, Mapping):
            raise TypeError("evidence must be a mapping")
        value = json.loads(_json(dict(evidence)))
        expected_fields = {
            "schema_version",
            "kind",
            "status",
            "campaign_id",
            "resource_id",
            "observed_epoch",
            "namespace_id",
            "config_digest",
            "execution_environment",
            "doctor_result",
        }
        if set(value) != expected_fields or value.get("schema_version") != 1:
            raise ValueError("trusted doctor evidence fields do not match schema")
        if (
            value.get("kind") != "CAMPAIGN_RESUME_DOCTOR"
            or value.get("status") != "SUCCESS"
            or value.get("campaign_id") != campaign_id
        ):
            raise ValueError("trusted doctor evidence is not a successful Campaign result")
        resource_id = _token(value.get("resource_id"), "resource_id")
        observed_epoch = _finite_epoch(
            value.get("observed_epoch"), "observed_epoch"
        )
        namespace_id = _digest(value.get("namespace_id"), "namespace_id")
        _digest(value.get("config_digest"), "config_digest")
        environment = ExecutionEnvironmentDigest.from_value(
            value.get("execution_environment")
        )
        if not environment.is_resolved:
            raise ValueError("trusted doctor evidence must bind a resolved environment")
        doctor = value.get("doctor_result")
        probe = doctor.get("c500_probe") if isinstance(doctor, Mapping) else None
        environment_value = (
            probe.get("environment") if isinstance(probe, Mapping) else None
        )
        if not (
            isinstance(doctor, Mapping)
            and doctor.get("status") == "SUCCESS"
            and isinstance(environment_value, Mapping)
            and environment_value.get("compile_probe_status") == "PASSED"
        ):
            raise ValueError("trusted doctor result did not pass the C500 compile probe")
        campaign = self.get_campaign(campaign_id)
        if campaign["status"] not in {
            CampaignStatus.PAUSED_HARD_FAILURE.value,
            CampaignStatus.PAUSED_UNKNOWN_OUTCOME.value,
        }:
            raise ValueError("trusted GPU doctor evidence requires a GPU-paused Campaign")
        if campaign["namespace_id"] != namespace_id:
            raise ValueError("trusted doctor evidence belongs to another namespace")
        now = _utc_now()
        with self._transaction():
            authoritative = self.connection.execute(
                "SELECT status, namespace_id FROM campaigns WHERE id = ?",
                (campaign_id,),
            ).fetchone()
            if (
                authoritative is None
                or authoritative["status"] != campaign["status"]
                or authoritative["namespace_id"] != namespace_id
            ):
                raise RuntimeError(
                    "Campaign changed while doctor evidence was being recorded"
                )
            latest_quarantine = self.connection.execute(
                """
                SELECT resource_id, fencing_epoch FROM resource_leases
                WHERE campaign_id = ? AND status = 'QUARANTINED'
                ORDER BY rowid DESC LIMIT 1
                """,
                (campaign_id,),
            ).fetchone()
            if (
                latest_quarantine is None
                or latest_quarantine["resource_id"] != resource_id
            ):
                raise ValueError(
                    "trusted doctor resource does not match the latest Campaign quarantine"
                )
            value["quarantine_fencing_epoch"] = int(
                latest_quarantine["fencing_epoch"]
            )
            digest = _canonical_digest(value)
            existing = self.connection.execute(
                "SELECT * FROM campaign_doctor_evidence WHERE digest = ?",
                (digest,),
            ).fetchone()
            if existing is None:
                self.connection.execute(
                    """
                    INSERT INTO campaign_doctor_evidence(
                        digest, campaign_id, resource_id, observed_epoch,
                        quarantine_fencing_epoch,
                        execution_environment_digest, evidence_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        digest,
                        campaign_id,
                        resource_id,
                        observed_epoch,
                        value["quarantine_fencing_epoch"],
                        environment.digest,
                        _json(value),
                        now,
                    ),
                )
            else:
                current = self._row(existing)
                if current["evidence"] != value:
                    raise ValueError("doctor evidence digest collision")
        return self.get_doctor_evidence(digest)

    def get_doctor_evidence(self, digest: str) -> dict[str, Any]:
        _digest(digest, "doctor evidence digest")
        row = self.connection.execute(
            "SELECT * FROM campaign_doctor_evidence WHERE digest = ?", (digest,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown Campaign doctor evidence: {digest}")
        return self._row(row)

    def finish_campaign(
        self, campaign_id: str, *, cancelled: bool = False, reason: str = ""
    ) -> dict[str, Any]:
        campaign = self.get_campaign(campaign_id)
        if campaign["status"] in TERMINAL_CAMPAIGN_STATUSES:
            return campaign
        active = self.connection.execute(
            """
            SELECT COUNT(*) FROM child_runs
            WHERE campaign_id = ? AND status IN ('PENDING', 'RUNNING')
            """,
            (campaign_id,),
        ).fetchone()[0]
        if active:
            raise ValueError("campaign has an active child run")
        return self._change_status(
            campaign_id,
            expected={campaign["status"]},
            target=(
                CampaignStatus.CANCELLED.value
                if cancelled
                else CampaignStatus.COMPLETED.value
            ),
            reason=_token(reason, "reason", maximum=2000) if reason else None,
        )

    def _change_status(
        self,
        campaign_id: str,
        *,
        expected: set[str],
        target: str,
        reason: str | None,
    ) -> dict[str, Any]:
        now = _utc_now()
        with self._transaction():
            placeholders = ",".join("?" for _ in expected)
            updated = self.connection.execute(
                f"""
                UPDATE campaigns SET status = ?, updated_at = ?, stop_reason = ?
                WHERE id = ? AND status IN ({placeholders})
                """,
                (target, now, reason, campaign_id, *sorted(expected)),
            )
            if updated.rowcount != 1:
                row = self.connection.execute(
                    "SELECT status FROM campaigns WHERE id = ?", (campaign_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown campaign: {campaign_id}")
                if row["status"] == target:
                    return self.get_campaign(campaign_id)
                raise ValueError(f"campaign cannot transition {row['status']} -> {target}")
        return self.get_campaign(campaign_id)

    def reserve_budget(
        self,
        campaign_id: str,
        *,
        idempotency_key: str,
        action_kind: str,
        amount: BudgetAmount,
    ) -> dict[str, Any]:
        _token(idempotency_key, "idempotency_key")
        _token(action_kind, "action_kind")
        if not any(amount.to_dict().values()):
            raise ValueError("budget reservation must reserve at least one unit")
        with self._transaction():
            existing = self.connection.execute(
                """
                SELECT * FROM budget_actions
                WHERE campaign_id = ? AND idempotency_key = ?
                """,
                (campaign_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                result = self._budget_action(existing)
                if result["action_kind"] != action_kind or result["reserved"] != amount.to_dict():
                    raise ValueError("idempotency key was reused with different budget intent")
                return result
            campaign = self.connection.execute(
                "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
            ).fetchone()
            if campaign is None:
                raise KeyError(f"unknown campaign: {campaign_id}")
            if campaign["status"] != CampaignStatus.RUNNING.value:
                raise ValueError("budget may only be reserved by a running campaign")
            used = self._budget_used(campaign_id)
            projected = used + amount
            limit = BudgetAmount(
                candidates=campaign["max_candidates"],
                wall_ms=campaign["max_wall_ms"],
                gpu_ms=campaign["max_gpu_ms"],
                tokens=campaign["max_tokens"],
                cost_microusd=campaign["max_cost_microusd"],
            )
            if not projected.fits_within(limit):
                raise ValueError("campaign budget reservation exceeds a frozen limit")
            now = _utc_now()
            cursor = self.connection.execute(
                """
                INSERT INTO budget_actions(
                    campaign_id, idempotency_key, action_kind, status,
                    reserved_candidates, reserved_wall_ms, reserved_gpu_ms,
                    reserved_tokens, reserved_cost_microusd, created_at
                ) VALUES (?, ?, ?, 'RESERVED', ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    idempotency_key,
                    action_kind,
                    amount.candidates,
                    amount.wall_ms,
                    amount.gpu_ms,
                    amount.tokens,
                    amount.cost_microusd,
                    now,
                ),
            )
            row = self.connection.execute(
                "SELECT * FROM budget_actions WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        assert row is not None
        return self._budget_action(row)

    def get_budget_action(
        self, campaign_id: str, *, idempotency_key: str
    ) -> dict[str, Any]:
        """Read one immutable budget intent for trusted runner reconciliation."""

        _token(campaign_id, "campaign_id")
        _token(idempotency_key, "idempotency_key")
        row = self.connection.execute(
            """
            SELECT * FROM budget_actions
            WHERE campaign_id = ? AND idempotency_key = ?
            """,
            (campaign_id, idempotency_key),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown budget action: {idempotency_key}")
        return self._budget_action(row)

    def settle_budget(
        self,
        campaign_id: str,
        *,
        idempotency_key: str,
        actual: BudgetAmount,
    ) -> dict[str, Any]:
        with self._transaction():
            row = self.connection.execute(
                """
                SELECT * FROM budget_actions
                WHERE campaign_id = ? AND idempotency_key = ?
                """,
                (campaign_id, idempotency_key),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown budget action: {idempotency_key}")
            current = self._budget_action(row)
            if current["status"] == "SETTLED":
                if current["actual"] != actual.to_dict():
                    raise ValueError("settled budget action is immutable")
                return current
            if current["status"] != "RESERVED":
                raise ValueError("cancelled budget action cannot be settled")
            if not actual.fits_within(BudgetAmount.from_mapping(current["reserved"])):
                raise ValueError("actual budget exceeds worst-case reservation")
            self.connection.execute(
                """
                UPDATE budget_actions
                SET status = 'SETTLED', actual_candidates = ?, actual_wall_ms = ?,
                    actual_gpu_ms = ?, actual_tokens = ?, actual_cost_microusd = ?,
                    settled_at = ?
                WHERE id = ? AND status = 'RESERVED'
                """,
                (*actual.to_dict().values(), _utc_now(), row["id"]),
            )
            updated = self.connection.execute(
                "SELECT * FROM budget_actions WHERE id = ?", (row["id"],)
            ).fetchone()
        assert updated is not None
        return self._budget_action(updated)

    def cancel_budget(self, campaign_id: str, *, idempotency_key: str) -> dict[str, Any]:
        with self._transaction():
            row = self.connection.execute(
                "SELECT * FROM budget_actions WHERE campaign_id = ? AND idempotency_key = ?",
                (campaign_id, idempotency_key),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown budget action: {idempotency_key}")
            if row["status"] == "SETTLED":
                raise ValueError("settled budget action cannot be cancelled")
            if row["status"] == "RESERVED":
                self.connection.execute(
                    """
                    UPDATE budget_actions SET status = 'CANCELLED', settled_at = ?
                    WHERE id = ?
                    """,
                    (_utc_now(), row["id"]),
                )
            updated = self.connection.execute(
                "SELECT * FROM budget_actions WHERE id = ?", (row["id"],)
            ).fetchone()
        assert updated is not None
        return self._budget_action(updated)

    @staticmethod
    def _budget_action(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["reserved"] = {
            name: result.pop(f"reserved_{name}")
            for name in BudgetAmount().to_dict()
        }
        actual_values = {
            name: result.pop(f"actual_{name}")
            for name in BudgetAmount().to_dict()
        }
        result["actual"] = (
            actual_values if all(value is not None for value in actual_values.values()) else None
        )
        return result

    def _budget_used(self, campaign_id: str) -> BudgetAmount:
        columns = []
        for name in BudgetAmount().to_dict():
            columns.append(
                f"COALESCE(SUM(CASE WHEN status = 'RESERVED' THEN reserved_{name} "
                f"WHEN status = 'SETTLED' THEN actual_{name} ELSE 0 END), 0) AS {name}"
            )
        row = self.connection.execute(
            "SELECT " + ", ".join(columns) + " FROM budget_actions WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchone()
        assert row is not None
        return BudgetAmount(**{name: int(row[name]) for name in BudgetAmount().to_dict()})

    def budget_status(self, campaign_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown campaign: {campaign_id}")
        used = self._budget_used(campaign_id)
        limit = BudgetAmount(
            candidates=row["max_candidates"],
            wall_ms=row["max_wall_ms"],
            gpu_ms=row["max_gpu_ms"],
            tokens=row["max_tokens"],
            cost_microusd=row["max_cost_microusd"],
        )
        return {
            "limit": limit.to_dict(),
            "committed_or_reserved": used.to_dict(),
            "remaining": {
                name: getattr(limit, name) - getattr(used, name)
                for name in limit.to_dict()
            },
        }

    def create_child_run(
        self,
        campaign_id: str,
        *,
        proposer_profile: Mapping[str, Any],
        controller_run_id: str | None = None,
        max_candidates: int = MAX_CHILD_CANDIDATES,
        max_wall_seconds: int = MAX_CHILD_WALL_SECONDS,
        max_consecutive_failures: int = MAX_CHILD_CONSECUTIVE_FAILURES,
        stop_after_promotion: bool = True,
    ) -> dict[str, Any]:
        if not 1 <= max_candidates <= MAX_CHILD_CANDIDATES:
            raise ValueError("child run candidate limit must be between 1 and 5")
        if not 1 <= max_wall_seconds <= MAX_CHILD_WALL_SECONDS:
            raise ValueError("child run wall limit must be at most six hours")
        if not 1 <= max_consecutive_failures <= MAX_CHILD_CONSECUTIVE_FAILURES:
            raise ValueError("child run failure limit must be between 1 and 3")
        if stop_after_promotion is not True:
            raise ValueError("campaign child runs must stop after promotion")
        if controller_run_id is not None:
            _token(controller_run_id, "controller_run_id")
        proposer_value = dict(proposer_profile)
        with self._transaction():
            if controller_run_id is not None:
                existing = self.connection.execute(
                    "SELECT * FROM child_runs WHERE controller_run_id = ?",
                    (controller_run_id,),
                ).fetchone()
                if existing is not None:
                    result = self._row(existing)
                    expected = {
                        "campaign_id": campaign_id,
                        "proposer_profile": proposer_value,
                        "max_candidates": max_candidates,
                        "max_wall_seconds": max_wall_seconds,
                        "max_consecutive_failures": max_consecutive_failures,
                        "stop_after_promotion": True,
                    }
                    if any(result[key] != value for key, value in expected.items()):
                        raise ValueError(
                            "controller_run_id was reused with a different child intent"
                        )
                    return result
            campaign = self.connection.execute(
                "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
            ).fetchone()
            if campaign is None:
                raise KeyError(f"unknown campaign: {campaign_id}")
            if campaign["status"] != CampaignStatus.RUNNING.value:
                raise ValueError("child run requires a running campaign")
            child_index = int(
                self.connection.execute(
                    "SELECT COALESCE(MAX(child_index), 0) + 1 FROM child_runs WHERE campaign_id = ?",
                    (campaign_id,),
                ).fetchone()[0]
            )
            cursor = self.connection.execute(
                """
                INSERT INTO child_runs(
                    campaign_id, child_index, controller_run_id, status,
                    baseline_revision_id,
                    proposer_profile_json, max_candidates, max_wall_seconds,
                    max_consecutive_failures, stop_after_promotion, created_at
                ) VALUES (?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    campaign_id,
                    child_index,
                    controller_run_id,
                    campaign["active_baseline_revision_id"],
                    _json(proposer_value),
                    max_candidates,
                    max_wall_seconds,
                    max_consecutive_failures,
                    _utc_now(),
                ),
            )
            row = self.connection.execute(
                "SELECT * FROM child_runs WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
        assert row is not None
        return self._row(row)

    def start_child_run(self, child_id: int, *, controller_run_id: str) -> dict[str, Any]:
        _token(controller_run_id, "controller_run_id")
        with self._transaction():
            changed = self.connection.execute(
                """
                UPDATE child_runs SET status = 'RUNNING', controller_run_id = ?, started_at = ?
                WHERE id = ? AND status = 'PENDING'
                """,
                (controller_run_id, _utc_now(), child_id),
            )
            if changed.rowcount != 1:
                row = self.connection.execute(
                    "SELECT * FROM child_runs WHERE id = ?", (child_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown child run: {child_id}")
                if row["status"] == "RUNNING" and row["controller_run_id"] == controller_run_id:
                    return self._row(row)
                raise ValueError("child run cannot be started from its current state")
        return self.get_child_run(child_id)

    def finish_child_run(
        self, child_id: int, *, status: str, result: Mapping[str, Any]
    ) -> dict[str, Any]:
        target = ChildRunStatus(status)
        if target.value not in TERMINAL_CHILD_STATUSES:
            raise ValueError("child finish status must be terminal")
        now = _utc_now()
        with self._transaction():
            row = self.connection.execute(
                "SELECT * FROM child_runs WHERE id = ?", (child_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown child run: {child_id}")
            if row["status"] in TERMINAL_CHILD_STATUSES:
                if row["status"] != target.value or json.loads(row["result_json"]) != dict(result):
                    raise ValueError("terminal child run is immutable")
                return self._row(row)
            if row["status"] != ChildRunStatus.RUNNING.value:
                raise ValueError("only a running child may finish")
            self.connection.execute(
                """
                UPDATE child_runs SET status = ?, result_json = ?, finished_at = ?
                WHERE id = ? AND status = 'RUNNING'
                """,
                (target.value, _json(dict(result)), now, child_id),
            )
            pause_status = None
            reason = None
            if target is ChildRunStatus.HARD_FAILED:
                pause_status = CampaignStatus.PAUSED_HARD_FAILURE.value
                reason = "child run reported a hard GPU failure"
            elif target is ChildRunStatus.UNKNOWN_GPU_OUTCOME:
                pause_status = CampaignStatus.PAUSED_UNKNOWN_OUTCOME.value
                reason = "child run GPU outcome is unknown"
            elif target is ChildRunStatus.PROMOTED:
                pause_status = CampaignStatus.PAUSED_OPERATOR.value
                reason = "promotion awaits checkpoint and staged-lineage CAS"
            if pause_status is not None:
                self.connection.execute(
                    """
                    UPDATE campaigns SET status = ?, stop_reason = ?, updated_at = ?,
                        pause_evidence_digest = NULL
                    WHERE id = ? AND status = 'RUNNING'
                    """,
                    (pause_status, reason, now, row["campaign_id"]),
                )
        return self.get_child_run(child_id)

    def get_child_run(self, child_id: int) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM child_runs WHERE id = ?", (child_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown child run: {child_id}")
        return self._row(row)

    def get_child_run_by_controller_run_id(
        self, controller_run_id: str
    ) -> dict[str, Any] | None:
        """Return the durable child intent used to reconcile a runner action."""

        _token(controller_run_id, "controller_run_id")
        row = self.connection.execute(
            "SELECT * FROM child_runs WHERE controller_run_id = ?",
            (controller_run_id,),
        ).fetchone()
        return None if row is None else self._row(row)

    def list_child_runs(self, campaign_id: str) -> list[dict[str, Any]]:
        return [
            self._row(row)
            for row in self.connection.execute(
                "SELECT * FROM child_runs WHERE campaign_id = ? ORDER BY child_index",
                (campaign_id,),
            )
        ]

    def acquire_resource(
        self,
        campaign_id: str,
        *,
        resource_id: str,
        ttl_seconds: float,
        now_epoch: float | None = None,
    ) -> ResourceLease:
        _token(resource_id, "resource_id")
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be finite and positive")
        now_value = time.time() if now_epoch is None else float(now_epoch)
        with self._transaction():
            campaign = self.connection.execute(
                "SELECT status, pause_evidence_digest FROM campaigns WHERE id = ?",
                (campaign_id,),
            ).fetchone()
            if campaign is None:
                raise KeyError(f"unknown campaign: {campaign_id}")
            if campaign["status"] != CampaignStatus.RUNNING.value:
                raise ValueError("resource lease requires a running campaign")
            latest = self.connection.execute(
                """
                SELECT * FROM resource_leases
                WHERE resource_id = ?
                ORDER BY fencing_epoch DESC LIMIT 1
                """,
                (resource_id,),
            ).fetchone()
            if latest is not None and latest["status"] == "QUARANTINED":
                doctor_row = None
                if campaign["pause_evidence_digest"] is not None:
                    doctor_row = self.connection.execute(
                        """
                        SELECT resource_id, quarantine_fencing_epoch
                        FROM campaign_doctor_evidence
                        WHERE digest = ? AND campaign_id = ?
                        """,
                        (campaign["pause_evidence_digest"], campaign_id),
                    ).fetchone()
                cleared_by_doctor = bool(
                    latest["campaign_id"] == campaign_id
                    and doctor_row is not None
                    and doctor_row["resource_id"] == resource_id
                    and doctor_row["quarantine_fencing_epoch"] is not None
                    and int(doctor_row["quarantine_fencing_epoch"])
                    == int(latest["fencing_epoch"])
                )
                if not cleared_by_doctor:
                    raise ValueError(
                        "resource is quarantined and requires trusted doctor clearance"
                    )
            active = self.connection.execute(
                """
                SELECT * FROM resource_leases
                WHERE resource_id = ? AND status = 'ACTIVE'
                """,
                (resource_id,),
            ).fetchone()
            if active is not None and float(active["expires_epoch"]) > now_value:
                raise ValueError("resource already has an active unexpired lease")
            if active is not None:
                self.connection.execute(
                    """
                    UPDATE resource_leases SET status = 'EXPIRED', released_at = ?,
                        reason = 'lease TTL expired'
                    WHERE resource_id = ? AND fencing_epoch = ? AND status = 'ACTIVE'
                    """,
                    (_utc_now(), resource_id, active["fencing_epoch"]),
                )
            epoch = int(
                self.connection.execute(
                    "SELECT COALESCE(MAX(fencing_epoch), 0) + 1 FROM resource_leases WHERE resource_id = ?",
                    (resource_id,),
                ).fetchone()[0]
            )
            expires = now_value + ttl_seconds
            self.connection.execute(
                """
                INSERT INTO resource_leases(
                    resource_id, fencing_epoch, campaign_id, status,
                    acquired_at, expires_epoch
                ) VALUES (?, ?, ?, 'ACTIVE', ?, ?)
                """,
                (resource_id, epoch, campaign_id, _utc_now(), expires),
            )
        return ResourceLease(resource_id, campaign_id, epoch, expires, "ACTIVE")

    def get_active_resource_lease(self, resource_id: str) -> ResourceLease | None:
        """Return the current fencing token without renewing or acquiring it."""

        _token(resource_id, "resource_id")
        row = self.connection.execute(
            """
            SELECT * FROM resource_leases
            WHERE resource_id = ? AND status = 'ACTIVE'
            """,
            (resource_id,),
        ).fetchone()
        if row is None:
            return None
        return ResourceLease(
            resource_id=row["resource_id"],
            campaign_id=row["campaign_id"],
            fencing_epoch=int(row["fencing_epoch"]),
            expires_epoch=float(row["expires_epoch"]),
            status=row["status"],
        )

    def get_latest_quarantined_resource(
        self, campaign_id: str
    ) -> ResourceLease | None:
        """Return the Campaign's newest quarantine without inferring ownership."""

        _token(campaign_id, "campaign_id")
        row = self.connection.execute(
            """
            SELECT * FROM resource_leases
            WHERE campaign_id = ? AND status = 'QUARANTINED'
            ORDER BY rowid DESC LIMIT 1
            """,
            (campaign_id,),
        ).fetchone()
        if row is None:
            return None
        return ResourceLease(
            resource_id=row["resource_id"],
            campaign_id=row["campaign_id"],
            fencing_epoch=int(row["fencing_epoch"]),
            expires_epoch=float(row["expires_epoch"]),
            status=row["status"],
        )

    def renew_resource(
        self,
        lease: ResourceLease,
        *,
        ttl_seconds: float,
        now_epoch: float | None = None,
    ) -> ResourceLease:
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be finite and positive")
        now_value = time.time() if now_epoch is None else float(now_epoch)
        expires = now_value + ttl_seconds
        with self._transaction():
            changed = self.connection.execute(
                """
                UPDATE resource_leases SET expires_epoch = ?
                WHERE resource_id = ? AND fencing_epoch = ? AND campaign_id = ?
                  AND status = 'ACTIVE' AND expires_epoch > ?
                """,
                (
                    expires,
                    lease.resource_id,
                    lease.fencing_epoch,
                    lease.campaign_id,
                    now_value,
                ),
            )
            if changed.rowcount != 1:
                raise ValueError("lease is stale, expired, released, or not owned")
        return ResourceLease(
            lease.resource_id, lease.campaign_id, lease.fencing_epoch, expires, "ACTIVE"
        )

    def release_resource(
        self, lease: ResourceLease, *, quarantine: bool = False, reason: str = ""
    ) -> ResourceLease:
        target = "QUARANTINED" if quarantine else "RELEASED"
        if quarantine and not reason:
            raise ValueError("quarantine requires a reason")
        with self._transaction():
            changed = self.connection.execute(
                """
                UPDATE resource_leases SET status = ?, released_at = ?, reason = ?
                WHERE resource_id = ? AND fencing_epoch = ? AND campaign_id = ?
                  AND status = 'ACTIVE'
                """,
                (
                    target,
                    _utc_now(),
                    _token(reason, "reason", maximum=2000) if reason else None,
                    lease.resource_id,
                    lease.fencing_epoch,
                    lease.campaign_id,
                ),
            )
            if changed.rowcount != 1:
                row = self.connection.execute(
                    """
                    SELECT status, expires_epoch FROM resource_leases
                    WHERE resource_id = ? AND fencing_epoch = ? AND campaign_id = ?
                    """,
                    (lease.resource_id, lease.fencing_epoch, lease.campaign_id),
                ).fetchone()
                if row is not None and row["status"] == target:
                    return ResourceLease(
                        lease.resource_id,
                        lease.campaign_id,
                        lease.fencing_epoch,
                        float(row["expires_epoch"]),
                        target,
                    )
                raise ValueError("lease is stale, released, or not owned")
        return ResourceLease(
            lease.resource_id,
            lease.campaign_id,
            lease.fencing_epoch,
            lease.expires_epoch,
            target,
        )

    def list_baseline_revisions(self, campaign_id: str) -> list[dict[str, Any]]:
        return [
            self._row(row)
            for row in self.connection.execute(
                """
                SELECT * FROM baseline_revisions
                WHERE campaign_id = ? ORDER BY revision_index
                """,
                (campaign_id,),
            )
        ]

    def advance_baseline(
        self,
        campaign_id: str,
        *,
        expected_parent_revision_id: str,
        parent_baseline_ref: Mapping[str, Any] | BaselineRef,
        artifact_id: str,
        primary_experiment_uid: str,
        confirmation_experiment_uid: str,
        evidence_namespace_id: str,
        evidence_execution_environment: Mapping[str, Any]
        | ExecutionEnvironmentDigest,
        policy_snapshot: Mapping[str, Any],
        idempotency_key: str,
    ) -> dict[str, Any]:
        _artifact_id(artifact_id)
        _token(primary_experiment_uid, "primary_experiment_uid")
        _token(confirmation_experiment_uid, "confirmation_experiment_uid")
        if primary_experiment_uid == confirmation_experiment_uid:
            raise ValueError("primary and confirmation experiments must differ")
        _digest(evidence_namespace_id, "evidence_namespace_id")
        _token(idempotency_key, "idempotency_key")
        selected_parent_ref = _baseline_ref(
            parent_baseline_ref, "parent_baseline_ref"
        )
        try:
            selected_environment = (
                evidence_execution_environment
                if isinstance(
                    evidence_execution_environment, ExecutionEnvironmentDigest
                )
                else ExecutionEnvironmentDigest.from_value(
                    dict(evidence_execution_environment)
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "evidence_execution_environment must be a resolved identity"
            ) from exc
        if not selected_parent_ref.is_scientifically_comparable:
            raise ValueError(
                "LEGACY_UNKNOWN baseline evidence cannot advance staged lineage"
            )
        selected_parent_ref.require_environment(selected_environment)
        if selected_parent_ref.namespace_id != evidence_namespace_id:
            raise ValueError("parent baseline belongs to another namespace")
        now = _utc_now()
        with self._transaction():
            existing = self.connection.execute(
                """
                SELECT * FROM baseline_revisions
                WHERE campaign_id = ? AND idempotency_key = ?
                """,
                (campaign_id, idempotency_key),
            ).fetchone()
            if existing is not None:
                result = self._row(existing)
                expected = {
                    "parent_revision_id": expected_parent_revision_id,
                    "artifact_id": artifact_id,
                    "primary_experiment_uid": primary_experiment_uid,
                    "confirmation_experiment_uid": confirmation_experiment_uid,
                    "namespace_id": evidence_namespace_id,
                    "policy_snapshot": dict(policy_snapshot),
                }
                if any(result[key] != value for key, value in expected.items()):
                    raise ValueError("lineage idempotency key was reused with different evidence")
                parent_row = self.connection.execute(
                    """
                    SELECT baseline_ref_json FROM baseline_revisions
                    WHERE id = ? AND campaign_id = ?
                    """,
                    (expected_parent_revision_id, campaign_id),
                ).fetchone()
                if parent_row is None or parent_row["baseline_ref_json"] is None:
                    raise ValueError(
                        "lineage idempotency parent has no environment identity"
                    )
                if _baseline_ref(
                    json.loads(parent_row["baseline_ref_json"]),
                    "idempotency parent baseline reference",
                ) != selected_parent_ref:
                    raise ValueError(
                        "lineage idempotency parent differs from frozen evidence"
                    )
                result_ref = _baseline_ref(
                    result.get("baseline_ref"), "staged baseline reference"
                )
                result_ref.require_environment(selected_environment)
                return result
            campaign = self.connection.execute(
                "SELECT * FROM campaigns WHERE id = ?", (campaign_id,)
            ).fetchone()
            if campaign is None:
                raise KeyError(f"unknown campaign: {campaign_id}")
            if campaign["mode"] != CampaignMode.DISCOVERY.value:
                raise ValueError("benchmark campaigns cannot advance a baseline")
            if not bool(campaign["allow_staged_lineage"]):
                raise ValueError("campaign staged lineage is not enabled")
            if campaign["status"] not in {
                CampaignStatus.RUNNING.value,
                CampaignStatus.PAUSED_OPERATOR.value,
            }:
                raise ValueError("campaign state does not allow lineage advancement")
            if campaign["namespace_id"] != evidence_namespace_id:
                raise ValueError("promotion evidence belongs to another namespace")
            if campaign["active_baseline_revision_id"] != expected_parent_revision_id:
                raise ValueError("baseline parent compare-and-swap failed")
            parent = self.connection.execute(
                """
                SELECT * FROM baseline_revisions
                WHERE id = ? AND campaign_id = ?
                """,
                (expected_parent_revision_id, campaign_id),
            ).fetchone()
            if parent is None:
                raise ValueError("baseline parent does not belong to campaign")
            if parent["baseline_ref_json"] is None:
                raise ValueError(
                    "legacy Campaign revision has no execution environment and cannot advance"
                )
            stored_parent_ref = _baseline_ref(
                json.loads(parent["baseline_ref_json"]),
                "stored parent baseline reference",
            )
            if stored_parent_ref != selected_parent_ref:
                raise ValueError(
                    "promotion evidence parent differs from the frozen Campaign baseline"
                )
            if (
                stored_parent_ref.namespace_id != parent["namespace_id"]
                or str(stored_parent_ref.artifact_id) != parent["artifact_id"]
            ):
                raise ValueError(
                    "stored Campaign baseline scalar identity is inconsistent"
                )
            stored_parent_ref.require_environment(selected_environment)
            revision_id = "baseline-" + uuid.uuid4().hex
            staged_ref = BaselineRef.create(
                namespace=evidence_namespace_id,
                artifact_id=artifact_id,
                source="campaign",
                revision=revision_id,
                execution_environment=selected_environment,
            )
            self.connection.execute(
                """
                INSERT INTO baseline_revisions(
                    id, campaign_id, revision_index, namespace_id,
                    parent_revision_id, artifact_id, baseline_ref_json,
                    revision_kind,
                    primary_experiment_uid, confirmation_experiment_uid,
                    policy_snapshot_json, idempotency_key, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'STAGED', ?, ?, ?, ?, ?)
                """,
                (
                    revision_id,
                    campaign_id,
                    int(parent["revision_index"]) + 1,
                    evidence_namespace_id,
                    expected_parent_revision_id,
                    artifact_id,
                    _json(staged_ref.to_dict()),
                    primary_experiment_uid,
                    confirmation_experiment_uid,
                    _json(dict(policy_snapshot)),
                    idempotency_key,
                    now,
                ),
            )
            changed = self.connection.execute(
                """
                UPDATE campaigns
                SET active_baseline_revision_id = ?, updated_at = ?,
                    status = 'RUNNING', stop_reason = NULL
                WHERE id = ? AND active_baseline_revision_id = ?
                  AND namespace_id = ?
                """,
                (
                    revision_id,
                    now,
                    campaign_id,
                    expected_parent_revision_id,
                    evidence_namespace_id,
                ),
            )
            if changed.rowcount != 1:
                raise RuntimeError("baseline pointer compare-and-swap failed")
            self._insert_outbox(
                campaign_id,
                f"baseline-advanced:{campaign_id}:{idempotency_key}",
                "BASELINE_ADVANCED",
                {
                    "revision_id": revision_id,
                    "parent_revision_id": expected_parent_revision_id,
                    "artifact_id": artifact_id,
                },
                now=now,
            )
            row = self.connection.execute(
                "SELECT * FROM baseline_revisions WHERE id = ?", (revision_id,)
            ).fetchone()
        assert row is not None
        return self._row(row)

    def create_oj_nomination(
        self,
        campaign_id: str,
        *,
        nomination_type: str,
        artifact_id: str,
        bundle_object_id: str,
        manifest_digest: str,
        local_metrics: Mapping[str, Any],
        baseline_revision_id: str | None = None,
    ) -> dict[str, Any]:
        if nomination_type not in OJ_TYPES:
            raise ValueError("unknown OJ nomination type")
        _artifact_id(artifact_id)
        _artifact_id(bundle_object_id)
        _digest(manifest_digest, "manifest_digest")
        campaign = self.get_campaign(campaign_id)
        if baseline_revision_id is not None:
            revision = self.connection.execute(
                """
                SELECT id FROM baseline_revisions
                WHERE id = ? AND campaign_id = ? AND namespace_id = ?
                """,
                (baseline_revision_id, campaign_id, campaign["namespace_id"]),
            ).fetchone()
            if revision is None:
                raise ValueError("OJ baseline revision does not belong to campaign namespace")
        nomination_id = "oj-" + uuid.uuid4().hex
        now = _utc_now()
        with self._transaction():
            self.connection.execute(
                """
                INSERT INTO oj_nominations(
                    id, campaign_id, namespace_id, nomination_type, artifact_id,
                    baseline_revision_id, bundle_object_id, manifest_digest,
                    local_metrics_json, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'NOMINATED', ?)
                """,
                (
                    nomination_id,
                    campaign_id,
                    campaign["namespace_id"],
                    nomination_type,
                    artifact_id,
                    baseline_revision_id,
                    bundle_object_id,
                    manifest_digest,
                    _json(dict(local_metrics)),
                    now,
                ),
            )
        return self.get_oj_nomination(nomination_id)

    def mark_oj_exported(self, nomination_id: str) -> dict[str, Any]:
        with self._transaction():
            changed = self.connection.execute(
                """
                UPDATE oj_nominations SET status = 'EXPORTED', exported_at = ?
                WHERE id = ? AND status = 'NOMINATED'
                """,
                (_utc_now(), nomination_id),
            )
            if changed.rowcount != 1:
                row = self.connection.execute(
                    "SELECT status FROM oj_nominations WHERE id = ?", (nomination_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown OJ nomination: {nomination_id}")
                if row["status"] != "EXPORTED":
                    raise ValueError("OJ nomination cannot be exported from its current state")
        return self.get_oj_nomination(nomination_id)

    def record_oj_feedback(
        self,
        nomination_id: str,
        *,
        submission_id: str,
        verdict: str,
        score: float | None,
        note: str = "",
    ) -> dict[str, Any]:
        _token(submission_id, "submission_id")
        if verdict not in OJ_VERDICTS:
            raise ValueError("OJ verdict is not in the trusted whitelist")
        if score is not None and not math.isfinite(float(score)):
            raise ValueError("OJ score must be finite or null")
        clean_note = _token(note, "note", maximum=2000) if note else ""
        with self._transaction():
            row = self.connection.execute(
                "SELECT * FROM oj_nominations WHERE id = ?", (nomination_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown OJ nomination: {nomination_id}")
            if row["status"] == "FEEDBACK_RECORDED":
                existing = self._row(row)
                if (
                    existing["submission_id"] != submission_id
                    or existing["verdict"] != verdict
                    or existing["score"] != score
                    or existing["note"] != clean_note
                ):
                    raise ValueError("recorded OJ feedback is immutable")
                return existing
            if row["status"] not in {"EXPORTED", "SUBMITTED"}:
                raise ValueError("OJ feedback requires an exported nomination")
            self.connection.execute(
                """
                UPDATE oj_nominations
                SET status = 'FEEDBACK_RECORDED', submission_id = ?, verdict = ?,
                    score = ?, note = ?, feedback_at = ?
                WHERE id = ?
                """,
                (
                    submission_id,
                    verdict,
                    None if score is None else float(score),
                    clean_note,
                    _utc_now(),
                    nomination_id,
                ),
            )
        return self.get_oj_nomination(nomination_id)

    def get_oj_nomination(self, nomination_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM oj_nominations WHERE id = ?", (nomination_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown OJ nomination: {nomination_id}")
        return self._row(row)

    def oj_agent_feedback(self, nomination_id: str) -> dict[str, Any]:
        """Return only structured OJ fields safe for proposer feedback.

        The operator note is retained in the audit row but is deliberately
        excluded here because it is arbitrary external text and must never be
        treated as a trusted prompt fragment.
        """

        nomination = self.get_oj_nomination(nomination_id)
        return {
            "nomination_type": nomination["nomination_type"],
            "artifact_id": nomination["artifact_id"],
            "verdict": nomination["verdict"],
            "score": nomination["score"],
        }

    def _insert_outbox(
        self,
        campaign_id: str,
        idempotency_key: str,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        now: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT OR IGNORE INTO outbox(
                campaign_id, idempotency_key, event_type, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                campaign_id,
                idempotency_key,
                event_type,
                _json(dict(payload)),
                now or _utc_now(),
            ),
        )

    def list_outbox(self, *, undelivered_only: bool = True) -> list[dict[str, Any]]:
        query = "SELECT * FROM outbox"
        if undelivered_only:
            query += " WHERE delivered_at IS NULL"
        query += " ORDER BY id"
        return [self._row(row) for row in self.connection.execute(query)]

    def mark_outbox_delivered(self, outbox_id: int) -> dict[str, Any]:
        with self._transaction():
            self.connection.execute(
                "UPDATE outbox SET delivered_at = COALESCE(delivered_at, ?) WHERE id = ?",
                (_utc_now(), outbox_id),
            )
            row = self.connection.execute(
                "SELECT * FROM outbox WHERE id = ?", (outbox_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown outbox event: {outbox_id}")
        return self._row(row)

    def integrity_check(self) -> dict[str, Any]:
        rows = [str(row[0]) for row in self.connection.execute("PRAGMA integrity_check")]
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "OK" if rows == ["ok"] else "FAILED",
            "details": rows,
        }

    def backup_to(self, destination: str | Path) -> None:
        target_path = Path(destination)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(target_path)
        try:
            self.connection.backup(target)
        finally:
            target.close()


__all__ = [
    "ARTIFACT_ID_RE",
    "CampaignStore",
    "OJ_TYPES",
    "OJ_VERDICTS",
    "SCHEMA_VERSION",
]
