"""Persistent experiment history for kernel research runs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any
import uuid

from .contract import EVALUATION_STATUSES
from .platform.canonical import canonical_json_bytes, require_sha256_digest
from .platform.profiles import LEGACY_RESEARCH_NAMESPACE


SCHEMA_VERSION = 3

# Deliberately identity-based and private.  The coordinated migration module is
# the only production caller allowed to present this object; a stray bool or
# similarly named configuration value cannot accidentally authorize a rewrite.
_V2_TO_V3_MIGRATION_CAPABILITY = object()
_V2_TO_V3_MIGRATION_INTENT = ".v2-to-v3-migration-intent.json"

_LEGACY_NAMESPACE_SNAPSHOT = LEGACY_RESEARCH_NAMESPACE.to_dict()
LEGACY_NAMESPACE_ID = LEGACY_RESEARCH_NAMESPACE.namespace_id

_SOURCE_ARTIFACT_PREFIX = "source-sha256-v1:"
_BUNDLE_ARTIFACT_PREFIX = "bundle-sha256-v1:"
_EXPERIMENT_UID_NAMESPACE = uuid.UUID("3de4f74a-11e3-46d6-a750-41c61f6d7f55")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_object(value: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(value or {})
    # Fail before opening a database transaction if the value is not JSON-safe.
    json.dumps(result, allow_nan=False)
    return result


def _json_snapshot(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _json_object(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return _json_object(converted)
    raise TypeError(f"{name} must be a mapping or expose to_dict()")


def _sample_tuple(values: Sequence[float] | None) -> tuple[float, ...]:
    if values is None:
        return ()
    return tuple(float(value) for value in values)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_artifact_id(candidate_hash: str) -> str:
    return f"{_SOURCE_ARTIFACT_PREFIX}{candidate_hash}"


def _validate_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _artifact_digest(artifact_id: str) -> str:
    _validate_identifier(artifact_id, "artifact_id")
    for prefix in (_SOURCE_ARTIFACT_PREFIX, _BUNDLE_ARTIFACT_PREFIX):
        if artifact_id.startswith(prefix):
            digest = artifact_id[len(prefix) :]
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(
                    "artifact_id digest must be 64 lowercase hexadecimal characters"
                )
            return digest
    raise ValueError(
        "artifact_id must use source-sha256-v1 or bundle-sha256-v1"
    )


@dataclass(frozen=True)
class ResearchNamespaceRecord:
    namespace_id: str
    created_at: str
    identity: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace_id": self.namespace_id,
            "created_at": self.created_at,
            "identity": dict(self.identity),
        }


@dataclass(frozen=True)
class CandidateArtifactRecord:
    artifact_id: str
    artifact_kind: str
    content_sha256: str
    object_path: str
    byte_size: int | None
    created_at: str
    manifest: Mapping[str, Any] = field(default_factory=dict)

    @property
    def candidate_hash(self) -> str:
        return _artifact_digest(self.artifact_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_kind": self.artifact_kind,
            "content_sha256": self.content_sha256,
            "object_path": self.object_path,
            "byte_size": self.byte_size,
            "created_at": self.created_at,
            "manifest": dict(self.manifest),
        }


@dataclass(frozen=True)
class ExperimentRelationRecord:
    id: int
    source_experiment_uid: str
    target_experiment_uid: str
    relation_type: str
    created_at: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_experiment_uid": self.source_experiment_uid,
            "target_experiment_uid": self.target_experiment_uid,
            "relation_type": self.relation_type,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class CaseMeasurement:
    """Correctness and raw timing observations for one benchmark case."""

    name: str
    matched_ratio: float | None = None
    passed: bool | None = None
    raw_samples: tuple[float, ...] = ()
    baseline_samples: tuple[float, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(
        cls, value: "CaseMeasurement | Mapping[str, Any]"
    ) -> "CaseMeasurement":
        if isinstance(value, cls):
            return cls(
                name=value.name,
                matched_ratio=value.matched_ratio,
                passed=value.passed,
                raw_samples=_sample_tuple(value.raw_samples),
                baseline_samples=_sample_tuple(value.baseline_samples),
                metrics=_json_object(value.metrics),
            )
        if not isinstance(value, Mapping):
            raise TypeError("case measurements must be CaseMeasurement or mappings")
        name = value.get("name", value.get("case_name"))
        if not isinstance(name, str) or not name:
            raise ValueError("case measurement name must be a non-empty string")
        raw_samples = value.get(
            "raw_samples", value.get("samples", value.get("timings_ms"))
        )
        baseline_samples = value.get(
            "baseline_samples", value.get("baseline_timings_ms")
        )
        matched_ratio = value.get("matched_ratio")
        passed = value.get("passed")
        return cls(
            name=name,
            matched_ratio=(None if matched_ratio is None else float(matched_ratio)),
            passed=(None if passed is None else bool(passed)),
            raw_samples=_sample_tuple(raw_samples),
            baseline_samples=_sample_tuple(baseline_samples),
            metrics=_json_object(value.get("metrics")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "matched_ratio": self.matched_ratio,
            "passed": self.passed,
            "raw_samples": list(self.raw_samples),
            "baseline_samples": list(self.baseline_samples),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class ExperimentRecord:
    """A complete experiment row with its case measurements."""

    id: int
    schema_version: int
    created_at: str
    candidate_hash: str
    git_commit: str | None
    backend: str
    suite: str
    status: str
    promotable: bool
    duplicate_of_id: int | None
    aggregate_score: float | None
    note: str
    environment: Mapping[str, Any]
    error_summary: str | None
    artifact_path: str
    result: Mapping[str, Any]
    namespace_id: str = LEGACY_NAMESPACE_ID
    artifact_id: str = ""
    experiment_uid: str = ""
    condition_digest: str = ""
    replicate_kind: str = "primary"
    replicate_index: int = 0
    baseline_experiment_uid: str | None = None
    identity: Mapping[str, Any] = field(default_factory=dict)
    case_measurements: tuple[CaseMeasurement, ...] = ()

    @property
    def duplicate(self) -> bool:
        return self.duplicate_of_id is not None

    @property
    def candidate_sha256(self) -> str:
        return self.candidate_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "candidate_hash": self.candidate_hash,
            "git_commit": self.git_commit,
            "backend": self.backend,
            "suite": self.suite,
            "status": self.status,
            "promotable": self.promotable,
            "duplicate": self.duplicate,
            "duplicate_of_id": self.duplicate_of_id,
            "aggregate_score": self.aggregate_score,
            "note": self.note,
            "environment": dict(self.environment),
            "error_summary": self.error_summary,
            "artifact_path": self.artifact_path,
            "artifact_id": self.artifact_id,
            "namespace_id": self.namespace_id,
            "experiment_uid": self.experiment_uid,
            "condition_digest": self.condition_digest,
            "replicate_kind": self.replicate_kind,
            "replicate_index": self.replicate_index,
            "baseline_experiment_uid": self.baseline_experiment_uid,
            "identity": dict(self.identity),
            "result": dict(self.result),
            "case_measurements": [case.to_dict() for case in self.case_measurements],
        }


class HistoryStore:
    """SQLite-backed history with content-addressed candidate artifacts."""

    def __init__(
        self,
        db_path: str | os.PathLike[str],
        state_dir: str | os.PathLike[str] | None = None,
        *,
        _v2_to_v3_migration_capability: object | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.state_dir = Path(state_dir) if state_dir is not None else self.db_path.parent
        self._authorize_persisted_legacy_schema(
            _v2_to_v3_migration_capability
        )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.state_dir / "artifacts"
        self.objects_dir = self.state_dir / "objects" / "sha256"
        self._connection = sqlite3.connect(self.db_path, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._initialize_schema()

    def _authorize_persisted_legacy_schema(self, capability: object | None) -> None:
        """Fail closed before SQLite pragmas can write a persisted V1/V2 DB."""

        intent = self.state_dir / _V2_TO_V3_MIGRATION_INTENT
        if intent.exists() or intent.is_symlink():
            if capability is not _V2_TO_V3_MIGRATION_CAPABILITY:
                raise ValueError(
                    "coordinated V2-to-V3 migration intent is active; "
                    "History Store access is fenced"
                )
        if not self.db_path.exists():
            return
        if self.db_path.is_symlink() or not self.db_path.is_file():
            raise ValueError("history database must be a regular non-symlink file")
        connection = sqlite3.connect(
            self.db_path.resolve().as_uri() + "?mode=ro", uri=True
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
            raise ValueError(
                "persisted History schema version 0 contains user objects; "
                "refusing implicit initialization or migration: " + rendered
            )
        if version in {1, 2} and capability is not _V2_TO_V3_MIGRATION_CAPABILITY:
            raise ValueError(
                "persisted History schema V1/V2 requires the coordinated "
                "V2-to-V3 migration capability; direct Store migration is disabled"
            )

    def _initialize_schema(self) -> None:
        current_version = int(
            self._connection.execute("PRAGMA user_version").fetchone()[0]
        )
        if current_version > SCHEMA_VERSION:
            self._connection.close()
            raise sqlite3.DatabaseError(
                f"history schema {current_version} is newer than supported "
                f"version {SCHEMA_VERSION}"
            )
        if current_version == 0:
            try:
                self._connection.executescript(
                    """
                    BEGIN IMMEDIATE;

                    CREATE TABLE IF NOT EXISTS experiments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        schema_version INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        candidate_hash TEXT NOT NULL,
                        git_commit TEXT,
                        backend TEXT NOT NULL,
                        suite TEXT NOT NULL,
                        status TEXT NOT NULL,
                        promotable INTEGER NOT NULL CHECK (promotable IN (0, 1)),
                        duplicate_of_id INTEGER REFERENCES experiments(id),
                        aggregate_score REAL,
                        note TEXT NOT NULL,
                        environment_json TEXT NOT NULL,
                        error_summary TEXT,
                        artifact_path TEXT NOT NULL,
                        result_json TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS experiments_candidate_hash_idx
                        ON experiments(candidate_hash, id);
                    CREATE INDEX IF NOT EXISTS experiments_best_idx
                        ON experiments(
                            backend, suite, status, promotable, aggregate_score
                        );

                    CREATE TABLE IF NOT EXISTS case_measurements (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        experiment_id INTEGER NOT NULL
                            REFERENCES experiments(id) ON DELETE CASCADE,
                        case_name TEXT NOT NULL,
                        matched_ratio REAL,
                        passed INTEGER
                            CHECK (passed IS NULL OR passed IN (0, 1)),
                        raw_samples_json TEXT NOT NULL,
                        baseline_samples_json TEXT NOT NULL,
                        metrics_json TEXT NOT NULL,
                        UNIQUE(experiment_id, case_name)
                    );

                    PRAGMA user_version = 1;
                    COMMIT;
                    """
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                self._connection.close()
                raise
            current_version = 1
        if current_version == 1:
            try:
                self._connection.executescript(
                    """
                    BEGIN IMMEDIATE;
                    CREATE INDEX IF NOT EXISTS experiments_autorun_note_idx
                        ON experiments(note, candidate_hash, backend, suite, id);
                    PRAGMA user_version = 2;
                    COMMIT;
                    """
                )
            except Exception:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
                self._connection.close()
                raise
            current_version = 2
        if current_version == 2:
            self._migrate_v2_to_v3()

    def _migrate_v2_to_v3(self) -> None:
        """Add namespaced scientific identities without rewriting legacy files."""

        connection = self._connection
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                CREATE TABLE store_metadata (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE research_namespaces (
                    namespace_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    identity_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO research_namespaces (
                    namespace_id, created_at, identity_json
                ) VALUES (?, ?, ?)
                """,
                (
                    LEGACY_NAMESPACE_ID,
                    _utc_now(),
                    _json_text(_LEGACY_NAMESPACE_SNAPSHOT),
                ),
            )
            connection.execute(
                """
                CREATE TABLE candidate_artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    artifact_kind TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    object_path TEXT NOT NULL,
                    byte_size INTEGER,
                    created_at TEXT NOT NULL,
                    manifest_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "ALTER TABLE experiments ADD COLUMN namespace_id TEXT"
            )
            connection.execute(
                "ALTER TABLE experiments ADD COLUMN artifact_id TEXT"
            )
            connection.execute(
                "ALTER TABLE experiments ADD COLUMN experiment_uid TEXT"
            )
            connection.execute(
                "ALTER TABLE experiments ADD COLUMN condition_digest TEXT"
            )
            connection.execute(
                "ALTER TABLE experiments ADD COLUMN replicate_kind TEXT"
            )
            connection.execute(
                "ALTER TABLE experiments ADD COLUMN replicate_index INTEGER"
            )
            connection.execute(
                "ALTER TABLE experiments ADD COLUMN baseline_experiment_uid TEXT"
            )
            connection.execute(
                "ALTER TABLE experiments ADD COLUMN identity_json TEXT"
            )
            connection.execute(
                "ALTER TABLE experiments ADD COLUMN request_digest TEXT"
            )

            artifact_rows = connection.execute(
                """
                SELECT candidate_hash, artifact_path, MIN(created_at) AS created_at
                FROM experiments
                GROUP BY candidate_hash, artifact_path
                ORDER BY MIN(id)
                """
            ).fetchall()
            artifacts_seen: set[str] = set()
            for row in artifact_rows:
                candidate_hash = str(row["candidate_hash"])
                artifact_id = _source_artifact_id(candidate_hash)
                _artifact_digest(artifact_id)
                relative_path = str(row["artifact_path"])
                artifact_path = self._state_path(relative_path)
                if not artifact_path.is_file():
                    raise RuntimeError(
                        f"legacy artifact is missing for {candidate_hash}"
                    )
                data = artifact_path.read_bytes()
                if _sha256_bytes(data) != candidate_hash:
                    raise RuntimeError(
                        f"legacy artifact hash mismatch for {candidate_hash}"
                    )
                if artifact_id in artifacts_seen:
                    continue
                artifacts_seen.add(artifact_id)
                byte_size = len(data)
                connection.execute(
                    """
                    INSERT INTO candidate_artifacts (
                        artifact_id, artifact_kind, content_sha256, object_path,
                        byte_size, created_at, manifest_json
                    ) VALUES (?, 'legacy_source_v1', ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact_id,
                        candidate_hash,
                        relative_path,
                        byte_size,
                        str(row["created_at"]),
                        _json_text(
                            {
                                "format": "legacy_python_source_v1",
                                "entrypoint": Path(relative_path).name,
                            }
                        ),
                    ),
                )

            rows = connection.execute(
                """
                SELECT id, created_at, candidate_hash, backend, suite,
                       artifact_path
                FROM experiments
                ORDER BY id
                """
            ).fetchall()
            for row in rows:
                experiment_uid = "exp-" + str(
                    uuid.uuid5(
                        _EXPERIMENT_UID_NAMESPACE,
                        ":".join(
                            (
                                str(row["id"]),
                                str(row["created_at"]),
                                str(row["candidate_hash"]),
                                str(row["backend"]),
                                str(row["suite"]),
                            )
                        ),
                    )
                )
                condition = {
                    "schema_version": 1,
                    "namespace_id": LEGACY_NAMESPACE_ID,
                    "backend": str(row["backend"]),
                    "suite": str(row["suite"]),
                    "legacy": True,
                }
                condition_digest = "sha256:" + _sha256_bytes(
                    _json_text(condition).encode("utf-8")
                )
                identity = {
                    **condition,
                    "experiment_uid": experiment_uid,
                    "candidate_artifact_id": _source_artifact_id(
                        str(row["candidate_hash"])
                    ),
                    "mode": "legacy_unknown",
                    "parent_artifact_id": "legacy_unknown",
                    "baseline": "legacy_unknown",
                    "replicate_kind": "legacy",
                    "replicate_index": 0,
                    "proposer": "legacy_unknown",
                }
                connection.execute(
                    """
                    UPDATE experiments
                    SET schema_version = ?, namespace_id = ?, artifact_id = ?,
                        experiment_uid = ?, condition_digest = ?,
                        replicate_kind = 'legacy', replicate_index = 0,
                        identity_json = ?, request_digest = ?
                    WHERE id = ?
                    """,
                    (
                        SCHEMA_VERSION,
                        LEGACY_NAMESPACE_ID,
                        _source_artifact_id(str(row["candidate_hash"])),
                        experiment_uid,
                        condition_digest,
                        _json_text(identity),
                        f"legacy:{experiment_uid}",
                        int(row["id"]),
                    ),
                )

            connection.execute(
                """
                CREATE UNIQUE INDEX experiments_uid_idx
                ON experiments(experiment_uid)
                """
            )
            connection.execute(
                """
                CREATE INDEX experiments_namespace_artifact_idx
                ON experiments(namespace_id, artifact_id, id)
                """
            )
            connection.execute(
                """
                CREATE INDEX experiments_namespace_best_idx
                ON experiments(
                    namespace_id, backend, suite, status, promotable,
                    aggregate_score
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE experiment_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_experiment_uid TEXT NOT NULL
                        REFERENCES experiments(experiment_uid) ON DELETE CASCADE,
                    target_experiment_uid TEXT NOT NULL
                        REFERENCES experiments(experiment_uid) ON DELETE CASCADE,
                    relation_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    UNIQUE(
                        source_experiment_uid,
                        target_experiment_uid,
                        relation_type
                    )
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX experiment_relations_target_idx
                ON experiment_relations(target_experiment_uid, relation_type, id)
                """
            )
            connection.execute(
                """
                CREATE TRIGGER experiments_v3_identity_insert
                BEFORE INSERT ON experiments
                WHEN NEW.namespace_id IS NULL
                  OR NEW.artifact_id IS NULL
                  OR NEW.experiment_uid IS NULL
                  OR NEW.condition_digest IS NULL
                  OR NEW.replicate_kind IS NULL
                  OR NEW.replicate_index IS NULL
                  OR NEW.replicate_index < 0
                  OR NEW.identity_json IS NULL
                  OR NEW.request_digest IS NULL
                  OR NOT EXISTS (
                      SELECT 1 FROM research_namespaces
                      WHERE namespace_id = NEW.namespace_id
                  )
                  OR NOT EXISTS (
                      SELECT 1 FROM candidate_artifacts
                      WHERE artifact_id = NEW.artifact_id
                  )
                  OR (
                      NEW.baseline_experiment_uid IS NOT NULL
                      AND NOT EXISTS (
                          SELECT 1 FROM experiments
                          WHERE experiment_uid = NEW.baseline_experiment_uid
                            AND namespace_id = NEW.namespace_id
                      )
                  )
                BEGIN
                    SELECT RAISE(ABORT, 'invalid V3 experiment identity');
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER experiments_v3_identity_update
                BEFORE UPDATE OF
                    namespace_id, artifact_id, experiment_uid,
                    condition_digest, replicate_kind, replicate_index,
                    baseline_experiment_uid, identity_json, request_digest
                ON experiments
                WHEN NEW.namespace_id IS NULL
                  OR NEW.artifact_id IS NULL
                  OR NEW.experiment_uid IS NULL
                  OR NEW.condition_digest IS NULL
                  OR NEW.replicate_kind IS NULL
                  OR NEW.replicate_index IS NULL
                  OR NEW.replicate_index < 0
                  OR NEW.identity_json IS NULL
                  OR NEW.request_digest IS NULL
                  OR NOT EXISTS (
                      SELECT 1 FROM research_namespaces
                      WHERE namespace_id = NEW.namespace_id
                  )
                  OR NOT EXISTS (
                      SELECT 1 FROM candidate_artifacts
                      WHERE artifact_id = NEW.artifact_id
                  )
                  OR (
                      NEW.baseline_experiment_uid IS NOT NULL
                      AND NOT EXISTS (
                          SELECT 1 FROM experiments
                          WHERE experiment_uid = NEW.baseline_experiment_uid
                            AND namespace_id = NEW.namespace_id
                      )
                  )
                BEGIN
                    SELECT RAISE(ABORT, 'invalid V3 experiment identity');
                END
                """
            )
            connection.execute(
                """
                INSERT INTO store_metadata(key, value_json, updated_at)
                VALUES ('schema', ?, ?)
                """,
                (
                    _json_text(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "legacy_namespace_id": LEGACY_NAMESPACE_ID,
                        }
                    ),
                    _utc_now(),
                ),
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.execute("COMMIT")
        except Exception:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            connection.close()
            raise

    @property
    def schema_version(self) -> int:
        return int(self._connection.execute("PRAGMA user_version").fetchone()[0])

    @property
    def journal_mode(self) -> str:
        return str(self._connection.execute("PRAGMA journal_mode").fetchone()[0])

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "HistoryStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _state_path(self, relative_path: str) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"unsafe history object path: {relative_path}")
        root = self.state_dir.resolve(strict=False)
        resolved = (self.state_dir / relative).resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise RuntimeError(f"history object escapes state directory: {relative_path}")
        return resolved

    def get_store_metadata(self, key: str) -> Any | None:
        _validate_identifier(key, "metadata key")
        row = self._connection.execute(
            "SELECT value_json FROM store_metadata WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else json.loads(row["value_json"])

    def set_store_metadata(self, key: str, value: Any) -> None:
        _validate_identifier(key, "metadata key")
        value_json = _json_text(value)
        self._connection.execute(
            """
            INSERT INTO store_metadata(key, value_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at
            """,
            (key, value_json, _utc_now()),
        )

    def ensure_namespace(
        self,
        namespace_id: str,
        identity: Any,
    ) -> ResearchNamespaceRecord:
        """Insert an immutable namespace snapshot or verify the existing one."""

        namespace_id = _validate_identifier(str(namespace_id), "namespace_id")
        identity_object = _json_snapshot(identity, "namespace identity")
        declared_namespace = identity_object.get("namespace_id")
        if declared_namespace is not None and declared_namespace != namespace_id:
            raise ValueError("identity namespace_id does not match namespace_id")
        identity_json = _json_text(identity_object)
        created_at = _utc_now()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT * FROM research_namespaces WHERE namespace_id = ?",
                (namespace_id,),
            ).fetchone()
            if row is None:
                self._connection.execute(
                    """
                    INSERT INTO research_namespaces(
                        namespace_id, created_at, identity_json
                    ) VALUES (?, ?, ?)
                    """,
                    (namespace_id, created_at, identity_json),
                )
            elif str(row["identity_json"]) != identity_json:
                raise ValueError(
                    f"namespace {namespace_id!r} already has a different identity"
                )
            else:
                created_at = str(row["created_at"])
            self._connection.execute("COMMIT")
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        return ResearchNamespaceRecord(
            namespace_id=namespace_id,
            created_at=created_at,
            identity=identity_object,
        )

    def get_namespace(self, namespace_id: str) -> ResearchNamespaceRecord | None:
        row = self._connection.execute(
            "SELECT * FROM research_namespaces WHERE namespace_id = ?",
            (str(namespace_id),),
        ).fetchone()
        if row is None:
            return None
        return ResearchNamespaceRecord(
            namespace_id=str(row["namespace_id"]),
            created_at=str(row["created_at"]),
            identity=json.loads(row["identity_json"]),
        )

    def list_namespaces(self) -> list[ResearchNamespaceRecord]:
        return [
            ResearchNamespaceRecord(
                namespace_id=str(row["namespace_id"]),
                created_at=str(row["created_at"]),
                identity=json.loads(row["identity_json"]),
            )
            for row in self._connection.execute(
                "SELECT * FROM research_namespaces ORDER BY created_at, namespace_id"
            ).fetchall()
        ]

    def _store_object(self, content: bytes) -> tuple[str, bool]:
        digest = _sha256_bytes(content)
        relative_path = f"objects/sha256/{digest[:2]}/{digest[2:]}"
        path = self._state_path(relative_path)
        directory = path.parent
        if path.exists():
            if not path.is_file() or path.read_bytes() != content:
                raise RuntimeError(f"content-addressed object collision for {digest}")
            return relative_path, False

        directory.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=directory,
                prefix=f".{digest}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_name = temporary.name
            os.replace(temporary_name, path)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink()
                except FileNotFoundError:
                    pass
        return relative_path, True

    def _candidate_artifact_from_row(
        self, row: sqlite3.Row
    ) -> CandidateArtifactRecord:
        return CandidateArtifactRecord(
            artifact_id=str(row["artifact_id"]),
            artifact_kind=str(row["artifact_kind"]),
            content_sha256=str(row["content_sha256"]),
            object_path=str(row["object_path"]),
            byte_size=(None if row["byte_size"] is None else int(row["byte_size"])),
            created_at=str(row["created_at"]),
            manifest=json.loads(row["manifest_json"]),
        )

    def get_candidate_artifact(
        self, artifact_id: str
    ) -> CandidateArtifactRecord | None:
        row = self._connection.execute(
            "SELECT * FROM candidate_artifacts WHERE artifact_id = ?",
            (str(artifact_id),),
        ).fetchone()
        return None if row is None else self._candidate_artifact_from_row(row)

    def has_artifact(self, artifact_id: str) -> bool:
        return self.get_candidate_artifact(str(artifact_id)) is not None

    def read_candidate_artifact(self, artifact_id: str) -> bytes:
        record = self.get_candidate_artifact(str(artifact_id))
        if record is None:
            raise KeyError(f"unknown candidate artifact {artifact_id!r}")
        path = self._state_path(record.object_path)
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"candidate artifact object is unavailable: {record.object_path}"
            ) from exc
        if _sha256_bytes(content) != record.content_sha256:
            raise RuntimeError(
                f"candidate artifact object hash mismatch for {record.artifact_id}"
            )
        return content

    def _insert_candidate_artifact(
        self,
        *,
        artifact_id: str,
        content: bytes,
        artifact_kind: str,
        manifest: Mapping[str, Any],
    ) -> tuple[CandidateArtifactRecord, bool]:
        artifact_id = str(artifact_id)
        expected_digest = _artifact_digest(artifact_id)
        content_digest = _sha256_bytes(content)
        artifact_kind = _validate_identifier(artifact_kind, "artifact_kind")
        manifest_object = _json_object(manifest)
        identity_digest = (
            _sha256_bytes(canonical_json_bytes(manifest_object))
            if artifact_id.startswith(_BUNDLE_ARTIFACT_PREFIX)
            else content_digest
        )
        if expected_digest != identity_digest:
            raise ValueError(
                "artifact_id does not match candidate artifact identity"
            )
        manifest_json = _json_text(manifest_object)
        existing = self._connection.execute(
            "SELECT * FROM candidate_artifacts WHERE artifact_id = ?",
            (artifact_id,),
        ).fetchone()
        if existing is not None:
            record = self._candidate_artifact_from_row(existing)
            if (
                record.content_sha256 != content_digest
                or record.artifact_kind != artifact_kind
                or _json_text(record.manifest) != manifest_json
            ):
                raise RuntimeError(
                    f"candidate artifact metadata collision for {artifact_id}"
                )
            if self.read_candidate_artifact(artifact_id) != content:
                raise RuntimeError(
                    f"content-addressed artifact collision for {artifact_id}"
                )
            return record, False

        object_path, object_created = self._store_object(content)
        created_at = _utc_now()
        self._connection.execute(
            """
            INSERT INTO candidate_artifacts(
                artifact_id, artifact_kind, content_sha256, object_path,
                byte_size, created_at, manifest_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                artifact_id,
                artifact_kind,
                content_digest,
                object_path,
                len(content),
                created_at,
                manifest_json,
            ),
        )
        return (
            CandidateArtifactRecord(
                artifact_id=artifact_id,
                artifact_kind=artifact_kind,
                content_sha256=content_digest,
                object_path=object_path,
                byte_size=len(content),
                created_at=created_at,
                manifest=manifest_object,
            ),
            object_created,
        )

    def store_candidate_artifact(
        self,
        content: bytes | str,
        *,
        artifact_id: str | None = None,
        artifact_kind: str = "source_text_v1",
        manifest: Mapping[str, Any] | None = None,
    ) -> CandidateArtifactRecord:
        """Persist an immutable scientific object in the V3 SHA-256 CAS."""

        raw = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        artifact_id = artifact_id or _source_artifact_id(_sha256_bytes(raw))
        manifest_object = _json_object(manifest)
        object_created = False
        object_path: str | None = None
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            record, object_created = self._insert_candidate_artifact(
                artifact_id=str(artifact_id),
                content=raw,
                artifact_kind=artifact_kind,
                manifest=manifest_object,
            )
            object_path = record.object_path
            self._connection.execute("COMMIT")
            return record
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            if object_created and object_path is not None:
                try:
                    self._state_path(object_path).unlink()
                except FileNotFoundError:
                    pass
            raise

    def store_candidate_bundle(self, bundle: Any) -> CandidateArtifactRecord:
        """Store a validated ``CandidateBundle`` without coupling its identity
        digest to the serialized payload bytes.

        Bundle IDs hash the canonical content-free manifest; the CAS object
        contains the canonical bundle including its exact source text.
        """

        try:
            artifact_id = str(bundle.artifact_id)
            manifest = bundle.manifest
            payload = bundle.to_dict(include_content=True)
        except (AttributeError, TypeError) as exc:
            raise TypeError("bundle must be a validated CandidateBundle") from exc
        return self.store_candidate_artifact(
            canonical_json_bytes(payload),
            artifact_id=artifact_id,
            artifact_kind="source_bundle_v1",
            manifest=manifest,
        )

    def _store_artifact(self, candidate_hash: str, source: str) -> tuple[str, bool]:
        relative_path = f"artifacts/{candidate_hash}.py"
        artifact = self._state_path(relative_path)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if artifact.exists():
            if artifact.read_text(encoding="utf-8") != source:
                raise RuntimeError(
                    f"content-addressed artifact collision for {candidate_hash}"
                )
            return relative_path, False

        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.artifacts_dir,
                prefix=f".{candidate_hash}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(source)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_name = temporary.name
            os.replace(temporary_name, artifact)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    Path(temporary_name).unlink()
                except FileNotFoundError:
                    pass
        return relative_path, True

    def record_experiment(
        self,
        *,
        candidate_source: str,
        status: str,
        backend: str,
        suite: str,
        git_commit: str | None = None,
        promotable: bool = False,
        aggregate_score: float | None = None,
        note: str = "",
        environment: Mapping[str, Any] | None = None,
        error_summary: str | None = None,
        case_measurements: Iterable[CaseMeasurement | Mapping[str, Any]] = (),
        candidate_hash: str | None = None,
        result: Mapping[str, Any] | None = None,
        created_at: str | None = None,
        experiment_uid: str | None = None,
        namespace_id: str | None = None,
        condition_digest: str | None = None,
        replicate_kind: str | None = None,
        replicate_index: int | None = None,
        candidate_artifact_id: str | None = None,
        baseline_experiment_uid: str | None = None,
        identity: Any | None = None,
    ) -> ExperimentRecord:
        """Atomically append, or idempotently replay, one experiment.

        Calls that omit every V3 identity argument preserve the V1/V2 global
        duplicate marker and legacy artifact path.  Supplying a namespace,
        UID, condition, replicate, artifact ID, baseline UID, or identity opts
        into V3 semantics: experiments may intentionally reuse an artifact and
        are idempotent by ``experiment_uid`` instead of candidate bytes.
        """

        if not isinstance(candidate_source, str):
            raise TypeError("candidate_source must be a string")
        source_bytes = candidate_source.encode("utf-8")
        digest = _sha256_bytes(source_bytes)
        if candidate_hash is not None and str(candidate_hash) != digest:
            raise ValueError("candidate_hash does not match candidate_source")
        candidate_hash = digest
        if not isinstance(status, str) or not status:
            raise ValueError("status must be a non-empty string")
        if not isinstance(backend, str) or not backend:
            raise ValueError("backend must be a non-empty string")
        if not isinstance(suite, str) or not suite:
            raise ValueError("suite must be a non-empty string")

        identity_object = (
            {} if identity is None else _json_snapshot(identity, "experiment identity")
        )
        legacy_call = all(
            value is None
            for value in (
                experiment_uid,
                namespace_id,
                condition_digest,
                replicate_kind,
                replicate_index,
                candidate_artifact_id,
                baseline_experiment_uid,
                identity,
            )
        )
        nested_namespace = identity_object.get("namespace")
        identity_namespace_id = identity_object.get("namespace_id")
        if isinstance(nested_namespace, Mapping):
            identity_namespace_id = nested_namespace.get("namespace_id")
        if namespace_id is None and identity_namespace_id is not None:
            namespace_id = str(identity_namespace_id)
        if experiment_uid is None and identity_object.get("experiment_uid") is not None:
            experiment_uid = str(identity_object["experiment_uid"])
        if condition_digest is None and identity_object.get("condition_digest") is not None:
            condition_digest = str(identity_object["condition_digest"])
        if replicate_kind is None and identity_object.get("replicate_kind") is not None:
            replicate_kind = str(identity_object["replicate_kind"])
        if replicate_index is None and identity_object.get("replicate_index") is not None:
            replicate_index = identity_object["replicate_index"]
        if (
            candidate_artifact_id is None
            and identity_object.get("candidate_artifact_id") is not None
        ):
            candidate_artifact_id = str(identity_object["candidate_artifact_id"])
        namespace_id = _validate_identifier(
            LEGACY_NAMESPACE_ID if namespace_id is None else str(namespace_id),
            "namespace_id",
        )
        experiment_uid = _validate_identifier(
            experiment_uid
            or (f"exp-{uuid.uuid4()}" if legacy_call else str(uuid.uuid4())),
            "experiment_uid",
        )
        replicate_kind = _validate_identifier(
            replicate_kind or ("legacy" if legacy_call else "primary"),
            "replicate_kind",
        )
        replicate_index = 0 if replicate_index is None else replicate_index
        if type(replicate_index) is not int or replicate_index < 0:
            raise ValueError("replicate_index must be a non-negative integer")
        candidate_artifact_id = str(
            candidate_artifact_id or _source_artifact_id(candidate_hash)
        )
        _artifact_digest(candidate_artifact_id)
        if baseline_experiment_uid is not None:
            baseline_experiment_uid = _validate_identifier(
                str(baseline_experiment_uid), "baseline_experiment_uid"
            )

        cases = tuple(CaseMeasurement.from_value(case) for case in case_measurements)
        if len({case.name for case in cases}) != len(cases):
            raise ValueError("case measurement names must be unique within an experiment")
        environment_object = _json_object(environment)
        result_object = _json_object(result)
        condition_material = {
            "schema_version": 1,
            "namespace_id": namespace_id,
            "candidate_artifact_id": candidate_artifact_id,
            "backend": backend,
            "suite": suite,
            "replicate_kind": replicate_kind,
            "replicate_index": replicate_index,
            "baseline_experiment_uid": baseline_experiment_uid,
            "identity": identity_object,
        }
        condition_digest = require_sha256_digest(
            condition_digest
            or str(identity_object.get("condition_digest") or "")
            or "sha256:"
            + _sha256_bytes(canonical_json_bytes(condition_material)),
            field="condition_digest",
        )
        required_identity = {
            "namespace_id": namespace_id,
            "experiment_uid": experiment_uid,
            "candidate_artifact_id": candidate_artifact_id,
            "condition_digest": condition_digest,
            "backend": backend,
            "suite": suite,
            "replicate_kind": replicate_kind,
            "replicate_index": replicate_index,
            "baseline_experiment_uid": baseline_experiment_uid,
        }
        if identity is not None:
            identity_checks = {
                "experiment_uid": experiment_uid,
                "candidate_artifact_id": candidate_artifact_id,
                "condition_digest": condition_digest,
                "suite": suite,
                "replicate_kind": replicate_kind,
                "replicate_index": replicate_index,
            }
            for key, value in identity_checks.items():
                if identity_object.get(key) != value:
                    raise ValueError(
                        f"identity {key} does not match record arguments"
                    )
            if identity_namespace_id != namespace_id:
                raise ValueError(
                    "identity namespace_id does not match record arguments"
                )
        else:
            identity_object.update(required_identity)
        requested_created_at = created_at
        created_at = created_at or _utc_now()
        request_digest = "sha256:" + _sha256_bytes(
            _json_text(
                {
                    "candidate_hash": candidate_hash,
                    "candidate_artifact_id": candidate_artifact_id,
                    "status": status,
                    "backend": backend,
                    "suite": suite,
                    "git_commit": git_commit,
                    "promotable": bool(promotable),
                    "aggregate_score": aggregate_score,
                    "note": note,
                    "environment": environment_object,
                    "error_summary": error_summary,
                    "cases": [case.to_dict() for case in cases],
                    "result": result_object,
                    "requested_created_at": requested_created_at,
                    "namespace_id": namespace_id,
                    "condition_digest": condition_digest,
                    "replicate_kind": replicate_kind,
                    "replicate_index": replicate_index,
                    "baseline_experiment_uid": baseline_experiment_uid,
                    "identity": identity_object,
                }
            ).encode("utf-8")
        )
        legacy_artifact_created = False
        cas_object_created = False
        artifact_path: str | None = None
        cas_object_path: str | None = None
        experiment_id: int | None = None

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            namespace_row = self._connection.execute(
                "SELECT 1 FROM research_namespaces WHERE namespace_id = ?",
                (namespace_id,),
            ).fetchone()
            if namespace_row is None:
                raise ValueError(f"unknown research namespace {namespace_id!r}")

            replay_row = self._connection.execute(
                "SELECT * FROM experiments WHERE experiment_uid = ?",
                (experiment_uid,),
            ).fetchone()
            if replay_row is not None:
                if str(replay_row["request_digest"]) != request_digest:
                    raise ValueError(
                        f"experiment_uid {experiment_uid!r} was reused with "
                        "different experiment data"
                    )
                experiment_id = int(replay_row["id"])
                self._connection.execute("COMMIT")
                replay = self.get_experiment(experiment_id)
                if replay is None:  # pragma: no cover
                    raise RuntimeError("idempotent experiment could not be reloaded")
                return replay

            baseline_row = None
            if baseline_experiment_uid is not None:
                baseline_row = self._connection.execute(
                    """
                    SELECT namespace_id FROM experiments
                    WHERE experiment_uid = ?
                    """,
                    (baseline_experiment_uid,),
                ).fetchone()
                if baseline_row is None:
                    raise ValueError(
                        f"unknown baseline experiment {baseline_experiment_uid!r}"
                    )
                if str(baseline_row["namespace_id"]) != namespace_id:
                    raise ValueError("baseline experiment belongs to another namespace")

            duplicate_row = None
            if legacy_call:
                duplicate_row = self._connection.execute(
                    """
                    SELECT id FROM experiments
                    WHERE candidate_hash = ? ORDER BY id LIMIT 1
                    """,
                    (candidate_hash,),
                ).fetchone()

            artifact_record = self.get_candidate_artifact(candidate_artifact_id)
            if artifact_record is None:
                if candidate_artifact_id.startswith(_BUNDLE_ARTIFACT_PREFIX):
                    raise ValueError(
                        "bundle candidate artifact must be stored before its experiment"
                    )
                artifact_record, cas_object_created = self._insert_candidate_artifact(
                    artifact_id=candidate_artifact_id,
                    content=source_bytes,
                    artifact_kind="source_text_v1",
                    manifest={
                        "format": "python_source_v1",
                        "entrypoint": "kernel.py",
                        "media_type": "text/x-python",
                    },
                )
                cas_object_path = artifact_record.object_path
            elif candidate_artifact_id.startswith(_SOURCE_ARTIFACT_PREFIX):
                if self.read_candidate_artifact(candidate_artifact_id) != source_bytes:
                    raise RuntimeError(
                        f"candidate source differs from {candidate_artifact_id}"
                    )
            else:
                manifest = artifact_record.manifest
                entrypoint = manifest.get("entrypoint")
                files = manifest.get("files")
                entrypoint_rows = (
                    [
                        item
                        for item in files
                        if isinstance(item, Mapping)
                        and item.get("path") == entrypoint
                    ]
                    if isinstance(files, list)
                    else []
                )
                if (
                    not isinstance(entrypoint, str)
                    or len(entrypoint_rows) != 1
                    or entrypoint_rows[0].get("sha256")
                    != f"sha256:{candidate_hash}"
                ):
                    raise ValueError(
                        "candidate source does not match bundle entrypoint manifest"
                    )

            if legacy_call:
                artifact_path, legacy_artifact_created = self._store_artifact(
                    candidate_hash, candidate_source
                )
            else:
                artifact_path = artifact_record.object_path
            cursor = self._connection.execute(
                """
                INSERT INTO experiments (
                    schema_version, created_at, candidate_hash, git_commit,
                    backend, suite, status, promotable, duplicate_of_id,
                    aggregate_score, note, environment_json, error_summary,
                    artifact_path, result_json, namespace_id, artifact_id,
                    experiment_uid, condition_digest, replicate_kind,
                    replicate_index, baseline_experiment_uid, identity_json,
                    request_digest
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    SCHEMA_VERSION,
                    created_at,
                    candidate_hash,
                    git_commit,
                    backend,
                    suite,
                    status,
                    int(bool(promotable)),
                    None if duplicate_row is None else int(duplicate_row["id"]),
                    aggregate_score,
                    note,
                    _json_text(environment_object),
                    error_summary,
                    artifact_path,
                    _json_text(result_object),
                    namespace_id,
                    candidate_artifact_id,
                    experiment_uid,
                    condition_digest,
                    replicate_kind,
                    replicate_index,
                    baseline_experiment_uid,
                    _json_text(identity_object),
                    request_digest,
                ),
            )
            experiment_id = int(cursor.lastrowid)
            for case in cases:
                self._connection.execute(
                    """
                    INSERT INTO case_measurements (
                        experiment_id, case_name, matched_ratio, passed,
                        raw_samples_json, baseline_samples_json, metrics_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        experiment_id,
                        case.name,
                        case.matched_ratio,
                        None if case.passed is None else int(case.passed),
                        _json_text(list(case.raw_samples)),
                        _json_text(list(case.baseline_samples)),
                        _json_text(dict(case.metrics)),
                    ),
                )
            if baseline_experiment_uid is not None:
                self._connection.execute(
                    """
                    INSERT INTO experiment_relations(
                        source_experiment_uid, target_experiment_uid,
                        relation_type, created_at, metadata_json
                    ) VALUES (?, ?, 'baseline', ?, '{}')
                    """,
                    (experiment_uid, baseline_experiment_uid, created_at),
                )
            self._connection.execute("COMMIT")
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            if legacy_artifact_created and artifact_path is not None:
                try:
                    self._state_path(artifact_path).unlink()
                except FileNotFoundError:
                    pass
            if cas_object_created and cas_object_path is not None:
                try:
                    self._state_path(cas_object_path).unlink()
                except FileNotFoundError:
                    pass
            raise

        if experiment_id is None:  # pragma: no cover
            raise RuntimeError("experiment insert did not produce an id")
        record = self.get_experiment(experiment_id)
        if record is None:  # pragma: no cover - guards an impossible committed state
            raise RuntimeError("recorded experiment could not be reloaded")
        return record

    def record(self, **values: Any) -> ExperimentRecord:
        """Short alias for ``record_experiment``."""

        return self.record_experiment(**values)

    def record_legacy_experiment(
        self,
        *,
        namespace_id: str,
        **values: Any,
    ) -> ExperimentRecord:
        """Record through the V1 compatibility path, explicitly namespaced.

        V1 callers historically relied on candidate-hash duplicate markers and
        ``artifacts/<hash>.py``.  Passing ``namespace_id`` directly to
        :meth:`record_experiment` opts into V3 experiment semantics and would
        silently change both behaviours.  This narrow adapter makes the legacy
        namespace choice auditable while deliberately preserving the V1 storage
        and output contract.
        """

        if namespace_id != LEGACY_NAMESPACE_ID:
            raise ValueError(
                "the V1 compatibility recorder only accepts the legacy namespace"
            )
        v3_fields = {
            "experiment_uid",
            "condition_digest",
            "replicate_kind",
            "replicate_index",
            "candidate_artifact_id",
            "baseline_experiment_uid",
            "identity",
        }
        supplied_v3_fields = sorted(v3_fields.intersection(values))
        if supplied_v3_fields:
            raise ValueError(
                "legacy recording cannot accept V3 identity fields: "
                + ", ".join(supplied_v3_fields)
            )
        return self.record_experiment(**values)

    def _record_from_row(
        self,
        row: sqlite3.Row,
        case_rows: Sequence[sqlite3.Row] | None = None,
    ) -> ExperimentRecord:
        if case_rows is None:
            case_rows = self._connection.execute(
                """
                SELECT experiment_id, case_name, matched_ratio, passed,
                       raw_samples_json, baseline_samples_json, metrics_json
                FROM case_measurements
                WHERE experiment_id = ?
                ORDER BY id
                """,
                (int(row["id"]),),
            ).fetchall()
        cases = tuple(
            CaseMeasurement(
                name=str(case["case_name"]),
                matched_ratio=(
                    None
                    if case["matched_ratio"] is None
                    else float(case["matched_ratio"])
                ),
                passed=(None if case["passed"] is None else bool(case["passed"])),
                raw_samples=tuple(json.loads(case["raw_samples_json"])),
                baseline_samples=tuple(json.loads(case["baseline_samples_json"])),
                metrics=json.loads(case["metrics_json"]),
            )
            for case in case_rows
        )
        return ExperimentRecord(
            id=int(row["id"]),
            schema_version=int(row["schema_version"]),
            created_at=str(row["created_at"]),
            candidate_hash=str(row["candidate_hash"]),
            git_commit=row["git_commit"],
            backend=str(row["backend"]),
            suite=str(row["suite"]),
            status=str(row["status"]),
            promotable=bool(row["promotable"]),
            duplicate_of_id=(
                None if row["duplicate_of_id"] is None else int(row["duplicate_of_id"])
            ),
            aggregate_score=(
                None
                if row["aggregate_score"] is None
                else float(row["aggregate_score"])
            ),
            note=str(row["note"]),
            environment=json.loads(row["environment_json"]),
            error_summary=row["error_summary"],
            artifact_path=str(row["artifact_path"]),
            result=json.loads(row["result_json"]),
            namespace_id=str(row["namespace_id"]),
            artifact_id=str(row["artifact_id"]),
            experiment_uid=str(row["experiment_uid"]),
            condition_digest=str(row["condition_digest"]),
            replicate_kind=str(row["replicate_kind"]),
            replicate_index=int(row["replicate_index"]),
            baseline_experiment_uid=row["baseline_experiment_uid"],
            identity=json.loads(row["identity_json"]),
            case_measurements=cases,
        )

    def _records_from_rows(
        self, rows: Sequence[sqlite3.Row]
    ) -> list[ExperimentRecord]:
        if not rows:
            return []
        ids = [int(row["id"]) for row in rows]
        placeholders = ",".join("?" for _ in ids)
        case_rows = self._connection.execute(
            f"""
            SELECT experiment_id, case_name, matched_ratio, passed,
                   raw_samples_json, baseline_samples_json, metrics_json, id
            FROM case_measurements
            WHERE experiment_id IN ({placeholders})
            ORDER BY experiment_id, id
            """,
            ids,
        ).fetchall()
        by_experiment: dict[int, list[sqlite3.Row]] = {
            experiment_id: [] for experiment_id in ids
        }
        for case in case_rows:
            by_experiment[int(case["experiment_id"])].append(case)
        return [
            self._record_from_row(row, by_experiment[int(row["id"])])
            for row in rows
        ]

    def get_experiment(self, experiment_id: int) -> ExperimentRecord | None:
        row = self._connection.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        return (
            None
            if row is None
            else self._records_from_rows([row])[0]
        )

    def get_experiment_by_uid(
        self, experiment_uid: str
    ) -> ExperimentRecord | None:
        row = self._connection.execute(
            "SELECT * FROM experiments WHERE experiment_uid = ?",
            (str(experiment_uid),),
        ).fetchone()
        return None if row is None else self._records_from_rows([row])[0]

    def list_experiments(
        self,
        *,
        candidate_hash: str | None = None,
        artifact_id: str | None = None,
        experiment_uid: str | None = None,
        status: str | None = None,
        backend: str | None = None,
        suite: str | None = None,
        condition_digest: str | None = None,
        replicate_kind: str | None = None,
        baseline_experiment_uid: str | None = None,
        namespace_id: str | None = LEGACY_NAMESPACE_ID,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[ExperimentRecord]:
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in (
            ("candidate_hash", candidate_hash),
            ("artifact_id", None if artifact_id is None else str(artifact_id)),
            ("experiment_uid", experiment_uid),
            ("status", status),
            ("backend", backend),
            ("suite", suite),
            ("condition_digest", condition_digest),
            ("replicate_kind", replicate_kind),
            ("baseline_experiment_uid", baseline_experiment_uid),
            ("namespace_id", namespace_id),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        query = "SELECT * FROM experiments"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id " + ("DESC" if newest_first else "ASC")
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            query += " LIMIT ?"
            parameters.append(limit)
        rows = self._connection.execute(query, parameters).fetchall()
        return self._records_from_rows(rows)

    def list_experiments_for_namespace(
        self, namespace_id: str, **filters: Any
    ) -> list[ExperimentRecord]:
        return self.list_experiments(namespace_id=str(namespace_id), **filters)

    def list_all_experiments(self, **filters: Any) -> list[ExperimentRecord]:
        return self.list_experiments(namespace_id=None, **filters)

    def list_noise_experiments(
        self,
        *,
        namespace_id: str,
        artifact_id: str,
        baseline_experiment_uid: str,
        backend: str = "c500",
        suite: str = "full",
        status: str | None = None,
    ) -> list[ExperimentRecord]:
        """Return a narrowly-scoped set of V3 same-artifact noise attempts.

        This is the trusted query boundary used by the V2 noise gate.  It is
        intentionally impossible to omit namespace, artifact or frozen
        baseline identity, and it never falls back to a candidate hash.  The
        caller still validates the full ExperimentIdentity condition family
        and execution-environment digest before treating rows as evidence.
        """

        namespace_id = _validate_identifier(str(namespace_id), "namespace_id")
        artifact_id = str(artifact_id)
        _artifact_digest(artifact_id)
        baseline_experiment_uid = _validate_identifier(
            str(baseline_experiment_uid), "baseline_experiment_uid"
        )
        return self.list_experiments(
            namespace_id=namespace_id,
            artifact_id=artifact_id,
            baseline_experiment_uid=baseline_experiment_uid,
            backend=backend,
            suite=suite,
            status=status,
            replicate_kind="noise",
        )

    def find_by_candidate_hash(
        self,
        candidate_hash: str,
        *,
        namespace_id: str | None = LEGACY_NAMESPACE_ID,
    ) -> list[ExperimentRecord]:
        return self.list_experiments(
            candidate_hash=candidate_hash, namespace_id=namespace_id
        )

    def find_by_note_candidate(
        self,
        *,
        note: str,
        candidate_hash: str,
        namespace_id: str = LEGACY_NAMESPACE_ID,
    ) -> ExperimentRecord | None:
        row = self._connection.execute(
            """
            SELECT * FROM experiments
            WHERE note = ? AND candidate_hash = ? AND namespace_id = ?
            ORDER BY id DESC LIMIT 1
            """,
            (note, candidate_hash, namespace_id),
        ).fetchone()
        return (
            None
            if row is None
            else self._records_from_rows([row])[0]
        )

    def list_autorun_experiments(
        self,
        *,
        run_id: str,
        backend: str | None = None,
        suite: str | None = None,
        promotable_only: bool = False,
        namespace_id: str = LEGACY_NAMESPACE_ID,
    ) -> list[ExperimentRecord]:
        clauses = ["note LIKE ?", "namespace_id = ?"]
        parameters: list[Any] = [f"autorun:{run_id}:%", namespace_id]
        if backend is not None:
            clauses.append("backend = ?")
            parameters.append(backend)
        if suite is not None:
            clauses.append("suite = ?")
            parameters.append(suite)
        if promotable_only:
            clauses.append("promotable = 1")
        rows = self._connection.execute(
            "SELECT * FROM experiments WHERE "
            + " AND ".join(clauses)
            + " ORDER BY id",
            parameters,
        ).fetchall()
        return self._records_from_rows(rows)

    def has_candidate(self, candidate_hash: str) -> bool:
        row = self._connection.execute(
            """
            SELECT 1 FROM experiments
            WHERE namespace_id = ? AND candidate_hash = ? LIMIT 1
            """,
            (LEGACY_NAMESPACE_ID, candidate_hash),
        ).fetchone()
        return row is not None

    def has_candidate_in_namespace(
        self, namespace_id: str, artifact_or_hash: str
    ) -> bool:
        value = str(artifact_or_hash)
        artifact_id = (
            value
            if value.startswith((_SOURCE_ARTIFACT_PREFIX, _BUNDLE_ARTIFACT_PREFIX))
            else _source_artifact_id(value)
        )
        row = self._connection.execute(
            """
            SELECT 1 FROM experiments
            WHERE namespace_id = ?
              AND (artifact_id = ? OR candidate_hash = ?)
            LIMIT 1
            """,
            (str(namespace_id), artifact_id, value),
        ).fetchone()
        return row is not None

    def candidate_seen_count(self, candidate_hash: str) -> int:
        return int(
            self._connection.execute(
                """
                SELECT COUNT(*) FROM experiments
                WHERE namespace_id = ? AND candidate_hash = ?
                """,
                (LEGACY_NAMESPACE_ID, candidate_hash),
            ).fetchone()[0]
        )

    def get_best(
        self,
        *,
        backend: str | None = None,
        suite: str | None = None,
    ) -> ExperimentRecord | None:
        return self.get_best_for_namespace(
            LEGACY_NAMESPACE_ID, backend=backend, suite=suite
        )

    def get_best_for_namespace(
        self,
        namespace_id: str,
        *,
        backend: str | None = None,
        suite: str | None = None,
    ) -> ExperimentRecord | None:
        clauses = [
            "namespace_id = ?",
            "status = 'SUCCESS'",
            "promotable = 1",
            "aggregate_score IS NOT NULL",
        ]
        parameters: list[Any] = [str(namespace_id)]
        if backend is not None:
            clauses.append("backend = ?")
            parameters.append(backend)
        if suite is not None:
            clauses.append("suite = ?")
            parameters.append(suite)
        row = self._connection.execute(
            "SELECT * FROM experiments WHERE "
            + " AND ".join(clauses)
            + " ORDER BY aggregate_score DESC, id DESC LIMIT 1",
            parameters,
        ).fetchone()
        return (
            None
            if row is None
            else self._records_from_rows([row])[0]
        )

    def _relation_from_row(
        self, row: sqlite3.Row
    ) -> ExperimentRelationRecord:
        return ExperimentRelationRecord(
            id=int(row["id"]),
            source_experiment_uid=str(row["source_experiment_uid"]),
            target_experiment_uid=str(row["target_experiment_uid"]),
            relation_type=str(row["relation_type"]),
            created_at=str(row["created_at"]),
            metadata=json.loads(row["metadata_json"]),
        )

    def add_experiment_relation(
        self,
        *,
        source_experiment_uid: str,
        target_experiment_uid: str,
        relation_type: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ExperimentRelationRecord:
        """Add an immutable, idempotent relation between two experiments."""

        source_experiment_uid = _validate_identifier(
            source_experiment_uid, "source_experiment_uid"
        )
        target_experiment_uid = _validate_identifier(
            target_experiment_uid, "target_experiment_uid"
        )
        relation_type = _validate_identifier(relation_type, "relation_type")
        if source_experiment_uid == target_experiment_uid:
            raise ValueError("an experiment relation cannot target itself")
        metadata_object = _json_object(metadata)
        metadata_json = _json_text(metadata_object)
        created_at = _utc_now()
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            endpoints = self._connection.execute(
                """
                SELECT experiment_uid, namespace_id FROM experiments
                WHERE experiment_uid IN (?, ?)
                """,
                (source_experiment_uid, target_experiment_uid),
            ).fetchall()
            namespaces = {
                str(row["experiment_uid"]): str(row["namespace_id"])
                for row in endpoints
            }
            missing = [
                uid
                for uid in (source_experiment_uid, target_experiment_uid)
                if uid not in namespaces
            ]
            if missing:
                raise ValueError(f"unknown experiment UID(s): {', '.join(missing)}")
            if namespaces[source_experiment_uid] != namespaces[target_experiment_uid]:
                raise ValueError("experiment relations cannot cross namespaces")
            existing = self._connection.execute(
                """
                SELECT * FROM experiment_relations
                WHERE source_experiment_uid = ?
                  AND target_experiment_uid = ?
                  AND relation_type = ?
                """,
                (
                    source_experiment_uid,
                    target_experiment_uid,
                    relation_type,
                ),
            ).fetchone()
            if existing is not None:
                if str(existing["metadata_json"]) != metadata_json:
                    raise ValueError(
                        "experiment relation already exists with different metadata"
                    )
                self._connection.execute("COMMIT")
                return self._relation_from_row(existing)
            cursor = self._connection.execute(
                """
                INSERT INTO experiment_relations(
                    source_experiment_uid, target_experiment_uid,
                    relation_type, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    source_experiment_uid,
                    target_experiment_uid,
                    relation_type,
                    created_at,
                    metadata_json,
                ),
            )
            relation_id = int(cursor.lastrowid)
            self._connection.execute("COMMIT")
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        row = self._connection.execute(
            "SELECT * FROM experiment_relations WHERE id = ?", (relation_id,)
        ).fetchone()
        if row is None:  # pragma: no cover
            raise RuntimeError("experiment relation could not be reloaded")
        return self._relation_from_row(row)

    def list_experiment_relations(
        self,
        experiment_uid: str,
        *,
        direction: str = "both",
        relation_type: str | None = None,
    ) -> list[ExperimentRelationRecord]:
        if direction not in {"source", "target", "both"}:
            raise ValueError("direction must be one of: source, target, both")
        column_clause = {
            "source": "source_experiment_uid = ?",
            "target": "target_experiment_uid = ?",
            "both": "(source_experiment_uid = ? OR target_experiment_uid = ?)",
        }[direction]
        parameters: list[Any] = (
            [experiment_uid, experiment_uid]
            if direction == "both"
            else [experiment_uid]
        )
        if relation_type is not None:
            column_clause += " AND relation_type = ?"
            parameters.append(relation_type)
        rows = self._connection.execute(
            f"""
            SELECT * FROM experiment_relations
            WHERE {column_clause}
            ORDER BY id
            """,
            parameters,
        ).fetchall()
        return [self._relation_from_row(row) for row in rows]

    def export_rows(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.list_experiments()]

    def export_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.export_rows(), ensure_ascii=False, indent=indent)

    def export_tsv(self) -> str:
        output = io.StringIO(newline="")
        columns = (
            "id",
            "created_at",
            "candidate_hash",
            "git_commit",
            "backend",
            "suite",
            "status",
            "promotable",
            "duplicate_of_id",
            "aggregate_score",
            "note",
            "environment",
            "error_summary",
            "artifact_path",
            "result",
            "case_measurements",
        )
        writer = csv.DictWriter(output, fieldnames=columns, dialect="excel-tab")
        writer.writeheader()
        for row in self.export_rows():
            writer.writerow(
                {
                    column: (
                        _json_text(row[column])
                        if column in {"environment", "result", "case_measurements"}
                        else row[column]
                    )
                    for column in columns
                }
            )
        return output.getvalue()

    def format_table(self) -> str:
        columns = (
            "id",
            "status",
            "backend",
            "suite",
            "accepted",
            "phase",
            "score",
            "duplicate",
            "hash",
            "reason/note",
        )
        rows = [
            (
                str(record.id),
                record.status,
                record.backend,
                record.suite,
                "yes" if record.promotable else "no",
                str(record.result.get("promotion", {}).get("phase", "")),
                "" if record.aggregate_score is None else f"{record.aggregate_score:.6g}",
                "yes" if record.duplicate else "no",
                record.candidate_hash[:12],
                str(
                    record.result.get("promotion", {}).get("reason")
                    or record.error_summary
                    or record.note
                )[:72],
            )
            for record in self.list_experiments()
        ]
        widths = [
            max(len(columns[index]), *(len(row[index]) for row in rows))
            if rows
            else len(columns[index])
            for index in range(len(columns))
        ]
        header = "  ".join(
            column.ljust(widths[index]) for index, column in enumerate(columns)
        )
        separator = "  ".join("-" * width for width in widths)
        body = [
            "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
            for row in rows
        ]
        return "\n".join([header, separator, *body])

    def export(self, output_format: str) -> str:
        if output_format == "json":
            return self.export_json()
        if output_format == "tsv":
            return self.export_tsv()
        if output_format == "table":
            return self.format_table()
        raise ValueError("output_format must be one of: table, json, tsv")


__all__ = [
    "CandidateArtifactRecord",
    "CaseMeasurement",
    "EVALUATION_STATUSES",
    "ExperimentRelationRecord",
    "ExperimentRecord",
    "HistoryStore",
    "LEGACY_NAMESPACE_ID",
    "ResearchNamespaceRecord",
    "SCHEMA_VERSION",
]
