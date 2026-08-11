"""Coordinated production migration for the History/Controller V2 pair.

Opening either persisted V1/V2 store normally fails closed.  This coordinator
is the only production caller that supplies the opaque local-migration
capabilities: it first inspects both V2 databases with raw read-only SQLite,
proves every Controller record is terminal, fences both Store entry points, and
atomically publishes a verified checkpoint containing both databases, the
complete legacy artifact tree, the exact controller config bytes, and a full
offline-verifiable Git bundle.  It dry-runs the complete recovery unit before
the live stores are opened sequentially.

There is intentionally no single-database rollback API.  Once live migration
starts, every failure points at the immutable checkpoint and requires the two
databases, artifacts, config, and code to be restored together into an inactive
runtime before an operator switches it active.  A live partial failure keeps
the migration intents in place so legacy CLI/admin paths remain fenced.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import subprocess
import tempfile
from typing import Any, Iterator, Mapping
import uuid

from .autorun.states import TERMINAL_RUN_STATUSES
from .autorun.store import (
    ControllerStore,
    _V2_TO_V3_MIGRATION_CAPABILITY as _CONTROLLER_MIGRATION_CAPABILITY,
    _V2_TO_V3_MIGRATION_INTENT as _CONTROLLER_MIGRATION_INTENT,
)
from .history import (
    HistoryStore,
    _V2_TO_V3_MIGRATION_CAPABILITY as _HISTORY_MIGRATION_CAPABILITY,
    _V2_TO_V3_MIGRATION_INTENT as _HISTORY_MIGRATION_INTENT,
)


MIGRATION_CHECKPOINT_SCHEMA_VERSION = 2
MIGRATION_KIND = "history-controller-v2-to-v3"
ROLLBACK_SCOPE = "FULL_CHECKPOINT_ONLY"
_SOURCE_VERSION = 2
_TARGET_VERSION = 3
_MAX_MANIFEST_BYTES = 32 * 1024 * 1024
_MAX_CONFIG_BYTES = 2 * 1024 * 1024
_OPERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_HASH = re.compile(r"^[0-9a-f]{64}$")
_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:"
    r"[0-9]{2}\.[0-9]{3}Z$"
)
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REFERENCE_KEY_SUFFIXES = (
    "_file",
    "_path",
    "_ref",
    "_reference",
    "_refs",
    "_id",
    "_name",
    "_env",
    "_env_var",
)
_SECRET_KEY_TERMS = (
    "secret",
    "password",
    "passwd",
    "token",
    "api_key",
    "access_key",
    "private_key",
    "credential",
)

_CONTROLLER_V2_TABLES = ("runs", "iterations", "events")
_HISTORY_V2_TABLES = ("experiments", "case_measurements")

# These projections deliberately name the V2 payload columns that migration
# must preserve byte-for-value.  They let post-migration verification prove
# that no legacy scientific or workflow record changed in the small window
# between source revalidation and the two local Store migrations.  V3-only
# identities and the intentionally advanced experiments.schema_version field
# are excluded.
_CONTROLLER_V2_PROJECTION = {
    "runs": (
        "id",
        "status",
        "created_at",
        "updated_at",
        "deadline_epoch",
        "valid_candidates",
        "consecutive_failures",
        "stop_requested",
        "stop_reason",
        "config_json",
        "preflight_json",
        "initial_best_hash",
        "final_best_hash",
    ),
    "iterations": (
        "id",
        "run_id",
        "iteration_index",
        "status",
        "stage",
        "created_at",
        "updated_at",
        "parent_hash",
        "candidate_hash",
        "hypothesis",
        "rationale",
        "prompt_path",
        "raw_output_path",
        "candidate_path",
        "active_container",
        "experiment_ids_json",
        "result_json",
        "error",
        "outcome",
    ),
    "events": (
        "id",
        "run_id",
        "iteration_id",
        "created_at",
        "event",
        "details_json",
    ),
}
_HISTORY_V2_PROJECTION = {
    "experiments": (
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
        "environment_json",
        "error_summary",
        "artifact_path",
        "result_json",
    ),
    "case_measurements": (
        "id",
        "experiment_id",
        "case_name",
        "matched_ratio",
        "passed",
        "raw_samples_json",
        "baseline_samples_json",
        "metrics_json",
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_existing_file(value: str | Path, name: str) -> Path:
    supplied = Path(value)
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} does not identify an existing file") from exc
    if supplied.is_symlink() or not resolved.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    return resolved


def _canonical_existing_directory(value: str | Path, name: str) -> Path:
    supplied = Path(value)
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} does not identify an existing directory") from exc
    if supplied.is_symlink() or not resolved.is_dir():
        raise ValueError(f"{name} must be a real directory")
    return resolved


def _checkpoint_root(value: str | Path) -> Path:
    supplied = Path(value)
    supplied.mkdir(parents=True, exist_ok=True)
    return _canonical_existing_directory(supplied, "checkpoint_root")


def _strict_json_bytes(raw: bytes, *, name: str) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{name} contains duplicate keys")
            result[key] = value
        return result

    def constant(_value: str) -> Any:
        raise ValueError(f"{name} contains a non-finite number")

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc


def _reject_inline_secret_keys(value: Any, *, location: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("controller config keys must be strings")
            normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
            is_reference = normalized.endswith(_REFERENCE_KEY_SUFFIXES)
            if any(term in normalized for term in _SECRET_KEY_TERMS) and not is_reference:
                # Report the field identity, never its possibly sensitive value.
                raise ValueError(
                    f"controller config contains forbidden inline secret key "
                    f"{location}.{key}"
                )
            _reject_inline_secret_keys(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_inline_secret_keys(child, location=f"{location}[{index}]")


def _validate_config_bindings(
    value: Mapping[str, Any],
    *,
    repository: str,
    state_dir: str,
    controller_dir: str,
    expected_git_commit: str,
) -> None:
    expected = {
        "repository_dir": repository,
        "state_dir": state_dir,
        "controller_dir": controller_dir,
        "expected_git_commit": expected_git_commit,
    }
    for field, identity in expected.items():
        if value.get(field) != identity:
            raise ValueError(
                f"controller config {field} does not match the migration source"
            )


def _config_source_identity(
    config_path: str | Path,
    *,
    repository: str,
    state_dir: str,
    controller_dir: str,
    expected_git_commit: str,
) -> dict[str, Any]:
    path = _canonical_existing_file(config_path, "config_path")
    raw = path.read_bytes()
    if len(raw) > _MAX_CONFIG_BYTES:
        raise ValueError("controller config exceeds the migration size limit")
    value = _strict_json_bytes(raw, name="controller config")
    if not isinstance(value, dict):
        raise ValueError("controller config must be a JSON object")
    _reject_inline_secret_keys(value)
    _validate_config_bindings(
        value,
        repository=repository,
        state_dir=state_dir,
        controller_dir=controller_dir,
        expected_git_commit=expected_git_commit,
    )
    return {
        "source_path": str(path),
        "path": "config.json",
        "size_bytes": len(raw),
        "sha256": _sha256_file(path),
    }


def _run_git(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ("git", "-c", "core.fsmonitor=false", "-C", str(repository), *args),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise ValueError(f"git {' '.join(args[:2])} could not be executed") from exc
    if result.returncode != 0:
        # Git output can contain remote URLs or configured values; do not copy it
        # into migration evidence or errors.
        raise ValueError(f"git {' '.join(args[:2])} failed")
    return result


def _repository_source_identity(
    repository: str | Path, expected_git_commit: str
) -> dict[str, Any]:
    if not isinstance(expected_git_commit, str) or not _GIT_COMMIT.fullmatch(
        expected_git_commit
    ):
        raise ValueError("expected_git_commit must be 40 lowercase hex digits")
    supplied = Path(repository)
    if supplied.is_symlink():
        raise ValueError("repository must not be a symlink")
    root = _canonical_existing_directory(supplied, "repository")
    top = Path(
        _run_git(root, "rev-parse", "--show-toplevel").stdout.strip()
    ).resolve(strict=True)
    if top != root:
        raise ValueError("repository must identify the Git worktree root")
    head = _run_git(root, "rev-parse", "--verify", "HEAD^{commit}").stdout.strip()
    if head != expected_git_commit:
        raise ValueError("repository HEAD does not match expected_git_commit")
    status = _run_git(
        root, "status", "--porcelain=v1", "--untracked-files=all"
    ).stdout
    if status:
        raise ValueError("repository must be clean before migration")
    return {
        "source_path": str(root),
        "kind": "git_bundle_v1",
        "expected_git_commit": expected_git_commit,
        "head": head,
        "path": "code/repository.bundle",
    }


def _create_git_bundle(repository: Path, destination: Path) -> None:
    destination.parent.mkdir(mode=0o700)
    _run_git(repository, "bundle", "create", str(destination), "HEAD")
    if destination.is_symlink() or not destination.is_file():
        raise ValueError("git did not create a regular repository bundle")
    destination.chmod(0o600)
    _fsync_file(destination)
    _fsync_directory(destination.parent)


def _verify_git_bundle(bundle: Path, expected_git_commit: str) -> None:
    if bundle.is_symlink() or not bundle.is_file():
        raise ValueError("migration code bundle is missing")
    with tempfile.TemporaryDirectory(prefix="migration-bundle-verify-") as temporary:
        bare = Path(temporary) / "verify.git"
        subprocess.run(
            ("git", "init", "--bare", "--quiet", str(bare)),
            check=True,
            capture_output=True,
            timeout=120,
        )
        _run_git(bare, "bundle", "verify", str(bundle))
        heads = _run_git(bare, "bundle", "list-heads", str(bundle)).stdout.splitlines()
    advertised = {
        line.split(" ", 1)[0]
        for line in heads
        if line and " " in line
    }
    if expected_git_commit not in advertised:
        raise ValueError("migration code bundle does not advertise expected HEAD")


def _open_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def _logical_database_digest(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for statement in connection.iterdump():
        digest.update(statement.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _legacy_projection_digest(
    path: Path,
    projection: Mapping[str, tuple[str, ...]],
) -> str:
    """Hash V2 columns in stable row order, even after columns are added."""

    connection = _open_read_only(path)
    digest = hashlib.sha256()
    try:
        connection.execute("BEGIN")
        for table, columns in projection.items():
            available = {
                str(row[1])
                for row in connection.execute(
                    f'PRAGMA table_info("{table}")'
                ).fetchall()
            }
            missing = sorted(set(columns) - available)
            if missing:
                raise ValueError(
                    f"{path.name} is missing legacy {table} columns: "
                    + ", ".join(missing)
                )
            quoted = ", ".join(f'"{column}"' for column in columns)
            rows = connection.execute(
                f'SELECT {quoted} FROM "{table}" ORDER BY "id"'
            ).fetchall()
            digest.update(table.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_canonical_json(list(columns)).encode("utf-8"))
            digest.update(b"\n")
            for row in rows:
                digest.update(
                    _canonical_json([row[column] for column in columns]).encode(
                        "utf-8"
                    )
                )
                digest.update(b"\n")
        connection.execute("ROLLBACK")
    finally:
        connection.close()
    return digest.hexdigest()


def _history_v2_projection_digest(path: Path) -> str:
    return _legacy_projection_digest(path, _HISTORY_V2_PROJECTION)


def _controller_v2_projection_digest(path: Path) -> str:
    return _legacy_projection_digest(path, _CONTROLLER_V2_PROJECTION)


def _inspect_database(
    path: Path,
    *,
    required_tables: tuple[str, ...],
    expected_version: int | None = None,
) -> dict[str, Any]:
    connection = _open_read_only(path)
    try:
        connection.execute("BEGIN")
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if expected_version is not None and version != expected_version:
            raise ValueError(
                f"{path.name} schema is {version}; expected {expected_version}"
            )
        integrity = [
            str(row[0])
            for row in connection.execute("PRAGMA integrity_check").fetchall()
        ]
        if integrity != ["ok"]:
            raise ValueError(
                f"{path.name} integrity_check failed: " + "; ".join(integrity)
            )
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_keys:
            raise ValueError(f"{path.name} contains foreign-key violations")
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        missing = sorted(set(required_tables) - tables)
        if missing:
            raise ValueError(
                f"{path.name} is missing tables: " + ", ".join(missing)
            )
        counts = {
            table: int(
                connection.execute(
                    f'SELECT COUNT(*) FROM "{table}"'
                ).fetchone()[0]
            )
            for table in required_tables
        }
        logical_digest = _logical_database_digest(connection)
        connection.execute("ROLLBACK")
    finally:
        connection.close()
    return {
        "user_version": version,
        "integrity_check": "ok",
        "foreign_key_check": "ok",
        "logical_sha256": logical_digest,
        "table_counts": counts,
    }


def _controller_terminal_preflight(
    path: Path, *, expected_version: int = _SOURCE_VERSION
) -> dict[str, Any]:
    summary = _inspect_database(
        path,
        required_tables=_CONTROLLER_V2_TABLES,
        expected_version=expected_version,
    )
    connection = _open_read_only(path)
    try:
        connection.execute("BEGIN")
        placeholders = ",".join("?" for _ in TERMINAL_RUN_STATUSES)
        active_runs = connection.execute(
            f"""
            SELECT id, status FROM runs
            WHERE status NOT IN ({placeholders})
            ORDER BY id
            """,
            tuple(sorted(TERMINAL_RUN_STATUSES)),
        ).fetchall()
        active_iterations = connection.execute(
            """
            SELECT run_id, iteration_index, status, stage, active_container
            FROM iterations
            WHERE status <> 'COMPLETED'
               OR stage <> 'DONE'
               OR active_container IS NOT NULL
            ORDER BY run_id, iteration_index
            """
        ).fetchall()
        connection.execute("ROLLBACK")
    finally:
        connection.close()
    if active_runs or active_iterations:
        identifiers = [
            f"run {row['id']} ({row['status']})" for row in active_runs
        ] + [
            (
                f"iteration {row['run_id']}:{row['iteration_index']} "
                f"({row['status']}/{row['stage']})"
            )
            for row in active_iterations
        ]
        raise ValueError(
            "all Controller runs and iterations must be terminal before "
            "coordinated migration: " + ", ".join(identifiers)
        )
    return {
        **summary,
        "legacy_projection_sha256": _controller_v2_projection_digest(path),
        "terminal_state_check": "ok",
    }


def _legacy_artifact_references(
    history_db: Path, state_dir: Path
) -> list[dict[str, Any]]:
    connection = _open_read_only(history_db)
    try:
        connection.execute("BEGIN")
        rows = connection.execute(
            """
            SELECT id, candidate_hash, artifact_path
            FROM experiments ORDER BY id
            """
        ).fetchall()
        connection.execute("ROLLBACK")
    finally:
        connection.close()
    references: list[dict[str, Any]] = []
    for row in rows:
        candidate_hash = str(row["candidate_hash"])
        relative_text = str(row["artifact_path"])
        relative = PurePosixPath(relative_text)
        if (
            not _CANDIDATE_HASH.fullmatch(candidate_hash)
            or relative.is_absolute()
            or relative.as_posix() != relative_text
            or not relative.parts
            or relative.parts[0] != "artifacts"
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError(
                f"legacy experiment {row['id']} has an unsafe artifact identity"
            )
        path = state_dir.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"legacy artifact is missing for experiment {row['id']}"
            )
        digest = _sha256_file(path)
        if digest != candidate_hash:
            raise ValueError(
                f"legacy artifact hash mismatch for experiment {row['id']}"
            )
        references.append(
            {
                "experiment_id": int(row["id"]),
                "candidate_hash": candidate_hash,
                "path": relative_text,
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    return references


def _artifact_tree_files(state_dir: Path) -> list[dict[str, Any]]:
    artifacts = state_dir / "artifacts"
    if not artifacts.exists():
        return []
    if artifacts.is_symlink() or not artifacts.is_dir():
        raise ValueError("legacy artifacts must be a real directory")
    descendants = list(artifacts.rglob("*"))
    if any(path.is_symlink() for path in descendants):
        raise ValueError("legacy artifacts must not contain symlinks")
    records = []
    for path in sorted(item for item in descendants if item.is_file()):
        relative = path.relative_to(state_dir).as_posix()
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return records


def preflight_v2_to_v3_migration(
    *,
    history_db: str | Path,
    controller_db: str | Path,
    state_dir: str | Path,
    config_path: str | Path,
    repository: str | Path,
    expected_git_commit: str,
) -> dict[str, Any]:
    """Read-only proof that the exact V2 pair is safe to migrate.

    This function never constructs either Store class and therefore cannot
    trigger an implicit schema migration.
    """

    history_path = _canonical_existing_file(history_db, "history_db")
    controller_path = _canonical_existing_file(controller_db, "controller_db")
    state_path = _canonical_existing_directory(state_dir, "state_dir")
    if history_path == controller_path:
        raise ValueError("History and Controller databases must be distinct")
    if history_path != state_path / "history.sqlite3":
        raise ValueError("history_db must be state_dir/history.sqlite3")
    if controller_path.name != "controller.sqlite3":
        raise ValueError("controller_db must be controller_dir/controller.sqlite3")
    history_summary = _inspect_database(
        history_path,
        required_tables=_HISTORY_V2_TABLES,
        expected_version=_SOURCE_VERSION,
    )
    history_summary["legacy_projection_sha256"] = (
        _history_v2_projection_digest(history_path)
    )
    controller_summary = _controller_terminal_preflight(controller_path)
    references = _legacy_artifact_references(history_path, state_path)
    artifact_files = _artifact_tree_files(state_path)
    code_identity = _repository_source_identity(
        repository, expected_git_commit
    )
    config_identity = _config_source_identity(
        config_path,
        repository=code_identity["source_path"],
        state_dir=str(state_path),
        controller_dir=str(controller_path.parent),
        expected_git_commit=expected_git_commit,
    )
    return {
        "schema_version": MIGRATION_CHECKPOINT_SCHEMA_VERSION,
        "kind": MIGRATION_KIND,
        "status": "READY",
        "source_version": _SOURCE_VERSION,
        "target_version": _TARGET_VERSION,
        "history_db": str(history_path),
        "controller_db": str(controller_path),
        "state_dir": str(state_path),
        "config_path": config_identity["source_path"],
        "repository": code_identity["source_path"],
        "expected_git_commit": expected_git_commit,
        "databases": {
            "history": history_summary,
            "controller": controller_summary,
        },
        "artifacts": {
            "referenced_count": len(references),
            "file_count": len(artifact_files),
            "total_bytes": sum(item["size_bytes"] for item in artifact_files),
            "references": references,
            "files": artifact_files,
        },
        "config": config_identity,
        "code": code_identity,
        "rollback_scope": ROLLBACK_SCOPE,
        "single_database_rollback_supported": False,
    }


def _sqlite_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_connection = _open_read_only(source)
    target_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(target_connection)
        journal_mode = target_connection.execute(
            "PRAGMA journal_mode = DELETE"
        ).fetchone()
        if (
            journal_mode is None
            or str(journal_mode[0]).lower() != "delete"
        ):
            raise RuntimeError(
                "SQLite checkpoint backup could not be normalized "
                "to a single-file journal"
            )
    finally:
        target_connection.close()
        source_connection.close()
    destination.chmod(0o600)
    _fsync_file(destination)


def _copy_artifact_tree(source: Path, destination: Path) -> None:
    if source.exists():
        shutil.copytree(source, destination)
    else:
        destination.mkdir()
    for path in destination.rglob("*"):
        if path.is_dir():
            path.chmod(0o700)
        else:
            path.chmod(0o600)
            _fsync_file(path)
    destination.chmod(0o700)


def _checkpoint_file_records(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path != root / "manifest.json"
    ]


def _strict_manifest(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise ValueError("migration manifest exceeds its size limit")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError("migration manifest contains duplicate keys")
            result[key] = value
        return result

    def constant(_value: str) -> Any:
        raise ValueError("migration manifest contains a non-finite number")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("migration manifest is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("migration manifest must be an object")
    return value


def _relative_checkpoint_file(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("checkpoint file path must be non-empty")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.parts[0]
        not in {
            "history.sqlite3",
            "controller.sqlite3",
            "artifacts",
            "config.json",
            "code",
        }
    ):
        raise ValueError("checkpoint file path is unsafe")
    if path.parts[0] in {
        "history.sqlite3",
        "controller.sqlite3",
        "config.json",
    } and len(
        path.parts
    ) != 1:
        raise ValueError("checkpoint singleton path must be top-level")
    if path.parts[0] == "code" and path.parts != (
        "code",
        "repository.bundle",
    ):
        raise ValueError("checkpoint code path is not recognized")
    return path


def _verify_checkpoint_path(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("migration checkpoint must be a real directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("migration checkpoint manifest is missing")
    manifest = _strict_manifest(manifest_path)
    expected_fields = {
        "schema_version",
        "kind",
        "operation_id",
        "created_at",
        "source",
        "source_versions",
        "target_version",
        "databases",
        "artifacts",
        "config",
        "code",
        "files",
        "rollback_scope",
        "single_database_rollback_supported",
    }
    if set(manifest) != expected_fields:
        raise ValueError("migration manifest fields do not match its schema")
    if (
        manifest["schema_version"] != MIGRATION_CHECKPOINT_SCHEMA_VERSION
        or manifest["kind"] != MIGRATION_KIND
        or manifest["source_versions"]
        != {"history": _SOURCE_VERSION, "controller": _SOURCE_VERSION}
        or manifest["target_version"] != _TARGET_VERSION
        or manifest["rollback_scope"] != ROLLBACK_SCOPE
        or manifest["single_database_rollback_supported"] is not False
        or not isinstance(manifest["operation_id"], str)
        or not _OPERATION_ID.fullmatch(manifest["operation_id"])
        or not isinstance(manifest["created_at"], str)
        or not _TIMESTAMP.fullmatch(manifest["created_at"])
    ):
        raise ValueError("migration manifest identity is invalid")
    source = manifest["source"]
    if not isinstance(source, dict) or set(source) != {
        "history_db",
        "controller_db",
        "state_dir",
        "config_path",
        "repository",
        "expected_git_commit",
    }:
        raise ValueError("migration manifest source is invalid")
    for field, value in source.items():
        if field == "expected_git_commit":
            if not isinstance(value, str) or not _GIT_COMMIT.fullmatch(value):
                raise ValueError("migration source Git commit is invalid")
            continue
        if (
            not isinstance(value, str)
            or not value
            or not Path(value).is_absolute()
            or ".." in Path(value).parts
        ):
            raise ValueError(
                f"migration manifest source {field} must be an absolute path"
            )
    rows = manifest["files"]
    if not isinstance(rows, list):
        raise ValueError("migration manifest files must be a list")
    expected: dict[str, tuple[int, str]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("migration checkpoint file record is invalid")
        relative = _relative_checkpoint_file(row["path"]).as_posix()
        if relative in expected:
            raise ValueError("migration checkpoint contains duplicate paths")
        size = row["size_bytes"]
        digest = row["sha256"]
        if (
            type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
        ):
            raise ValueError("migration checkpoint file metadata is invalid")
        expected[relative] = (size, digest)
    descendants = list(root.rglob("*"))
    if any(path.is_symlink() for path in descendants):
        raise ValueError("migration checkpoint must not contain symlinks")
    actual = {
        path.relative_to(root).as_posix()
        for path in descendants
        if path.is_file() and path != manifest_path
    }
    if actual != set(expected):
        raise ValueError("migration manifest does not exactly cover checkpoint files")
    for relative, (size, digest) in expected.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        if path.stat().st_size != size or _sha256_file(path) != digest:
            raise ValueError(f"migration checkpoint file is corrupted: {relative}")
    for required in (
        "history.sqlite3",
        "controller.sqlite3",
        "config.json",
        "code/repository.bundle",
    ):
        if required not in actual:
            raise ValueError("migration checkpoint is missing a rollback component")
    if not (root / "artifacts").is_dir():
        raise ValueError("migration checkpoint is missing legacy artifacts")

    database_manifest = manifest["databases"]
    if not isinstance(database_manifest, dict) or set(database_manifest) != {
        "history",
        "controller",
    }:
        raise ValueError("migration database manifest is invalid")
    database_summaries = {
        "history": _inspect_database(
            root / "history.sqlite3",
            required_tables=_HISTORY_V2_TABLES,
            expected_version=_SOURCE_VERSION,
        ),
        "controller": _controller_terminal_preflight(
            root / "controller.sqlite3"
        ),
    }
    database_summaries["history"]["legacy_projection_sha256"] = (
        _history_v2_projection_digest(root / "history.sqlite3")
    )
    for name, summary in database_summaries.items():
        expected_summary = database_manifest.get(name)
        if not isinstance(expected_summary, dict):
            raise ValueError(f"migration {name} database summary is missing")
        expected_summary_fields = {
            "user_version",
            "integrity_check",
            "foreign_key_check",
            "logical_sha256",
            "legacy_projection_sha256",
            "table_counts",
            "path",
            "size_bytes",
            "sha256",
        }
        if name == "controller":
            expected_summary_fields.add("terminal_state_check")
        if set(expected_summary) != expected_summary_fields:
            raise ValueError(
                f"migration {name} database summary fields are invalid"
            )
        for field in (
            "user_version",
            "integrity_check",
            "foreign_key_check",
            "logical_sha256",
            "legacy_projection_sha256",
            "table_counts",
        ):
            if expected_summary.get(field) != summary.get(field):
                raise ValueError(
                    f"migration {name} database summary does not match bytes"
                )
        if name == "controller" and expected_summary.get(
            "terminal_state_check"
        ) != summary.get("terminal_state_check"):
            raise ValueError(
                "migration controller terminal-state summary does not match bytes"
            )
        file_name = f"{name}.sqlite3"
        if (
            expected_summary.get("path") != file_name
            or expected_summary.get("size_bytes")
            != (root / file_name).stat().st_size
            or expected_summary.get("sha256") != _sha256_file(root / file_name)
        ):
            raise ValueError(
                f"migration {name} database file identity is invalid"
            )

    artifact_manifest = manifest["artifacts"]
    if not isinstance(artifact_manifest, dict) or set(artifact_manifest) != {
        "root",
        "referenced_count",
        "file_count",
        "total_bytes",
        "references",
        "files",
    }:
        raise ValueError("migration artifact manifest is invalid")
    references = _legacy_artifact_references(
        root / "history.sqlite3", root
    )
    artifact_files = _artifact_tree_files(root)
    expected_artifacts = {
        "root": "artifacts",
        "referenced_count": len(references),
        "file_count": len(artifact_files),
        "total_bytes": sum(item["size_bytes"] for item in artifact_files),
        "references": references,
        "files": artifact_files,
    }
    if artifact_manifest != expected_artifacts:
        raise ValueError("migration artifact manifest does not match checkpoint")

    config_manifest = manifest["config"]
    expected_config_fields = {"source_path", "path", "size_bytes", "sha256"}
    if (
        not isinstance(config_manifest, dict)
        or set(config_manifest) != expected_config_fields
        or config_manifest.get("source_path") != source["config_path"]
        or config_manifest.get("path") != "config.json"
        or config_manifest.get("size_bytes") != (root / "config.json").stat().st_size
        or config_manifest.get("sha256") != _sha256_file(root / "config.json")
    ):
        raise ValueError("migration config manifest does not match checkpoint")
    config_raw = (root / "config.json").read_bytes()
    if len(config_raw) > _MAX_CONFIG_BYTES:
        raise ValueError("checkpoint config exceeds its size limit")
    config_value = _strict_json_bytes(config_raw, name="checkpoint config")
    if not isinstance(config_value, dict):
        raise ValueError("checkpoint config must be a JSON object")
    _reject_inline_secret_keys(config_value, location="checkpoint.config")
    _validate_config_bindings(
        config_value,
        repository=source["repository"],
        state_dir=source["state_dir"],
        controller_dir=str(Path(source["controller_db"]).parent),
        expected_git_commit=source["expected_git_commit"],
    )

    code_manifest = manifest["code"]
    expected_code_fields = {
        "source_path",
        "kind",
        "expected_git_commit",
        "head",
        "path",
        "size_bytes",
        "sha256",
        "offline_verify",
    }
    bundle = root / "code" / "repository.bundle"
    if (
        not isinstance(code_manifest, dict)
        or set(code_manifest) != expected_code_fields
        or code_manifest.get("source_path") != source["repository"]
        or code_manifest.get("kind") != "git_bundle_v1"
        or code_manifest.get("expected_git_commit")
        != source["expected_git_commit"]
        or code_manifest.get("head") != source["expected_git_commit"]
        or code_manifest.get("path") != "code/repository.bundle"
        or code_manifest.get("size_bytes") != bundle.stat().st_size
        or code_manifest.get("sha256") != _sha256_file(bundle)
        or code_manifest.get("offline_verify") != "git_bundle_verify_ok"
    ):
        raise ValueError("migration code manifest does not match checkpoint")
    _verify_git_bundle(bundle, source["expected_git_commit"])
    return {
        "schema_version": MIGRATION_CHECKPOINT_SCHEMA_VERSION,
        "kind": MIGRATION_KIND,
        "status": "VERIFIED",
        "operation_id": manifest["operation_id"],
        "checkpoint": str(root),
        "manifest_digest": "sha256:" + _sha256_file(manifest_path),
        "databases": database_summaries,
        "artifacts": expected_artifacts,
        "config": config_manifest,
        "code": code_manifest,
        "rollback_scope": ROLLBACK_SCOPE,
        "single_database_rollback_supported": False,
        "manifest": manifest,
    }


def verify_v2_migration_checkpoint(
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Verify a published V2 migration checkpoint without modifying it."""

    supplied = Path(checkpoint_dir)
    if supplied.is_symlink():
        raise ValueError("migration checkpoint must not be a symlink")
    root = supplied.resolve(strict=True)
    return _verify_checkpoint_path(root)


def _create_checkpoint(
    *,
    preflight: Mapping[str, Any],
    checkpoint_root: Path,
    operation_id: str,
) -> dict[str, Any]:
    destination = checkpoint_root / f"v2-to-v3-{operation_id}"
    if destination.exists() or destination.is_symlink():
        raise ValueError("migration checkpoint destination already exists")
    temporary = checkpoint_root / f".tmp-v2-to-v3-{operation_id}"
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("migration checkpoint staging path already exists")
    temporary.mkdir(mode=0o700)
    try:
        history_source = Path(str(preflight["history_db"]))
        controller_source = Path(str(preflight["controller_db"]))
        state_dir = Path(str(preflight["state_dir"]))
        config_source = Path(str(preflight["config_path"]))
        repository = Path(str(preflight["repository"]))
        history_backup = temporary / "history.sqlite3"
        controller_backup = temporary / "controller.sqlite3"
        _sqlite_backup(history_source, history_backup)
        _sqlite_backup(controller_source, controller_backup)
        _copy_artifact_tree(
            state_dir / "artifacts", temporary / "artifacts"
        )
        config_backup = temporary / "config.json"
        shutil.copyfile(config_source, config_backup)
        config_backup.chmod(0o600)
        _fsync_file(config_backup)
        bundle = temporary / "code" / "repository.bundle"
        _create_git_bundle(repository, bundle)
        _verify_git_bundle(bundle, str(preflight["expected_git_commit"]))
        history_summary = _inspect_database(
            history_backup,
            required_tables=_HISTORY_V2_TABLES,
            expected_version=_SOURCE_VERSION,
        )
        history_summary["legacy_projection_sha256"] = (
            _history_v2_projection_digest(history_backup)
        )
        controller_summary = _controller_terminal_preflight(controller_backup)
        references = _legacy_artifact_references(history_backup, temporary)
        artifact_files = _artifact_tree_files(temporary)
        database_summaries = {
            "history": {
                **history_summary,
                "path": "history.sqlite3",
                "size_bytes": history_backup.stat().st_size,
                "sha256": _sha256_file(history_backup),
            },
            "controller": {
                **controller_summary,
                "path": "controller.sqlite3",
                "size_bytes": controller_backup.stat().st_size,
                "sha256": _sha256_file(controller_backup),
            },
        }
        manifest = {
            "schema_version": MIGRATION_CHECKPOINT_SCHEMA_VERSION,
            "kind": MIGRATION_KIND,
            "operation_id": operation_id,
            "created_at": _utc_now(),
            "source": {
                "history_db": str(history_source),
                "controller_db": str(controller_source),
                "state_dir": str(state_dir),
                "config_path": str(config_source),
                "repository": str(repository),
                "expected_git_commit": str(preflight["expected_git_commit"]),
            },
            "source_versions": {
                "history": _SOURCE_VERSION,
                "controller": _SOURCE_VERSION,
            },
            "target_version": _TARGET_VERSION,
            "databases": database_summaries,
            "artifacts": {
                "root": "artifacts",
                "referenced_count": len(references),
                "file_count": len(artifact_files),
                "total_bytes": sum(
                    item["size_bytes"] for item in artifact_files
                ),
                "references": references,
                "files": artifact_files,
            },
            "config": dict(preflight["config"]),
            "code": {
                **dict(preflight["code"]),
                "size_bytes": bundle.stat().st_size,
                "sha256": _sha256_file(bundle),
                "offline_verify": "git_bundle_verify_ok",
            },
            "files": _checkpoint_file_records(temporary),
            "rollback_scope": ROLLBACK_SCOPE,
            "single_database_rollback_supported": False,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                manifest,
                sort_keys=True,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest_path.chmod(0o600)
        _fsync_file(manifest_path)
        _fsync_directory(temporary / "artifacts")
        _fsync_directory(temporary / "code")
        _fsync_directory(temporary)
        _verify_checkpoint_path(temporary)
        os.replace(temporary, destination)
        _fsync_directory(checkpoint_root)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return verify_v2_migration_checkpoint(destination)


def _live_sources_match_checkpoint(
    preflight: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> None:
    history_path = Path(str(preflight["history_db"]))
    controller_path = Path(str(preflight["controller_db"]))
    state_dir = Path(str(preflight["state_dir"]))
    live_config = _config_source_identity(
        str(preflight["config_path"]),
        repository=str(preflight["repository"]),
        state_dir=str(preflight["state_dir"]),
        controller_dir=str(controller_path.parent),
        expected_git_commit=str(preflight["expected_git_commit"]),
    )
    live_code = _repository_source_identity(
        str(preflight["repository"]),
        str(preflight["expected_git_commit"]),
    )
    live_history = _inspect_database(
        history_path,
        required_tables=_HISTORY_V2_TABLES,
        expected_version=_SOURCE_VERSION,
    )
    live_history["legacy_projection_sha256"] = (
        _history_v2_projection_digest(history_path)
    )
    live_controller = _controller_terminal_preflight(controller_path)
    for name, live in (
        ("history", live_history),
        ("controller", live_controller),
    ):
        frozen = checkpoint["databases"][name]
        if (
            live["logical_sha256"] != frozen["logical_sha256"]
            or live["legacy_projection_sha256"]
            != frozen["legacy_projection_sha256"]
        ):
            raise ValueError(
                f"live {name} database changed after checkpoint publication"
            )
    references = _legacy_artifact_references(history_path, state_dir)
    artifact_files = _artifact_tree_files(state_dir)
    if (
        references != checkpoint["artifacts"]["references"]
        or artifact_files != checkpoint["artifacts"]["files"]
    ):
        raise ValueError("legacy artifacts changed after checkpoint publication")
    if live_config != checkpoint["config"]:
        raise ValueError("controller config changed after checkpoint publication")
    frozen_code = checkpoint["code"]
    for field in (
        "source_path",
        "kind",
        "expected_git_commit",
        "head",
        "path",
    ):
        if live_code.get(field) != frozen_code.get(field):
            raise ValueError("repository identity changed after checkpoint publication")


def _dry_run_checkpoint_migration(
    checkpoint: Mapping[str, Any], checkpoint_root: Path
) -> None:
    source = Path(str(checkpoint["checkpoint"]))
    scratch = Path(
        tempfile.mkdtemp(prefix=".v2-to-v3-dry-run-", dir=checkpoint_root)
    )
    scratch.chmod(0o700)
    try:
        shutil.copy2(source / "history.sqlite3", scratch / "history.sqlite3")
        shutil.copy2(
            source / "controller.sqlite3", scratch / "controller.sqlite3"
        )
        shutil.copytree(source / "artifacts", scratch / "artifacts")
        shutil.copy2(source / "config.json", scratch / "config.json")
        shutil.copytree(source / "code", scratch / "code")
        config_value = _strict_json_bytes(
            (scratch / "config.json").read_bytes(), name="dry-run config"
        )
        if not isinstance(config_value, dict):
            raise RuntimeError("dry-run config is not an object")
        _reject_inline_secret_keys(config_value, location="dry-run.config")
        manifest_source = checkpoint["manifest"]["source"]
        _validate_config_bindings(
            config_value,
            repository=manifest_source["repository"],
            state_dir=manifest_source["state_dir"],
            controller_dir=str(
                Path(manifest_source["controller_db"]).parent
            ),
            expected_git_commit=manifest_source["expected_git_commit"],
        )
        _verify_git_bundle(
            scratch / "code" / "repository.bundle",
            str(checkpoint["code"]["expected_git_commit"]),
        )
        with HistoryStore(
            scratch / "history.sqlite3",
            scratch,
            _v2_to_v3_migration_capability=_HISTORY_MIGRATION_CAPABILITY,
        ) as history:
            if history.schema_version != _TARGET_VERSION:
                raise RuntimeError("History dry-run did not reach schema V3")
        with ControllerStore(
            scratch / "controller.sqlite3",
            _v2_to_v3_migration_capability=_CONTROLLER_MIGRATION_CAPABILITY,
        ) as controller:
            version = int(
                controller.connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if version != _TARGET_VERSION:
                raise RuntimeError("Controller dry-run did not reach schema V3")
        _inspect_database(
            scratch / "history.sqlite3",
            required_tables=(
                "experiments",
                "case_measurements",
                "research_namespaces",
                "candidate_artifacts",
                "experiment_relations",
            ),
            expected_version=_TARGET_VERSION,
        )
        _inspect_database(
            scratch / "controller.sqlite3",
            required_tables=(
                "runs",
                "iterations",
                "events",
                "proposal_attempts",
                "evaluation_attempts",
            ),
            expected_version=_TARGET_VERSION,
        )
        if (
            _history_v2_projection_digest(scratch / "history.sqlite3")
            != checkpoint["databases"]["history"][
                "legacy_projection_sha256"
            ]
            or _controller_v2_projection_digest(
                scratch / "controller.sqlite3"
            )
            != checkpoint["databases"]["controller"][
                "legacy_projection_sha256"
            ]
        ):
            raise RuntimeError("dry-run migration changed legacy V2 records")
    finally:
        shutil.rmtree(scratch)


def _raw_versions(history_db: Path, controller_db: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, path in (("history", history_db), ("controller", controller_db)):
        try:
            connection = _open_read_only(path)
            try:
                result[name] = int(
                    connection.execute("PRAGMA user_version").fetchone()[0]
                )
            finally:
                connection.close()
        except Exception as exc:  # evidence only; preserve the primary failure
            result[name] = f"UNAVAILABLE:{type(exc).__name__}"
    return result


def production_cli_schema_guard(
    *, history_db: str | Path, controller_db: str | Path
) -> dict[str, int | None]:
    """Prevent normal CLI commands from triggering one-store migration.

    Missing/version-zero files are new installations and remain supported.
    Any persisted legacy schema is blocked.  In particular, a V2/V3 mix is
    treated as evidence of a partial migration and never repaired in place.
    """

    versions: dict[str, int | None] = {}
    for name, supplied in (
        ("history", Path(history_db)),
        ("controller", Path(controller_db)),
    ):
        if supplied.is_symlink():
            raise ValueError(f"{name} database must not be a symlink")
        if not supplied.exists():
            versions[name] = None
            continue
        path = _canonical_existing_file(supplied, f"{name}_db")
        connection = _open_read_only(path)
        try:
            versions[name] = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
        finally:
            connection.close()

    legacy = {
        name: version
        for name, version in versions.items()
        if version in {1, _SOURCE_VERSION}
    }
    if not legacy:
        return versions
    if set(legacy.values()) == {_SOURCE_VERSION} and len(legacy) == 2:
        raise ValueError(
            "History and Controller are schema V2; normal commands are "
            "blocked until `kernel-autoresearch migrate-v3` completes"
        )
    raise ValueError(
        "History/Controller schemas are legacy or mixed "
        f"({versions}); do not open either Store independently. Restore the "
        "full migration checkpoint if migration already started, or first "
        "prepare one matching offline V2 pair before migrate-v3."
    )


@contextmanager
def _migration_lock(checkpoint_root: Path) -> Iterator[None]:
    lock_path = checkpoint_root / ".v2-to-v3-migration.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another coordinated migration is active") from exc
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _publish_migration_intents(
    *,
    state_dir: Path,
    controller_db: Path,
    operation_id: str,
    checkpoint_root: Path,
) -> tuple[tuple[Path, ...], bytes]:
    payload = (
        _canonical_json(
            {
                "schema_version": MIGRATION_CHECKPOINT_SCHEMA_VERSION,
                "kind": MIGRATION_KIND,
                "operation_id": operation_id,
                "checkpoint_root": str(checkpoint_root),
                "rollback_scope": ROLLBACK_SCOPE,
            }
        ).encode("utf-8")
        + b"\n"
    )
    paths = tuple(
        dict.fromkeys(
            (
                state_dir / _HISTORY_MIGRATION_INTENT,
                controller_db.parent / _CONTROLLER_MIGRATION_INTENT,
            )
        )
    )
    published: list[Path] = []
    try:
        for path in paths:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(descriptor)
            published.append(path)
            _fsync_directory(path.parent)
    except BaseException:
        for path in reversed(published):
            try:
                path.unlink()
                _fsync_directory(path.parent)
            except OSError:
                pass
        raise ValueError(
            "could not exclusively publish the coordinated migration intent"
        )
    return paths, payload


def _remove_migration_intents(paths: tuple[Path, ...], payload: bytes) -> None:
    for path in paths:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RuntimeError("migration intent identity changed; refusing to remove it")
    for path in paths:
        path.unlink()
        _fsync_directory(path.parent)


class CoordinatedMigrationError(RuntimeError):
    """Failure contract that never advertises a one-database rollback."""

    def __init__(
        self,
        message: str,
        *,
        phase: str,
        checkpoint_path: Path | None = None,
        observed_versions: Mapping[str, Any] | None = None,
        live_migration_started: bool = False,
    ) -> None:
        recovery = (
            " Restore both databases, legacy artifacts, controller config, and "
            "repository code from the migration checkpoint together; never "
            "roll back one database or any other component independently."
            if live_migration_started
            else " No live Store migration was intentionally started."
        )
        super().__init__(message + recovery)
        self.phase = phase
        self.checkpoint_path = checkpoint_path
        self.observed_versions = dict(observed_versions or {})
        self.live_migration_started = bool(live_migration_started)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MIGRATION_CHECKPOINT_SCHEMA_VERSION,
            "kind": MIGRATION_KIND,
            "status": "FAILED",
            "phase": self.phase,
            "error": str(self),
            "checkpoint": (
                str(self.checkpoint_path)
                if self.checkpoint_path is not None
                else None
            ),
            "observed_versions": dict(self.observed_versions),
            "live_migration_started": self.live_migration_started,
            "rollback_scope": ROLLBACK_SCOPE,
            "single_database_rollback_supported": False,
        }


def coordinate_v2_to_v3_migration(
    *,
    history_db: str | Path,
    controller_db: str | Path,
    state_dir: str | Path,
    config_path: str | Path,
    repository: str | Path,
    expected_git_commit: str,
    checkpoint_root: str | Path,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Migrate one exact History/Controller V2 pair under one checkpoint.

    The function keeps the checkpoint after success.  It never automatically
    restores a partially migrated live runtime because doing so cannot be made
    atomic across two SQLite files and an artifact tree.
    """

    selected_operation_id = operation_id or uuid.uuid4().hex
    if not isinstance(selected_operation_id, str) or not _OPERATION_ID.fullmatch(
        selected_operation_id
    ):
        raise ValueError("operation_id must be a safe bounded identifier")
    root = _checkpoint_root(checkpoint_root)
    history_path: Path | None = None
    controller_path: Path | None = None
    checkpoint_path: Path | None = None
    intent_paths: tuple[Path, ...] = ()
    intent_payload: bytes | None = None
    phase = "preflight"
    live_started = False
    try:
        with _migration_lock(root):
            preflight = preflight_v2_to_v3_migration(
                history_db=history_db,
                controller_db=controller_db,
                state_dir=state_dir,
                config_path=config_path,
                repository=repository,
                expected_git_commit=expected_git_commit,
            )
            history_path = Path(preflight["history_db"])
            controller_path = Path(preflight["controller_db"])
            state_path = Path(preflight["state_dir"])

            phase = "migration-intent"
            intent_paths, intent_payload = _publish_migration_intents(
                state_dir=state_path,
                controller_db=controller_path,
                operation_id=selected_operation_id,
                checkpoint_root=root,
            )

            phase = "checkpoint"
            checkpoint = _create_checkpoint(
                preflight=preflight,
                checkpoint_root=root,
                operation_id=selected_operation_id,
            )
            checkpoint_path = Path(checkpoint["checkpoint"])

            phase = "checkpoint-dry-run"
            _dry_run_checkpoint_migration(checkpoint, root)

            phase = "source-revalidation"
            _live_sources_match_checkpoint(preflight, checkpoint)

            phase = "live-history-migration"
            live_started = True
            with HistoryStore(
                history_path,
                state_path,
                _v2_to_v3_migration_capability=_HISTORY_MIGRATION_CAPABILITY,
            ) as history:
                if history.schema_version != _TARGET_VERSION:
                    raise RuntimeError("History Store did not reach schema V3")

            phase = "live-controller-migration"
            with ControllerStore(
                controller_path,
                _v2_to_v3_migration_capability=_CONTROLLER_MIGRATION_CAPABILITY,
            ) as controller:
                controller_version = int(
                    controller.connection.execute(
                        "PRAGMA user_version"
                    ).fetchone()[0]
                )
                if controller_version != _TARGET_VERSION:
                    raise RuntimeError("Controller Store did not reach schema V3")

            phase = "post-migration-verification"
            history_after = _inspect_database(
                history_path,
                required_tables=(
                    "experiments",
                    "case_measurements",
                    "research_namespaces",
                    "candidate_artifacts",
                    "experiment_relations",
                ),
                expected_version=_TARGET_VERSION,
            )
            controller_after = _inspect_database(
                controller_path,
                required_tables=(
                    "runs",
                    "iterations",
                    "events",
                    "proposal_attempts",
                    "evaluation_attempts",
                ),
                expected_version=_TARGET_VERSION,
            )
            if (
                _history_v2_projection_digest(history_path)
                != checkpoint["databases"]["history"][
                    "legacy_projection_sha256"
                ]
                or _controller_v2_projection_digest(controller_path)
                != checkpoint["databases"]["controller"][
                    "legacy_projection_sha256"
                ]
            ):
                raise RuntimeError(
                    "live migration changed legacy V2 records relative to "
                    "the published checkpoint"
                )
            phase = "migration-intent-release"
            _remove_migration_intents(intent_paths, intent_payload)
            intent_paths = ()
            intent_payload = None
            return {
                "schema_version": MIGRATION_CHECKPOINT_SCHEMA_VERSION,
                "kind": MIGRATION_KIND,
                "status": "SUCCESS",
                "operation_id": selected_operation_id,
                "checkpoint": str(checkpoint_path),
                "checkpoint_manifest_digest": checkpoint["manifest_digest"],
                "versions_before": {
                    "history": _SOURCE_VERSION,
                    "controller": _SOURCE_VERSION,
                },
                "versions_after": {
                    "history": history_after["user_version"],
                    "controller": controller_after["user_version"],
                },
                "artifacts": checkpoint["artifacts"],
                "config": checkpoint["config"],
                "code": checkpoint["code"],
                "rollback_scope": ROLLBACK_SCOPE,
                "single_database_rollback_supported": False,
            }
    except CoordinatedMigrationError:
        raise
    except BaseException as exc:
        if intent_paths and intent_payload is not None and not live_started:
            try:
                _remove_migration_intents(intent_paths, intent_payload)
                intent_paths = ()
            except Exception:
                # A mismatched or unremovable intent is itself a reason to stay
                # fail-closed; preserve the primary migration failure below.
                pass
        published = root / f"v2-to-v3-{selected_operation_id}"
        if (
            checkpoint_path is None
            and published.is_dir()
            and not published.is_symlink()
        ):
            checkpoint_path = published
        versions = (
            _raw_versions(history_path, controller_path)
            if history_path is not None and controller_path is not None
            else {}
        )
        raise CoordinatedMigrationError(
            f"coordinated V2 to V3 migration failed during {phase}: "
            f"{type(exc).__name__}: {exc}",
            phase=phase,
            checkpoint_path=checkpoint_path,
            observed_versions=versions,
            live_migration_started=live_started,
        ) from exc


# Concise alias for callers that do not need the implementation-oriented name.
migrate_v2_to_v3 = coordinate_v2_to_v3_migration


__all__ = [
    "CoordinatedMigrationError",
    "MIGRATION_CHECKPOINT_SCHEMA_VERSION",
    "MIGRATION_KIND",
    "ROLLBACK_SCOPE",
    "coordinate_v2_to_v3_migration",
    "migrate_v2_to_v3",
    "preflight_v2_to_v3_migration",
    "production_cli_schema_guard",
    "verify_v2_migration_checkpoint",
]
