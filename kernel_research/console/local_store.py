"""Mac-local drafts, operation mirrors, and tamper-evident audit records."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import tempfile
from typing import Any, Mapping

from ..platform.canonical import canonical_json_bytes, canonical_sha256
from ..platform.proposal import CandidateBundle, TRITON_PYTHON_BUNDLE_LIMITS
from .protocol import DraftV1, OperationReceiptV1, PreparedOperationV1


LOCAL_SCHEMA_VERSION = 1
_STATUS_RANK = {
    "PREPARED": 0,
    "EXECUTING": 1,
    "SUCCEEDED": 2,
    "FAILED": 2,
    "UNKNOWN_OUTCOME": 2,
    "EXPIRED": 2,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _strict_directory(path: Path) -> Path:
    selected = Path(path)
    if not selected.is_absolute():
        raise ValueError("Console data directory must be absolute")
    selected.mkdir(parents=True, exist_ok=True, mode=0o700)
    if selected.is_symlink() or selected.resolve(strict=True) != selected:
        raise ValueError("Console data directory must be canonical and non-symlinked")
    metadata = selected.stat(follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise ValueError("Console data directory has an unsafe owner or type")
    os.chmod(selected, 0o700)
    return selected


class ConsoleLocalStore:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = _strict_directory(data_dir)
        self.database = self.data_dir / "console.sqlite3"
        self.objects = _strict_directory(self.data_dir / "objects")
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=5.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version != LOCAL_SCHEMA_VERSION:
                raise ValueError(
                    f"Console local database schema {version} != {LOCAL_SCHEMA_VERSION}"
                )
        except BaseException:
            connection.close()
            raise
        return connection

    def _initialize(self) -> None:
        existed = self.database.exists()
        if existed and (self.database.is_symlink() or not self.database.is_file()):
            raise ValueError("Console local database must be a regular file")
        connection = sqlite3.connect(self.database, timeout=5.0)
        try:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            objects = connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','view','trigger')"
            ).fetchall()
            if version not in {0, LOCAL_SCHEMA_VERSION}:
                raise ValueError("Console local database schema is unsupported")
            if version == 0 and objects:
                raise ValueError("Console local database has unversioned objects")
            if version == 0:
                connection.executescript(
                    """
                    CREATE TABLE drafts (
                        id TEXT PRIMARY KEY,
                        task_kind TEXT NOT NULL,
                        title TEXT NOT NULL,
                        digest TEXT NOT NULL,
                        value_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE operations (
                        operation_id TEXT PRIMARY KEY,
                        kind TEXT NOT NULL,
                        operation_digest TEXT NOT NULL,
                        status TEXT NOT NULL,
                        prepared_json TEXT,
                        receipt_json TEXT,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE audit (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        operation_id TEXT,
                        payload_digest TEXT NOT NULL,
                        previous_digest TEXT,
                        entry_digest TEXT NOT NULL UNIQUE,
                        payload_json TEXT NOT NULL
                    );
                    PRAGMA user_version = 1;
                    """
                )
                connection.commit()
        finally:
            connection.close()
        os.chmod(self.database, 0o600)
        if stat.S_IMODE(self.database.stat().st_mode) != 0o600:
            raise ValueError("Console local database mode is not 0600")

    def store_candidate_bundle(self, bundle: CandidateBundle) -> str:
        bundle.validate(TRITON_PYTHON_BUNDLE_LIMITS)
        payload = bundle.bundle_bytes
        digest = hashlib.sha256(payload).hexdigest()
        directory = _strict_directory(self.objects / digest[:2])
        path = directory / digest[2:]
        if path.exists():
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise ValueError("Console local CAS collision")
            return str(bundle.artifact_id)
        descriptor, temporary = tempfile.mkstemp(prefix=".candidate-", dir=directory)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            os.chmod(path, 0o600)
            directory_descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        return str(bundle.artifact_id)

    def put_draft(self, draft: DraftV1) -> None:
        value = draft.to_dict()
        encoded = canonical_json_bytes(value).decode("utf-8")
        with closing(self._connect()) as connection, connection:
            existing = connection.execute(
                "SELECT created_at FROM drafts WHERE id = ?", (draft.draft_id,)
            ).fetchone()
            if existing is not None and str(existing[0]) != draft.created_at:
                raise ValueError("draft immutable creation identity changed")
            connection.execute(
                """
                INSERT INTO drafts (
                    id, task_kind, title, digest, value_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    task_kind=excluded.task_kind,
                    title=excluded.title,
                    digest=excluded.digest,
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (
                    draft.draft_id,
                    draft.task_kind,
                    draft.title,
                    draft.digest,
                    encoded,
                    draft.created_at,
                    draft.updated_at,
                ),
            )
        self.append_audit("DRAFT_SAVED", None, value)

    def list_drafts(self) -> list[Mapping[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT value_json FROM drafts ORDER BY updated_at DESC, id"
            ).fetchall()
        return [json.loads(str(row[0])) for row in rows]

    def record_prepared(self, prepared: PreparedOperationV1) -> None:
        encoded = canonical_json_bytes(prepared.to_dict()).decode("utf-8")
        with closing(self._connect()) as connection, connection:
            existing = connection.execute(
                "SELECT kind, operation_digest FROM operations WHERE operation_id = ?",
                (prepared.operation_id,),
            ).fetchone()
            if existing is not None and (
                str(existing["kind"]) != prepared.kind
                or str(existing["operation_digest"]) != prepared.operation_digest
            ):
                raise ValueError("local operation identity collision")
            connection.execute(
                """
                INSERT INTO operations (
                    operation_id, kind, operation_digest, status,
                    prepared_json, receipt_json, updated_at
                ) VALUES (?, ?, ?, 'PREPARED', ?, NULL, ?)
                ON CONFLICT(operation_id) DO UPDATE SET
                    prepared_json=excluded.prepared_json,
                    updated_at=excluded.updated_at
                """,
                (
                    prepared.operation_id,
                    prepared.kind,
                    prepared.operation_digest,
                    encoded,
                    _utc_now(),
                ),
            )
        self.append_audit("OPERATION_PREPARED", prepared.operation_id, prepared.to_dict())

    def record_receipt(self, receipt: OperationReceiptV1) -> None:
        encoded = canonical_json_bytes(receipt.to_dict()).decode("utf-8")
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT kind, operation_digest, status FROM operations WHERE operation_id = ?",
                (receipt.operation_id,),
            ).fetchone()
            if row is None:
                raise ValueError("receipt has no locally prepared operation")
            if (
                str(row["kind"]) != receipt.kind
                or str(row["operation_digest"]) != receipt.operation_digest
                or _STATUS_RANK[receipt.status] < _STATUS_RANK[str(row["status"])]
            ):
                raise ValueError("receipt conflicts with local operation identity/state")
            connection.execute(
                "UPDATE operations SET status=?, receipt_json=?, updated_at=? WHERE operation_id=?",
                (receipt.status, encoded, _utc_now(), receipt.operation_id),
            )
        self.append_audit("OPERATION_RECEIPT", receipt.operation_id, receipt.to_dict())

    def get_operation(self, operation_id: str) -> Mapping[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
            ).fetchone()
        if row is None:
            return None
        return {
            "operation_id": str(row["operation_id"]),
            "kind": str(row["kind"]),
            "operation_digest": str(row["operation_digest"]),
            "status": str(row["status"]),
            "prepared": (
                None if row["prepared_json"] is None else json.loads(row["prepared_json"])
            ),
            "receipt": (
                None if row["receipt_json"] is None else json.loads(row["receipt_json"])
            ),
            "updated_at": str(row["updated_at"]),
        }

    def append_audit(
        self, event_type: str, operation_id: str | None, payload: Mapping[str, Any]
    ) -> str:
        if not event_type.isascii() or not event_type.replace("_", "").isalnum():
            raise ValueError("audit event_type is invalid")
        payload_value = dict(payload)
        payload_digest = canonical_sha256(payload_value)
        created_at = _utc_now()
        with closing(self._connect()) as connection, connection:
            previous = connection.execute(
                "SELECT entry_digest FROM audit ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            previous_digest = None if previous is None else str(previous[0])
            material = {
                "schema_version": 1,
                "created_at": created_at,
                "event_type": event_type,
                "operation_id": operation_id,
                "payload_digest": payload_digest,
                "previous_digest": previous_digest,
            }
            entry_digest = canonical_sha256(material)
            connection.execute(
                """
                INSERT INTO audit (
                    created_at, event_type, operation_id, payload_digest,
                    previous_digest, entry_digest, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    event_type,
                    operation_id,
                    payload_digest,
                    previous_digest,
                    entry_digest,
                    canonical_json_bytes(payload_value).decode("utf-8"),
                ),
            )
        return entry_digest

    def audit_records(self, *, limit: int = 100) -> list[Mapping[str, Any]]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("audit limit must be between 1 and 100")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM audit ORDER BY sequence DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]


__all__ = ["ConsoleLocalStore", "LOCAL_SCHEMA_VERSION"]
