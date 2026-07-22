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

from .contract import EVALUATION_STATUSES


SCHEMA_VERSION = 1


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


def _sample_tuple(values: Sequence[float] | None) -> tuple[float, ...]:
    if values is None:
        return ()
    return tuple(float(value) for value in values)


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
            "result": dict(self.result),
            "case_measurements": [case.to_dict() for case in self.case_measurements],
        }


class HistoryStore:
    """SQLite-backed history with content-addressed candidate artifacts."""

    def __init__(
        self,
        db_path: str | os.PathLike[str],
        state_dir: str | os.PathLike[str] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.state_dir = Path(state_dir) if state_dir is not None else self.db_path.parent
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.state_dir / "artifacts"
        self._connection = sqlite3.connect(self.db_path, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute("PRAGMA synchronous = NORMAL")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        current_version = int(
            self._connection.execute("PRAGMA user_version").fetchone()[0]
        )
        if current_version > SCHEMA_VERSION:
            self._connection.close()
            raise RuntimeError(
                f"history schema {current_version} is newer than supported "
                f"version {SCHEMA_VERSION}"
            )
        if current_version == SCHEMA_VERSION:
            return
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
                    ON experiments(backend, suite, status, promotable, aggregate_score);

                CREATE TABLE IF NOT EXISTS case_measurements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id INTEGER NOT NULL
                        REFERENCES experiments(id) ON DELETE CASCADE,
                    case_name TEXT NOT NULL,
                    matched_ratio REAL,
                    passed INTEGER CHECK (passed IS NULL OR passed IN (0, 1)),
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

    def _store_artifact(self, candidate_hash: str, source: str) -> tuple[str, bool]:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact = self.artifacts_dir / f"{candidate_hash}.py"
        relative_path = artifact.relative_to(self.state_dir).as_posix()
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
    ) -> ExperimentRecord:
        """Atomically append an experiment and all of its case measurements."""

        if not isinstance(candidate_source, str):
            raise TypeError("candidate_source must be a string")
        digest = hashlib.sha256(candidate_source.encode("utf-8")).hexdigest()
        if candidate_hash is not None and candidate_hash != digest:
            raise ValueError("candidate_hash does not match candidate_source")
        candidate_hash = digest
        if not isinstance(status, str) or not status:
            raise ValueError("status must be a non-empty string")
        if not isinstance(backend, str) or not backend:
            raise ValueError("backend must be a non-empty string")
        if not isinstance(suite, str) or not suite:
            raise ValueError("suite must be a non-empty string")

        cases = tuple(CaseMeasurement.from_value(case) for case in case_measurements)
        if len({case.name for case in cases}) != len(cases):
            raise ValueError("case measurement names must be unique within an experiment")
        environment_object = _json_object(environment)
        result_object = _json_object(result)
        created_at = created_at or _utc_now()
        artifact_created = False
        artifact_path: str | None = None

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            duplicate_row = self._connection.execute(
                "SELECT id FROM experiments WHERE candidate_hash = ? ORDER BY id LIMIT 1",
                (candidate_hash,),
            ).fetchone()
            artifact_path, artifact_created = self._store_artifact(
                candidate_hash, candidate_source
            )
            cursor = self._connection.execute(
                """
                INSERT INTO experiments (
                    schema_version, created_at, candidate_hash, git_commit,
                    backend, suite, status, promotable, duplicate_of_id,
                    aggregate_score, note, environment_json, error_summary,
                    artifact_path, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            self._connection.execute("COMMIT")
        except Exception:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            if artifact_created and artifact_path is not None:
                try:
                    (self.state_dir / artifact_path).unlink()
                except FileNotFoundError:
                    pass
            raise

        record = self.get_experiment(experiment_id)
        if record is None:  # pragma: no cover - guards an impossible committed state
            raise RuntimeError("recorded experiment could not be reloaded")
        return record

    def record(self, **values: Any) -> ExperimentRecord:
        """Short alias for ``record_experiment``."""

        return self.record_experiment(**values)

    def _record_from_row(self, row: sqlite3.Row) -> ExperimentRecord:
        case_rows = self._connection.execute(
            """
            SELECT case_name, matched_ratio, passed, raw_samples_json,
                   baseline_samples_json, metrics_json
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
            case_measurements=cases,
        )

    def get_experiment(self, experiment_id: int) -> ExperimentRecord | None:
        row = self._connection.execute(
            "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
        ).fetchone()
        return None if row is None else self._record_from_row(row)

    def list_experiments(
        self,
        *,
        candidate_hash: str | None = None,
        status: str | None = None,
        backend: str | None = None,
        suite: str | None = None,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[ExperimentRecord]:
        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in (
            ("candidate_hash", candidate_hash),
            ("status", status),
            ("backend", backend),
            ("suite", suite),
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
        return [self._record_from_row(row) for row in rows]

    def find_by_candidate_hash(self, candidate_hash: str) -> list[ExperimentRecord]:
        return self.list_experiments(candidate_hash=candidate_hash)

    def has_candidate(self, candidate_hash: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM experiments WHERE candidate_hash = ? LIMIT 1",
            (candidate_hash,),
        ).fetchone()
        return row is not None

    def candidate_seen_count(self, candidate_hash: str) -> int:
        return int(
            self._connection.execute(
                "SELECT COUNT(*) FROM experiments WHERE candidate_hash = ?",
                (candidate_hash,),
            ).fetchone()[0]
        )

    def get_best(
        self,
        *,
        backend: str | None = None,
        suite: str | None = None,
    ) -> ExperimentRecord | None:
        clauses = ["status = 'SUCCESS'", "promotable = 1", "aggregate_score IS NOT NULL"]
        parameters: list[Any] = []
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
        return None if row is None else self._record_from_row(row)

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
    "CaseMeasurement",
    "EVALUATION_STATUSES",
    "ExperimentRecord",
    "HistoryStore",
    "SCHEMA_VERSION",
]
