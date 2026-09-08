"""Read-only, schema-guarded views over the live scientific databases."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import stat
import subprocess
from typing import Any, Iterable, Mapping

from ..autorun.admin import AdminManifest
from ..autorun.deployment import (
    DEPLOYMENT_BASELINE_FILENAME,
    DeploymentBaselinePin,
)
from ..autorun.models import ControllerConfig
from ..campaign.paths import conventional_campaign_database
from ..platform.canonical import canonical_sha256
from ..platform.artifacts import ArtifactId
from ..platform.proposal import CandidateBundle, TRITON_PYTHON_BUNDLE_LIMITS
from ..profiler_contract import PROFILER_ACTIVATION_PROFILE_DIGEST
from ..scoring_shadow import public_scoring_shadow_summary
from ..scoring_shadow import SCORING_SHADOW_PROFILE_DIGEST
from .protocol import (
    AGENT_PROTOCOL_DIGEST,
    CompositeEventCursorV1,
    ConsoleSnapshotV1,
    RuntimeIdentityV1,
)


CONTROLLER_SCHEMA_VERSION = 3
HISTORY_SCHEMA_VERSION = 3
CAMPAIGN_SCHEMA_VERSION = 1
DEFAULT_PAGE_LIMIT = 50
MAX_PAGE_LIMIT = 100
MAX_STABILITY_RETRIES = 3


_CONTROLLER_COLUMNS: Mapping[str, frozenset[str]] = {
    "runs": frozenset(
        {
            "id",
            "status",
            "created_at",
            "updated_at",
            "deadline_epoch",
            "valid_candidates",
            "stop_requested",
            "stop_reason",
            "namespace_id",
            "resolved_config_digest",
            "workflow_snapshot_json",
            "baseline_ref_json",
            "history_cutoff_json",
        }
    ),
    "iterations": frozenset(
        {
            "id",
            "run_id",
            "iteration_index",
            "status",
            "stage",
            "candidate_hash",
            "outcome",
            "error",
            "updated_at",
        }
    ),
    "events": frozenset(
        {"id", "run_id", "iteration_id", "created_at", "event", "details_json"}
    ),
    "evaluation_attempts": frozenset(
        {
            "id",
            "experiment_uid",
            "run_id",
            "iteration_id",
            "status",
            "stage",
            "suite",
            "replicate_kind",
            "replicate_index",
            "candidate_artifact_id",
            "history_experiment_id",
            "error",
        }
    ),
}

_HISTORY_COLUMNS: Mapping[str, frozenset[str]] = {
    "experiments": frozenset(
        {
            "id",
            "created_at",
            "candidate_hash",
            "backend",
            "suite",
            "status",
            "promotable",
            "aggregate_score",
            "artifact_id",
            "namespace_id",
            "experiment_uid",
            "condition_digest",
            "replicate_kind",
            "replicate_index",
            "baseline_experiment_uid",
            "result_json",
        }
    ),
    "case_measurements": frozenset(
        {"id", "experiment_id", "case_name", "matched_ratio", "passed"}
    ),
    "experiment_relations": frozenset(
        {
            "id",
            "source_experiment_uid",
            "target_experiment_uid",
            "relation_type",
            "created_at",
        }
    ),
    "candidate_artifacts": frozenset(
        {
            "artifact_id",
            "artifact_kind",
            "content_sha256",
            "object_path",
            "byte_size",
            "manifest_json",
        }
    ),
}

_CAMPAIGN_COLUMNS: Mapping[str, frozenset[str]] = {
    "campaigns": frozenset(
        {
            "id",
            "namespace_id",
            "mode",
            "status",
            "created_at",
            "updated_at",
            "snapshot_digest",
            "active_baseline_revision_id",
            "stop_reason",
        }
    ),
    "child_runs": frozenset(
        {
            "id",
            "campaign_id",
            "child_index",
            "controller_run_id",
            "status",
            "baseline_revision_id",
            "created_at",
            "started_at",
            "finished_at",
        }
    ),
    "baseline_revisions": frozenset(
        {
            "id",
            "campaign_id",
            "revision_index",
            "namespace_id",
            "parent_revision_id",
            "artifact_id",
            "revision_kind",
            "primary_experiment_uid",
            "confirmation_experiment_uid",
            "created_at",
        }
    ),
    "budget_actions": frozenset(
        {
            "id",
            "campaign_id",
            "idempotency_key",
            "action_kind",
            "status",
            "reserved_candidates",
            "reserved_wall_ms",
            "reserved_gpu_ms",
            "reserved_tokens",
            "reserved_cost_microusd",
            "actual_candidates",
            "actual_wall_ms",
            "actual_gpu_ms",
            "actual_tokens",
            "actual_cost_microusd",
        }
    ),
    "resource_leases": frozenset(
        {
            "resource_id",
            "fencing_epoch",
            "campaign_id",
            "status",
            "acquired_at",
            "expires_epoch",
            "released_at",
            "reason",
        }
    ),
    "outbox": frozenset(
        {"id", "campaign_id", "event_type", "created_at", "delivered_at"}
    ),
    "soak_generations": frozenset(
        {
            "id",
            "gate_id",
            "stage",
            "generation_index",
            "status",
            "required_seconds",
            "started_epoch",
            "last_heartbeat_epoch",
            "accumulated_seconds",
            "invariant_snapshot_digest",
        }
    ),
    "soak_violations": frozenset(
        {
            "id",
            "gate_id",
            "stage",
            "generation_id",
            "observed_epoch",
            "reason_code",
        }
    ),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _canonical_regular_file(path: Path, field_name: str) -> Path:
    selected = Path(path)
    if not selected.is_absolute():
        raise ValueError(f"{field_name} must be absolute")
    try:
        resolved = selected.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{field_name} is unavailable") from exc
    if resolved != selected:
        raise ValueError(f"{field_name} must be canonical and non-symlinked")
    metadata = selected.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{field_name} must be a regular file")
    return selected


def _json_value(raw: object, field_name: str) -> Any:
    if not isinstance(raw, str):
        raise ValueError(f"{field_name} must be stored as JSON text")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} contains invalid JSON") from exc
    return value


def _row_value(row: sqlite3.Row) -> dict[str, Any]:
    value = dict(row)
    for field in tuple(value):
        if field.endswith("_json"):
            decoded = _json_value(value.pop(field), field)
            value[field.removesuffix("_json")] = decoded
    for field in ("promotable", "stop_requested", "delivered_at"):
        if field in value and value[field] is not None and field != "delivered_at":
            value[field] = bool(value[field])
    return value


def _page_limit(value: int) -> int:
    if type(value) is not int or not 1 <= value <= MAX_PAGE_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_PAGE_LIMIT}")
    return value


@dataclass(frozen=True)
class ConsolePaths:
    manifest_path: Path
    repository_dir: Path
    runtime_root: Path
    state_dir: Path
    controller_dir: Path
    checkpoint_dir: Path
    controller_db: Path
    history_db: Path
    campaign_db: Path
    deployment_pin: Path
    config_paths: tuple[Path, ...]
    expected_git_commit: str

    @classmethod
    def from_admin_manifest(cls, manifest_path: Path) -> "ConsolePaths":
        manifest_path = _canonical_regular_file(Path(manifest_path), "admin manifest")
        manifest = AdminManifest.load(manifest_path)
        config_paths = (
            manifest.base_config,
            manifest.pro_config,
            manifest.flash_config,
            manifest.pro_canary_config,
            manifest.flash_canary_config,
        )
        configs = tuple(ControllerConfig.load(path) for path in config_paths)
        first = configs[0]
        common_fields = (
            "repository_dir",
            "state_dir",
            "controller_dir",
            "checkpoint_dir",
            "expected_git_commit",
            "framework_git_commit",
            "expected_kernel_hash",
            "evaluator_image",
        )
        for config in configs[1:]:
            for field_name in common_fields:
                if getattr(config, field_name) != getattr(first, field_name):
                    raise ValueError(
                        f"console configs disagree on frozen field {field_name}"
                    )
        runtime_root = manifest.runtime_root
        campaign_db = conventional_campaign_database(runtime_root)
        values = cls(
            manifest_path=manifest_path,
            repository_dir=first.repository_dir,
            runtime_root=runtime_root,
            state_dir=first.state_dir,
            controller_dir=first.controller_dir,
            checkpoint_dir=first.checkpoint_dir,
            controller_db=first.controller_dir / "controller.sqlite3",
            history_db=first.state_dir / "history.sqlite3",
            campaign_db=campaign_db,
            deployment_pin=runtime_root / DEPLOYMENT_BASELINE_FILENAME,
            config_paths=tuple(config_paths),
            expected_git_commit=first.expected_git_commit,
        )
        if manifest.repository_dir != values.repository_dir:
            raise ValueError("admin manifest and configs disagree on repository_dir")
        for field_name in ("controller_db", "history_db", "campaign_db"):
            _canonical_regular_file(getattr(values, field_name), field_name)
        _canonical_regular_file(values.deployment_pin, "deployment baseline pin")
        return values


class ReadOnlyDatabase:
    def __init__(
        self,
        path: Path,
        *,
        expected_version: int,
        required_columns: Mapping[str, frozenset[str]],
    ) -> None:
        self.path = _canonical_regular_file(path, "console database")
        self.expected_version = expected_version
        self.required_columns = required_columns

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path.as_uri() + "?mode=ro",
            uri=True,
            isolation_level=None,
            timeout=0.25,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 250")
        try:
            self.validate(connection)
        except BaseException:
            connection.close()
            raise
        return connection

    def validate(self, connection: sqlite3.Connection, *, deep: bool = False) -> None:
        query_only = int(connection.execute("PRAGMA query_only").fetchone()[0])
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if query_only != 1:
            raise ValueError("console database is not query-only")
        if version != self.expected_version:
            raise ValueError(
                f"console database schema {version} != {self.expected_version}"
            )
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        missing_tables = sorted(set(self.required_columns) - tables)
        if missing_tables:
            raise ValueError(
                "console database is missing tables: " + ", ".join(missing_tables)
            )
        for table, required in self.required_columns.items():
            columns = {
                str(row[1])
                for row in connection.execute(f'PRAGMA table_info("{table}")')
            }
            missing = sorted(required - columns)
            if missing:
                raise ValueError(
                    f"console database table {table} is missing columns: "
                    + ", ".join(missing)
                )
        if deep:
            quick = str(connection.execute("PRAGMA quick_check").fetchone()[0])
            if quick != "ok":
                raise ValueError("console database quick_check failed")
            foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchone()
            if foreign_keys is not None:
                raise ValueError("console database foreign-key check failed")


class ConsoleReadModel:
    """Build bounded snapshots without owning or mutating scientific state."""

    def __init__(
        self,
        paths: ConsolePaths,
        *,
        git_binary: str = "git",
        stability_retries: int = MAX_STABILITY_RETRIES,
    ) -> None:
        if not isinstance(paths, ConsolePaths):
            raise TypeError("paths must be ConsolePaths")
        if type(stability_retries) is not int or not 1 <= stability_retries <= 5:
            raise ValueError("stability_retries must be between 1 and 5")
        self.paths = paths
        self.git_binary = git_binary
        self.stability_retries = stability_retries
        self.controller = ReadOnlyDatabase(
            paths.controller_db,
            expected_version=CONTROLLER_SCHEMA_VERSION,
            required_columns=_CONTROLLER_COLUMNS,
        )
        self.history = ReadOnlyDatabase(
            paths.history_db,
            expected_version=HISTORY_SCHEMA_VERSION,
            required_columns=_HISTORY_COLUMNS,
        )
        self.campaign = ReadOnlyDatabase(
            paths.campaign_db,
            expected_version=CAMPAIGN_SCHEMA_VERSION,
            required_columns=_CAMPAIGN_COLUMNS,
        )

    def _git_commit(self) -> str:
        process = subprocess.run(
            [
                self.git_binary,
                "-C",
                str(self.paths.repository_dir),
                "rev-parse",
                "HEAD",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10.0,
            check=False,
        )
        if process.returncode != 0 or len(process.stdout) > 128:
            raise ValueError("could not resolve console repository HEAD")
        try:
            commit = process.stdout.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("repository HEAD is not ASCII") from exc
        return commit

    def runtime_identity(self) -> RuntimeIdentityV1:
        pin = DeploymentBaselinePin.load(self.paths.deployment_pin)
        config_digest = canonical_sha256(
            {
                "schema_version": 1,
                "admin_manifest_sha256": _sha256_file(self.paths.manifest_path),
                "config_sha256": [
                    _sha256_file(path) for path in self.paths.config_paths
                ],
            }
        )
        return RuntimeIdentityV1(
            git_commit=self._git_commit(),
            expected_git_commit=self.paths.expected_git_commit,
            config_digest=config_digest,
            deployment_evidence_digest=pin.evidence_digest,
            namespace_id=pin.namespace_id,
            execution_environment_digest=pin.execution_environment.digest,
            profiler_activation_profile_digest=PROFILER_ACTIVATION_PROFILE_DIGEST,
            scoring_shadow_profile_digest=SCORING_SHADOW_PROFILE_DIGEST,
            controller_schema_version=CONTROLLER_SCHEMA_VERSION,
            history_schema_version=HISTORY_SCHEMA_VERSION,
            campaign_schema_version=CAMPAIGN_SCHEMA_VERSION,
            agent_protocol_digest=AGENT_PROTOCOL_DIGEST,
        )

    @staticmethod
    def _max(connection: sqlite3.Connection, table: str, column: str = "id") -> int:
        value = connection.execute(
            f'SELECT COALESCE(MAX("{column}"), 0) FROM "{table}"'
        ).fetchone()[0]
        return int(value)

    def _cursor(self, identity_digest: str) -> CompositeEventCursorV1:
        with closing(self.controller.connect()) as controller, closing(
            self.history.connect()
        ) as history, closing(self.campaign.connect()) as campaign:
            return CompositeEventCursorV1(
                controller_event_id=self._max(controller, "events"),
                controller_attempt_id=self._max(controller, "evaluation_attempts"),
                history_experiment_id=self._max(history, "experiments"),
                history_relation_id=self._max(history, "experiment_relations"),
                campaign_outbox_id=self._max(campaign, "outbox"),
                campaign_child_id=self._max(campaign, "child_runs"),
                campaign_lease_epoch=self._max(
                    campaign, "resource_leases", "fencing_epoch"
                ),
                soak_generation_id=self._max(campaign, "soak_generations"),
                soak_violation_id=self._max(campaign, "soak_violations"),
                runtime_identity_digest=identity_digest,
            )

    @staticmethod
    def _rows(
        connection: sqlite3.Connection,
        query: str,
        parameters: Iterable[Any] = (),
    ) -> list[dict[str, Any]]:
        return [_row_value(row) for row in connection.execute(query, tuple(parameters))]

    def _read_once(self, *, limit: int) -> dict[str, Any]:
        limit = _page_limit(limit)
        with closing(self.controller.connect()) as controller, closing(
            self.history.connect()
        ) as history, closing(self.campaign.connect()) as campaign:
            runs = self._rows(
                controller,
                """
                SELECT id, status, created_at, updated_at, deadline_epoch,
                       valid_candidates, stop_requested, stop_reason,
                       namespace_id, resolved_config_digest
                FROM runs ORDER BY updated_at DESC, id DESC LIMIT ?
                """,
                (limit,),
            )
            iterations = self._rows(
                controller,
                """
                SELECT id, run_id, iteration_index, status, stage,
                       candidate_hash, outcome, error, updated_at
                FROM iterations ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            attempts = self._rows(
                controller,
                """
                SELECT id, experiment_uid, run_id, iteration_id, status,
                       stage, suite, replicate_kind, replicate_index,
                       candidate_artifact_id, history_experiment_id, error
                FROM evaluation_attempts ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            experiments = self._rows(
                history,
                """
                SELECT id, created_at, candidate_hash, backend, suite, status,
                       promotable, aggregate_score, artifact_id, namespace_id,
                       experiment_uid, condition_digest, replicate_kind,
                       replicate_index, baseline_experiment_uid, result_json
                FROM experiments ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            for experiment in experiments:
                result = experiment.pop("result", None)
                scoring = public_scoring_shadow_summary(
                    result.get("objective_scoring")
                    if isinstance(result, Mapping)
                    else None
                )
                experiment.update(
                    {
                        "xpuoj_proxy_status": scoring["status"],
                        "xpuoj_proxy_score": scoring["objective_score"],
                        "xpuoj_proxy_reason": scoring["reason"],
                        "xpuoj_proxy_paired_speedup": scoring[
                            "paired_aggregate_speedup"
                        ],
                        "xpuoj_proxy_worst_case_regression": scoring[
                            "paired_worst_case_regression"
                        ],
                        "xpuoj_proxy_promotion_authority": scoring[
                            "promotion_authority"
                        ],
                        "xpuoj_proxy_profile_digest": scoring[
                            "profile_digest"
                        ],
                    }
                )
            relations = self._rows(
                history,
                """
                SELECT id, source_experiment_uid, target_experiment_uid,
                       relation_type, created_at
                FROM experiment_relations ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            campaigns = self._rows(
                campaign,
                """
                SELECT id, namespace_id, mode, status, created_at, updated_at,
                       snapshot_digest, active_baseline_revision_id, stop_reason
                FROM campaigns ORDER BY updated_at DESC, id DESC LIMIT ?
                """,
                (limit,),
            )
            children = self._rows(
                campaign,
                """
                SELECT id, campaign_id, child_index, controller_run_id, status,
                       baseline_revision_id, created_at, started_at, finished_at
                FROM child_runs ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            leases = self._rows(
                campaign,
                """
                SELECT resource_id, fencing_epoch, campaign_id, status,
                       acquired_at, expires_epoch, released_at, reason
                FROM resource_leases ORDER BY fencing_epoch DESC LIMIT ?
                """,
                (limit,),
            )
            budgets = self._rows(
                campaign,
                """
                SELECT id, campaign_id, idempotency_key, action_kind, status,
                       reserved_candidates, reserved_wall_ms, reserved_gpu_ms,
                       reserved_tokens, reserved_cost_microusd,
                       actual_candidates, actual_wall_ms, actual_gpu_ms,
                       actual_tokens, actual_cost_microusd
                FROM budget_actions ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            soak = self._rows(
                campaign,
                """
                SELECT id, gate_id, stage, generation_index, status,
                       required_seconds, started_epoch, last_heartbeat_epoch,
                       accumulated_seconds, invariant_snapshot_digest
                FROM soak_generations ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
            violations = self._rows(
                campaign,
                """
                SELECT id, gate_id, stage, generation_id, observed_epoch,
                       reason_code
                FROM soak_violations ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
        return {
            "runs": runs,
            "iterations": iterations,
            "evaluation_attempts": attempts,
            "experiments": experiments,
            "experiment_relations": relations,
            "campaigns": campaigns,
            "child_runs": children,
            "resource_leases": leases,
            "budget_actions": budgets,
            "soak_generations": soak,
            "soak_violations": violations,
        }

    def health(self, *, deep: bool = False) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for name, database in (
            ("controller", self.controller),
            ("history", self.history),
            ("campaign", self.campaign),
        ):
            with closing(database.connect()) as connection:
                database.validate(connection, deep=deep)
            results[name] = {
                "status": "AVAILABLE",
                "schema_version": database.expected_version,
            }
        return {"schema_version": 1, "status": "READY", "sources": results}

    def scientific_artifact(self, artifact_id: str) -> dict[str, Any]:
        """Return one verified scientific bundle and its entrypoint source."""

        parsed = ArtifactId.parse(artifact_id)
        if parsed.is_legacy_source:
            raise ValueError("Console serves only authoritative V2 bundle artifacts")
        with closing(self.history.connect()) as history:
            row = history.execute(
                """
                SELECT * FROM candidate_artifacts WHERE artifact_id = ?
                """,
                (str(parsed),),
            ).fetchone()
            referenced = history.execute(
                """
                SELECT COUNT(*) FROM experiments
                WHERE artifact_id = ? AND status = 'SUCCESS'
                """,
                (str(parsed),),
            ).fetchone()[0]
        if row is None or int(referenced) <= 0:
            raise ValueError("artifact is not referenced by successful scientific evidence")
        record = _row_value(row)
        if record.get("artifact_kind") != "source_bundle_v1":
            raise ValueError("artifact is not a source bundle")
        relative = PurePosixPath(str(record.get("object_path", "")))
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError("artifact CAS path is invalid")
        object_path = self.paths.state_dir.joinpath(*relative.parts)
        try:
            resolved = object_path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError("artifact CAS object is unavailable") from exc
        if (
            self.paths.state_dir not in resolved.parents
            or resolved != object_path
            or object_path.is_symlink()
            or not object_path.is_file()
        ):
            raise ValueError("artifact CAS object escapes its trusted root")
        payload = object_path.read_bytes()
        expected_size = record.get("byte_size")
        expected_hash = record.get("content_sha256")
        if (
            type(expected_size) is not int
            or expected_size != len(payload)
            or expected_hash != hashlib.sha256(payload).hexdigest()
        ):
            raise ValueError("artifact CAS object differs from its metadata")
        try:
            bundle = CandidateBundle.from_value(
                json.loads(payload.decode("utf-8")),
                limits=TRITON_PYTHON_BUNDLE_LIMITS,
            )
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError("artifact bundle payload is invalid") from exc
        if (
            bundle.bundle_bytes != payload
            or str(bundle.artifact_id) != str(parsed)
            or record.get("manifest") != bundle.manifest
        ):
            raise ValueError("artifact bundle identity or manifest differs")
        entrypoint = next(item for item in bundle.files if item.path == bundle.entrypoint)
        return {
            "schema_version": 1,
            "artifact_id": str(parsed),
            "manifest": bundle.manifest,
            "entrypoint": bundle.entrypoint,
            "media_type": entrypoint.media_type,
            "source": entrypoint.content,
        }

    def snapshot(self, *, limit: int = DEFAULT_PAGE_LIMIT) -> ConsoleSnapshotV1:
        identity = self.runtime_identity()
        if identity.git_commit != identity.expected_git_commit:
            raise ValueError("repository HEAD differs from the frozen config commit")
        last_data: dict[str, Any] | None = None
        before: CompositeEventCursorV1 | None = None
        after: CompositeEventCursorV1 | None = None
        for _attempt in range(self.stability_retries):
            before = self._cursor(identity.digest)
            last_data = self._read_once(limit=limit)
            after = self._cursor(identity.digest)
            if before == after:
                source_digests = {
                    "controller": canonical_sha256(
                        {
                            "event": after.controller_event_id,
                            "attempt": after.controller_attempt_id,
                        }
                    ),
                    "history": canonical_sha256(
                        {
                            "experiment": after.history_experiment_id,
                            "relation": after.history_relation_id,
                        }
                    ),
                    "campaign": canonical_sha256(
                        {
                            "outbox": after.campaign_outbox_id,
                            "child": after.campaign_child_id,
                            "lease": after.campaign_lease_epoch,
                            "soak_generation": after.soak_generation_id,
                            "soak_violation": after.soak_violation_id,
                        }
                    ),
                }
                return ConsoleSnapshotV1(
                    status="STABLE",
                    identity=identity,
                    cursor=after,
                    observed_at=_utc_now(),
                    source_digests=source_digests,
                    data=last_data,
                )
        assert last_data is not None and after is not None and before is not None
        return ConsoleSnapshotV1(
            status="CHANGING",
            identity=identity,
            cursor=after,
            observed_at=_utc_now(),
            source_digests={},
            data=last_data,
        )


__all__ = [
    "CAMPAIGN_SCHEMA_VERSION",
    "CONTROLLER_SCHEMA_VERSION",
    "HISTORY_SCHEMA_VERSION",
    "ConsolePaths",
    "ConsoleReadModel",
    "ReadOnlyDatabase",
]
