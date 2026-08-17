"""Trusted, fail-closed observations for long-running Campaign soak gates.

The command-line interface never supplies counters.  This module derives them
from the Campaign and Controller ledgers plus a fixed-argv Docker inventory,
then signs the complete observation by content address.  It deliberately uses
read-only SQLite connections so qualification cannot mutate the evidence it is
measuring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..autorun.models import ControllerConfig
from ..autorun.runtime import CommandResult, CommandRunner
from ..platform.canonical import (
    canonical_json_text,
    canonical_sha256,
    require_sha256_digest,
)
from ..platform.profiles import BUILTIN_PROFILE_REGISTRY
from ..profiler_contract import (
    PROFILER_ACTIVATION_PROFILE_DIGEST,
    PROFILER_BUILD_PROFILE_DIGEST,
    profiler_activation_profile_snapshot,
    profiler_build_profile_snapshot,
)
from .paths import validate_production_campaign_database


COLLECTOR_REVISION = "trusted-soak-observation-v1"
OBSERVATION_SCHEMA_VERSION = 1
CAMPAIGN_SCHEMA_VERSION = 1
CONTROLLER_SCHEMA_VERSION = 3
DOCKER_INVENTORY_TIMEOUT_SECONDS = 10.0
DOCKER_INVENTORY_MAX_OUTPUT_BYTES = 64 * 1024
GIT_PROBE_TIMEOUT_SECONDS = 10.0
GIT_PROBE_MAX_OUTPUT_BYTES = 64 * 1024

COUNT_FIELDS = (
    "lease_overlap_count",
    "orphan_container_count",
    "hard_fault_retry_count",
    "budget_leak_count",
)
_TRUSTED_OBSERVATION_SEAL = object()


class _Runner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None,
        timeout_sec: float,
        max_output_bytes: int,
        container_name: str | None = None,
        docker_binary: Path | None = None,
    ) -> CommandResult: ...


def _finite_epoch(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return result


def _plain_json(value: object, field: str) -> Any:
    try:
        return json.loads(canonical_json_text(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must use the strict JSON data model") from exc


def _strict_count(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class SoakObservation:
    """One immutable, content-addressed observation from trusted sources."""

    collector_revision: str
    status: str
    invariant_snapshot: Mapping[str, Any]
    invariant_snapshot_digest: str
    interval_start_epoch: float
    interval_end_epoch: float
    counts: Mapping[str, int]
    sources: Mapping[str, Any]
    unavailable_reasons: tuple[str, ...]
    evidence_id: str
    _trusted_seal: object = field(repr=False, compare=False)
    schema_version: int = OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self._trusted_seal is not _TRUSTED_OBSERVATION_SEAL:
            raise TypeError(
                "SoakObservation values may only be minted by the trusted collector"
            )
        if self.schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError("soak observation schema_version must be 1")
        if self.collector_revision != COLLECTOR_REVISION:
            raise ValueError("unknown soak collector revision")
        if self.status not in {"AVAILABLE", "UNAVAILABLE"}:
            raise ValueError("soak observation status is invalid")
        start = _finite_epoch(self.interval_start_epoch, "interval_start_epoch")
        end = _finite_epoch(self.interval_end_epoch, "interval_end_epoch")
        if end < start:
            raise ValueError("soak observation interval moved backwards")
        snapshot = _plain_json(self.invariant_snapshot, "invariant_snapshot")
        sources = _plain_json(self.sources, "sources")
        if not isinstance(snapshot, dict) or not isinstance(sources, dict):
            raise ValueError("soak snapshot and sources must be JSON objects")
        if set(self.counts) != set(COUNT_FIELDS):
            raise ValueError("soak observation counts have an invalid schema")
        counts = {
            field: _strict_count(self.counts[field], field)
            for field in COUNT_FIELDS
        }
        reasons = tuple(self.unavailable_reasons)
        if any(not isinstance(reason, str) or not reason for reason in reasons):
            raise ValueError("unavailable reasons must be non-empty strings")
        if len(set(reasons)) != len(reasons):
            raise ValueError("unavailable reasons must be unique")
        if (self.status == "AVAILABLE") != (not reasons):
            raise ValueError("observation availability and reasons disagree")
        expected_snapshot_digest = canonical_sha256(snapshot)
        require_sha256_digest(
            self.invariant_snapshot_digest,
            field="invariant_snapshot_digest",
        )
        if self.invariant_snapshot_digest != expected_snapshot_digest:
            raise ValueError("invariant snapshot digest does not match snapshot")
        material = self._material(
            snapshot=snapshot,
            counts=counts,
            sources=sources,
            reasons=reasons,
            start=start,
            end=end,
        )
        require_sha256_digest(self.evidence_id, field="evidence_id")
        if self.evidence_id != canonical_sha256(material):
            raise ValueError("soak evidence_id does not match observation")
        object.__setattr__(self, "invariant_snapshot", snapshot)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "unavailable_reasons", reasons)
        object.__setattr__(self, "interval_start_epoch", start)
        object.__setattr__(self, "interval_end_epoch", end)

    def _material(
        self,
        *,
        snapshot: Mapping[str, Any],
        counts: Mapping[str, int],
        sources: Mapping[str, Any],
        reasons: tuple[str, ...],
        start: float,
        end: float,
    ) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "collector_revision": self.collector_revision,
            "status": self.status,
            "invariant_snapshot": dict(snapshot),
            "invariant_snapshot_digest": self.invariant_snapshot_digest,
            "interval_start_epoch": start,
            "interval_end_epoch": end,
            "counts": dict(counts),
            "sources": dict(sources),
            "unavailable_reasons": list(reasons),
        }

    @classmethod
    def _create_trusted(
        cls,
        *,
        status: str,
        invariant_snapshot: Mapping[str, Any],
        interval_start_epoch: float,
        interval_end_epoch: float,
        counts: Mapping[str, int],
        sources: Mapping[str, Any],
        unavailable_reasons: Sequence[str] = (),
    ) -> "SoakObservation":
        snapshot = _plain_json(invariant_snapshot, "invariant_snapshot")
        normalized_counts = {
            field: _strict_count(counts[field], field) for field in COUNT_FIELDS
        }
        normalized_sources = _plain_json(sources, "sources")
        reasons = tuple(sorted(set(unavailable_reasons)))
        digest = canonical_sha256(snapshot)
        start = _finite_epoch(interval_start_epoch, "interval_start_epoch")
        end = _finite_epoch(interval_end_epoch, "interval_end_epoch")
        material = {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "collector_revision": COLLECTOR_REVISION,
            "status": status,
            "invariant_snapshot": snapshot,
            "invariant_snapshot_digest": digest,
            "interval_start_epoch": start,
            "interval_end_epoch": end,
            "counts": normalized_counts,
            "sources": normalized_sources,
            "unavailable_reasons": list(reasons),
        }
        return cls(
            collector_revision=COLLECTOR_REVISION,
            status=status,
            invariant_snapshot=snapshot,
            invariant_snapshot_digest=digest,
            interval_start_epoch=start,
            interval_end_epoch=end,
            counts=normalized_counts,
            sources=normalized_sources,
            unavailable_reasons=reasons,
            evidence_id=canonical_sha256(material),
            _trusted_seal=_TRUSTED_OBSERVATION_SEAL,
        )

    @classmethod
    def unsafe_create_for_tests(
        cls,
        *,
        status: str,
        invariant_snapshot: Mapping[str, Any],
        interval_start_epoch: float,
        interval_end_epoch: float,
        counts: Mapping[str, int],
        sources: Mapping[str, Any],
        unavailable_reasons: Sequence[str] = (),
    ) -> "SoakObservation":
        """Mint injected test evidence; no production CLI reaches this hook."""

        return cls._create_trusted(
            status=status,
            invariant_snapshot=invariant_snapshot,
            interval_start_epoch=interval_start_epoch,
            interval_end_epoch=interval_end_epoch,
            counts=counts,
            sources=sources,
            unavailable_reasons=unavailable_reasons,
        )

    def _validated_copy(self) -> "SoakObservation":
        return SoakObservation(
            collector_revision=self.collector_revision,
            status=self.status,
            invariant_snapshot=self.invariant_snapshot,
            invariant_snapshot_digest=self.invariant_snapshot_digest,
            interval_start_epoch=self.interval_start_epoch,
            interval_end_epoch=self.interval_end_epoch,
            counts=self.counts,
            sources=self.sources,
            unavailable_reasons=self.unavailable_reasons,
            evidence_id=self.evidence_id,
            _trusted_seal=_TRUSTED_OBSERVATION_SEAL,
            schema_version=self.schema_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self._material(
            snapshot=self.invariant_snapshot,
            counts=self.counts,
            sources=self.sources,
            reasons=self.unavailable_reasons,
            start=self.interval_start_epoch,
            end=self.interval_end_epoch,
        ), "evidence_id": self.evidence_id}


def _read_only_connection(path: Path) -> sqlite3.Connection:
    if path.is_symlink() or not path.is_file():
        raise ValueError("database is not a regular non-symlink file")
    connection = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro", uri=True, isolation_level=None
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    return connection


def _require_schema(
    connection: sqlite3.Connection,
    *,
    version: int,
    tables: Mapping[str, frozenset[str]],
) -> None:
    quick_check = connection.execute("PRAGMA quick_check").fetchone()
    if quick_check is None or quick_check[0] != "ok":
        raise ValueError("database integrity is unavailable")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise ValueError("database foreign-key integrity is unavailable")
    actual_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if actual_version != version:
        raise ValueError("database schema version is unavailable")
    for table, required_columns in tables.items():
        columns = {
            str(row[1])
            for row in connection.execute(f"PRAGMA table_info({table})")
        }
        if not required_columns.issubset(columns):
            raise ValueError("database schema fields are unavailable")


def _rows(connection: sqlite3.Connection, query: str) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query)]


def _parse_json_fields(rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    for row in rows:
        for field in fields:
            value = row.get(field)
            if not isinstance(value, str):
                raise ValueError("database JSON field is unavailable")
            decoded = json.loads(value)
            canonical_json_text(decoded)


def _timestamp(value: object) -> float:
    if not isinstance(value, str) or not value:
        raise ValueError("database timestamp is unavailable")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("database timestamp has no timezone")
    result = parsed.astimezone(timezone.utc).timestamp()
    if not math.isfinite(result) or result < 0:
        raise ValueError("database timestamp is unavailable")
    return result


def _source_record(
    *, name: str, rows: Mapping[str, Any], cursor: Mapping[str, Any], summary: Mapping[str, Any]
) -> dict[str, Any]:
    material = {"source": name, "rows": rows}
    return {
        "status": "AVAILABLE",
        "cursor": _plain_json(cursor, "source cursor"),
        "digest": canonical_sha256(material),
        "summary": _plain_json(summary, "source summary"),
    }


def _unavailable_source(name: str, reason: str) -> dict[str, Any]:
    return {
        "status": "UNAVAILABLE",
        "cursor": {},
        "digest": canonical_sha256(
            {"source": name, "status": "UNAVAILABLE", "reason": reason}
        ),
        "summary": {"reason": reason},
    }


class SoakObservationCollector:
    """Collect one bounded, trusted observation for a conventional runtime."""

    def __init__(
        self,
        config: ControllerConfig,
        *,
        campaign_database: Path,
        runner: _Runner | None = None,
        invariant_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        if not isinstance(config, ControllerConfig):
            raise TypeError("config must be ControllerConfig")
        self.config = config
        self.runtime_root = self._runtime_root(config)
        selected_campaign_database = validate_production_campaign_database(
            campaign_database,
            runtime_root=self.runtime_root,
        )
        if (
            selected_campaign_database.is_symlink()
            or not selected_campaign_database.is_file()
        ):
            raise ValueError(
                "soak collector requires a regular non-symlink Campaign DB"
            )
        self.campaign_database = selected_campaign_database
        self.controller_database = (
            config.controller_dir / "controller.sqlite3"
        ).resolve(strict=False)
        self._runner = runner or CommandRunner()
        self._invariant_provider = invariant_provider

    @staticmethod
    def _runtime_root(config: ControllerConfig) -> Path:
        try:
            return Path(
                os.path.commonpath(
                    [
                        str(config.state_dir.resolve()),
                        str(config.controller_dir.resolve()),
                        str(config.checkpoint_dir.resolve()),
                    ]
                )
            )
        except (OSError, ValueError) as exc:
            raise ValueError("controller runtime root cannot be resolved") from exc

    def _run_fixed(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        limit: int,
    ) -> CommandResult:
        result = self._runner.run(
            tuple(argv),
            input_text=None,
            timeout_sec=timeout,
            max_output_bytes=limit,
        )
        if tuple(result.argv) != tuple(argv):
            raise ValueError("command runner did not preserve fixed argv")
        if result.timed_out or result.output_limited or result.returncode != 0:
            raise ValueError("bounded command result is unavailable")
        return result

    def _invariant_snapshot(self) -> tuple[dict[str, Any], list[str]]:
        config_value = self.config.redacted_dict()
        profiles = [
            definition.to_dict()
            for definition in BUILTIN_PROFILE_REGISTRY.definitions()
        ]
        base: dict[str, Any] = {
            "schema_version": 1,
            "collector_revision": COLLECTOR_REVISION,
            "runtime": {
                "root": str(self.runtime_root),
                "campaign_database": str(self.campaign_database),
                "controller_database": str(self.controller_database),
            },
            "config_digest": canonical_sha256(config_value),
            "profile_registry_digest": canonical_sha256(profiles),
            "expected_git_commit": self.config.expected_git_commit,
            "proposer_image": self.config.proposer_image,
            "evaluator_image": self.config.evaluator_image,
            "profiler": {
                "build_profile_digest": PROFILER_BUILD_PROFILE_DIGEST,
                "activation_profile_digest": (
                    PROFILER_ACTIVATION_PROFILE_DIGEST
                ),
                "build_profile": profiler_build_profile_snapshot(),
                "activation_profile": profiler_activation_profile_snapshot(),
            },
        }
        reasons: list[str] = []
        try:
            if self._invariant_provider is not None:
                probe = _plain_json(
                    self._invariant_provider(), "trusted invariant probe"
                )
                if not isinstance(probe, dict):
                    raise ValueError("trusted invariant probe must be an object")
            else:
                head = self._run_fixed(
                    (
                        "git",
                        "-c",
                        "core.fsmonitor=false",
                        "-C",
                        str(self.config.repository_dir),
                        "rev-parse",
                        "HEAD",
                    ),
                    timeout=GIT_PROBE_TIMEOUT_SECONDS,
                    limit=GIT_PROBE_MAX_OUTPUT_BYTES,
                ).stdout.strip()
                dirty = self._run_fixed(
                    (
                        "git",
                        "-c",
                        "core.fsmonitor=false",
                        "-C",
                        str(self.config.repository_dir),
                        "status",
                        "--porcelain",
                        "--untracked-files=all",
                    ),
                    timeout=GIT_PROBE_TIMEOUT_SECONDS,
                    limit=GIT_PROBE_MAX_OUTPUT_BYTES,
                ).stdout
                if head != self.config.expected_git_commit or dirty:
                    raise ValueError("Git identity does not match the frozen config")
                probe = {
                    "git_commit": head,
                    "worktree_clean_including_untracked": True,
                }
            base["code"] = probe
        except (AttributeError, OSError, TypeError, ValueError):
            base["code"] = {"status": "UNAVAILABLE"}
            reasons.append("INVARIANT_PROBE_UNAVAILABLE")
        return _plain_json(base, "invariant snapshot"), reasons

    def _campaign_source(
        self, *, interval_start: float, interval_end: float
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        connection = _read_only_connection(self.campaign_database)
        try:
            connection.execute("BEGIN")
            _require_schema(
                connection,
                version=CAMPAIGN_SCHEMA_VERSION,
                tables={
                    "campaigns": frozenset({"id", "snapshot_json"}),
                    "child_runs": frozenset(
                        {"id", "controller_run_id", "status", "result_json"}
                    ),
                    "budget_actions": frozenset(
                        {"id", "idempotency_key", "action_kind", "status"}
                    ),
                    "resource_leases": frozenset(
                        {
                            "resource_id",
                            "fencing_epoch",
                            "status",
                            "acquired_at",
                            "expires_epoch",
                            "released_at",
                        }
                    ),
                    "baseline_revisions": frozenset(
                        {
                            "id",
                            "campaign_id",
                            "revision_index",
                            "revision_kind",
                            "primary_experiment_uid",
                            "confirmation_experiment_uid",
                            "policy_snapshot_json",
                            "baseline_ref_json",
                        }
                    ),
                },
            )
            campaigns = _rows(
                connection, "SELECT * FROM campaigns ORDER BY created_at, id"
            )
            children = _rows(
                connection, "SELECT * FROM child_runs ORDER BY id"
            )
            budgets = _rows(
                connection, "SELECT * FROM budget_actions ORDER BY id"
            )
            leases = _rows(
                connection,
                "SELECT * FROM resource_leases ORDER BY resource_id, fencing_epoch",
            )
            revisions = _rows(
                connection,
                "SELECT * FROM baseline_revisions "
                "ORDER BY campaign_id, revision_index, id",
            )
            connection.execute("COMMIT")
        finally:
            connection.close()
        _parse_json_fields(campaigns, ("snapshot_json",))
        _parse_json_fields(children, ("proposer_profile_json", "result_json"))
        _parse_json_fields(
            revisions, ("policy_snapshot_json", "baseline_ref_json")
        )
        valid_child_statuses = {
            "PENDING", "RUNNING", "PROMOTED", "BUDGET_EXHAUSTED", "STOPPED",
            "FAILED", "HARD_FAILED", "UNKNOWN_GPU_OUTCOME",
        }
        valid_campaign_statuses = {
            "CREATED", "RUNNING", "PAUSED_OPERATOR", "PAUSED_BUDGET",
            "PAUSED_UNKNOWN_OUTCOME", "PAUSED_HARD_FAILURE",
            "PAUSED_DATA_INTEGRITY", "COMPLETED", "CANCELLED",
        }
        for campaign in campaigns:
            if (
                not isinstance(campaign.get("id"), str)
                or not campaign["id"]
                or campaign.get("status") not in valid_campaign_statuses
            ):
                raise ValueError("campaign identity fields are unavailable")
        for child in children:
            if (
                type(child.get("id")) is not int
                or child["id"] <= 0
                or child.get("status") not in valid_child_statuses
                or (
                    child.get("controller_run_id") is not None
                    and (
                        not isinstance(child["controller_run_id"], str)
                        or not child["controller_run_id"]
                    )
                )
            ):
                raise ValueError("child run status is unavailable")
        for budget in budgets:
            if (
                type(budget.get("id")) is not int
                or budget["id"] <= 0
                or not isinstance(budget.get("idempotency_key"), str)
                or not budget["idempotency_key"]
                or not isinstance(budget.get("action_kind"), str)
                or not budget["action_kind"]
            ):
                raise ValueError("budget identity fields are unavailable")
        for lease in leases:
            if (
                type(lease.get("fencing_epoch")) is not int
                or lease["fencing_epoch"] <= 0
            ):
                raise ValueError("lease fencing fields are unavailable")
        overlap_count = self._lease_overlap_count(
            leases, interval_start=interval_start, interval_end=interval_end
        )
        child_by_key = {
            "child-run:" + hashlib.sha256(
                str(child["controller_run_id"]).encode("utf-8")
            ).hexdigest(): child
            for child in children
            if isinstance(child.get("controller_run_id"), str)
            and child["controller_run_id"]
        }
        budget_leaks = 0
        for budget in budgets:
            status = budget.get("status")
            if status not in {"RESERVED", "SETTLED", "CANCELLED"}:
                raise ValueError("budget status is unavailable")
            if status != "RESERVED":
                continue
            if budget.get("action_kind") != "CHILD_RUN":
                budget_leaks += 1
                continue
            child = child_by_key.get(budget.get("idempotency_key"))
            if child is None or child["status"] not in {"PENDING", "RUNNING"}:
                budget_leaks += 1
        bad_runs = sorted(
            {
                str(child["controller_run_id"])
                for child in children
                if child["status"] in {"HARD_FAILED", "UNKNOWN_GPU_OUTCOME"}
                and isinstance(child.get("controller_run_id"), str)
                and child["controller_run_id"]
            }
        )
        terminal_children = sorted(
            (
                {
                    "child_id": int(child["id"]),
                    "controller_run_id": str(child["controller_run_id"]),
                    "status": str(child["status"]),
                }
                for child in children
                if child["status"] not in {"PENDING", "RUNNING"}
                and isinstance(child.get("controller_run_id"), str)
                and child["controller_run_id"]
            ),
            key=lambda item: (item["child_id"], item["controller_run_id"]),
        )
        staged_revisions: list[dict[str, Any]] = []
        for revision in revisions:
            if revision.get("revision_kind") not in {"DEPLOYMENT_SEED", "STAGED"}:
                raise ValueError("baseline revision kind is unavailable")
            if revision["revision_kind"] != "STAGED":
                continue
            primary = revision.get("primary_experiment_uid")
            confirmation = revision.get("confirmation_experiment_uid")
            if (
                not isinstance(primary, str)
                or not primary
                or not isinstance(confirmation, str)
                or not confirmation
            ):
                raise ValueError("staged revision evidence is unavailable")
            staged_revisions.append(
                {
                    "revision_id": str(revision["id"]),
                    "campaign_id": str(revision["campaign_id"]),
                    "revision_index": int(revision["revision_index"]),
                    "primary_experiment_uid": primary,
                    "confirmation_experiment_uid": confirmation,
                }
            )
        serialized = {
            "campaigns": campaigns,
            "children": children,
            "budgets": budgets,
            "leases": leases,
            "baseline_revisions": revisions,
        }
        cursor = {
            "campaign_count": len(campaigns),
            "child_max_id": max((int(row["id"]) for row in children), default=0),
            "budget_max_id": max((int(row["id"]) for row in budgets), default=0),
            "lease_max_fencing_epoch": max(
                (int(row["fencing_epoch"]) for row in leases), default=0
            ),
            "baseline_revision_max_index": max(
                (int(row["revision_index"]) for row in revisions), default=-1
            ),
        }
        source = _source_record(
            name="campaign.sqlite3",
            rows=serialized,
            cursor=cursor,
            summary={
                "campaign_count": len(campaigns),
                "child_count": len(children),
                "budget_action_count": len(budgets),
                "lease_count": len(leases),
                "baseline_revision_count": len(revisions),
                "activity": {
                    "terminal_children": terminal_children,
                    "staged_revisions": staged_revisions,
                },
            },
        )
        return source, {
            "lease_overlap_count": overlap_count,
            "budget_leak_count": budget_leaks,
            "bad_controller_runs": bad_runs,
        }

    @staticmethod
    def _lease_overlap_count(
        leases: Sequence[Mapping[str, Any]], *, interval_start: float, interval_end: float
    ) -> int:
        by_resource: dict[str, list[tuple[float, float]]] = {}
        for lease in leases:
            resource = lease.get("resource_id")
            status = lease.get("status")
            if not isinstance(resource, str) or not resource:
                raise ValueError("lease resource is unavailable")
            if status not in {"ACTIVE", "RELEASED", "EXPIRED", "QUARANTINED"}:
                raise ValueError("lease status is unavailable")
            start = _timestamp(lease.get("acquired_at"))
            expiry = _finite_epoch(lease.get("expires_epoch"), "expires_epoch")
            released = lease.get("released_at")
            if released is None:
                end = expiry
            else:
                end = min(expiry, _timestamp(released))
            if end < start:
                raise ValueError("lease interval is unavailable")
            by_resource.setdefault(resource, []).append((start, end))
        count = 0
        for intervals in by_resource.values():
            ordered = sorted(intervals)
            for index, left in enumerate(ordered):
                for right in ordered[index + 1 :]:
                    if right[0] >= left[1]:
                        break
                    overlap_start = max(left[0], right[0])
                    overlap_end = min(left[1], right[1])
                    if (
                        overlap_start < overlap_end
                        and overlap_end >= interval_start
                        and overlap_start <= interval_end
                    ):
                        count += 1
        return count

    def _controller_source(
        self,
        *,
        bad_runs: Sequence[str],
        interval_start: float,
        interval_end: float,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        connection = _read_only_connection(self.controller_database)
        try:
            connection.execute("BEGIN")
            _require_schema(
                connection,
                version=CONTROLLER_SCHEMA_VERSION,
                tables={
                    "runs": frozenset(
                        {"id", "status", "namespace_id", "config_json"}
                    ),
                    "iterations": frozenset(
                        {
                            "id", "run_id", "status", "created_at", "updated_at",
                            "candidate_hash", "active_container", "result_json",
                            "outcome",
                        }
                    ),
                    "events": frozenset(
                        {"id", "run_id", "iteration_id", "created_at", "event", "details_json"}
                    ),
                    "evaluation_attempts": frozenset(
                        {
                            "id", "experiment_uid", "run_id", "iteration_id",
                            "status", "baseline_ref_json", "request_json",
                            "result_json", "history_experiment_id",
                        }
                    ),
                },
            )
            runs = _rows(connection, "SELECT * FROM runs ORDER BY created_at, id")
            iterations = _rows(connection, "SELECT * FROM iterations ORDER BY id")
            events = _rows(connection, "SELECT * FROM events ORDER BY id")
            attempts = _rows(
                connection, "SELECT * FROM evaluation_attempts ORDER BY id"
            )
            connection.execute("COMMIT")
        finally:
            connection.close()
        _parse_json_fields(
            runs,
            (
                "config_json", "preflight_json", "workflow_snapshot_json",
                "baseline_ref_json", "history_cutoff_json",
            ),
        )
        _parse_json_fields(iterations, ("experiment_ids_json", "result_json"))
        _parse_json_fields(events, ("details_json",))
        _parse_json_fields(
            attempts, ("baseline_ref_json", "request_json", "result_json")
        )
        valid_run_statuses = {
            "RUNNING", "PROMOTED", "PROPOSAL_READY", "BUDGET_EXHAUSTED",
            "STOPPED", "HARD_FAILED", "FAILED",
        }
        for run in runs:
            if (
                not isinstance(run.get("id"), str)
                or not run["id"]
                or run.get("status") not in valid_run_statuses
            ):
                raise ValueError("Controller run fields are unavailable")
            require_sha256_digest(
                run.get("namespace_id"), field="Controller run namespace"
            )
            _timestamp(run.get("created_at"))
            _timestamp(run.get("updated_at"))
        run_ids = {str(row["id"]) for row in runs}
        for iteration in iterations:
            candidate = iteration.get("candidate_hash")
            if (
                type(iteration.get("id")) is not int
                or iteration["id"] <= 0
                or iteration.get("run_id") not in run_ids
                or iteration.get("status") not in {"RUNNING", "COMPLETED"}
                or (
                    candidate is not None
                    and (
                        not isinstance(candidate, str)
                        or re.fullmatch(r"[0-9a-f]{64}", candidate) is None
                    )
                )
                or (
                    iteration.get("active_container") is not None
                    and (
                        not isinstance(iteration["active_container"], str)
                        or not iteration["active_container"]
                    )
                )
                or (
                    iteration.get("outcome") is not None
                    and not isinstance(iteration["outcome"], str)
                )
            ):
                raise ValueError("Controller iteration fields are unavailable")
            _timestamp(iteration.get("created_at"))
            _timestamp(iteration.get("updated_at"))
        iteration_ids = {int(row["id"]) for row in iterations}
        for event in events:
            if (
                type(event.get("id")) is not int
                or event["id"] <= 0
                or event.get("run_id") not in run_ids
                or (
                    event.get("iteration_id") is not None
                    and event["iteration_id"] not in iteration_ids
                )
                or not isinstance(event.get("event"), str)
                or not event["event"]
            ):
                raise ValueError("Controller event fields are unavailable")
            _timestamp(event.get("created_at"))
        valid_attempt_statuses = {
            "PENDING", "RUNNING", "SUCCEEDED", "FAILED", "UNKNOWN_OUTCOME"
        }
        for attempt in attempts:
            if (
                type(attempt.get("id")) is not int
                or attempt["id"] <= 0
                or attempt.get("run_id") not in run_ids
                or attempt.get("iteration_id") not in iteration_ids
                or attempt.get("status") not in valid_attempt_statuses
                or not isinstance(attempt.get("experiment_uid"), str)
                or not attempt["experiment_uid"]
                or (
                    attempt.get("history_experiment_id") is not None
                    and (
                        type(attempt["history_experiment_id"]) is not int
                        or attempt["history_experiment_id"] <= 0
                    )
                )
            ):
                raise ValueError("Controller evaluation fields are unavailable")
        run_status = {str(row["id"]): str(row["status"]) for row in runs}
        run_namespace = {
            str(row["id"]): str(row["namespace_id"]) for row in runs
        }
        expected_containers = sorted(
            {
                str(row["active_container"])
                for row in iterations
                if row.get("status") != "COMPLETED"
                and run_status.get(str(row["run_id"])) == "RUNNING"
                and isinstance(row.get("active_container"), str)
                and row["active_container"]
            }
        )
        bad = set(bad_runs)
        bad_candidates = [
            (
                int(row["id"]),
                str(row["candidate_hash"]),
                run_namespace[str(row["run_id"])],
            )
            for row in iterations
            if str(row["run_id"]) in bad
            and row.get("outcome")
            in {"HARD_FAILURE", "UNKNOWN_GPU_OUTCOME"}
            and isinstance(row.get("candidate_hash"), str)
            and row["candidate_hash"]
        ]
        retried: set[tuple[int, str, str]] = set()
        for bad_id, candidate, namespace_id in bad_candidates:
            for later in iterations:
                if (
                    int(later["id"]) <= bad_id
                    or later.get("candidate_hash") != candidate
                    or run_namespace.get(str(later["run_id"])) != namespace_id
                ):
                    continue
                updated = _timestamp(later.get("updated_at"))
                if interval_start <= updated <= interval_end:
                    retried.add((bad_id, candidate, namespace_id))
                    break
        confirmed_experiment_uids = sorted(
            {
                str(attempt["experiment_uid"])
                for attempt in attempts
                if attempt.get("status") == "SUCCEEDED"
                and attempt.get("history_experiment_id") is not None
                and isinstance(attempt.get("experiment_uid"), str)
                and attempt["experiment_uid"]
            }
        )
        serialized = {
            "runs": runs,
            "iterations": iterations,
            "events": events,
            "evaluation_attempts": attempts,
        }
        source = _source_record(
            name="controller.sqlite3",
            rows=serialized,
            cursor={
                "run_count": len(runs),
                "iteration_max_id": max((int(row["id"]) for row in iterations), default=0),
                "event_max_id": max((int(row["id"]) for row in events), default=0),
                "evaluation_attempt_max_id": max(
                    (int(row["id"]) for row in attempts), default=0
                ),
            },
            summary={
                "run_count": len(runs),
                "iteration_count": len(iterations),
                "event_count": len(events),
                "evaluation_attempt_count": len(attempts),
                "expected_container_count": len(expected_containers),
                "activity": {
                    "controller_run_ids": sorted(run_status),
                    "confirmed_experiment_uids": confirmed_experiment_uids,
                },
            },
        )
        return source, {
            "hard_fault_retry_count": len(retried),
            "expected_containers": expected_containers,
        }

    def _docker_source(
        self, *, expected_containers: Sequence[str]
    ) -> tuple[dict[str, Any], int]:
        argv = (
            str(self.config.docker_binary),
            "ps",
            "-a",
            "--filter",
            "name=kar-",
            "--format",
            "{{.Names}}",
        )
        result = self._run_fixed(
            argv,
            timeout=DOCKER_INVENTORY_TIMEOUT_SECONDS,
            limit=DOCKER_INVENTORY_MAX_OUTPUT_BYTES,
        )
        names: list[str] = []
        for line in result.stdout.splitlines():
            name = line.strip()
            if not name:
                continue
            if not name.startswith("kar-") or any(char.isspace() for char in name):
                raise ValueError("Docker inventory format is unavailable")
            names.append(name)
        if len(names) != len(set(names)):
            raise ValueError("Docker inventory contains duplicate names")
        actual = set(names)
        expected = set(expected_containers)
        missing = sorted(expected - actual)
        if missing:
            raise ValueError("Controller/Docker active-container evidence disagrees")
        orphans = sorted(actual - expected)
        source = _source_record(
            name="docker-inventory",
            rows={"container_names": sorted(actual)},
            cursor={"container_names": sorted(actual)},
            summary={
                "container_count": len(actual),
                "expected_container_count": len(expected),
                "orphan_container_count": len(orphans),
            },
        )
        return source, len(orphans)

    def collect(
        self, *, interval_start_epoch: float, interval_end_epoch: float
    ) -> SoakObservation:
        start = _finite_epoch(interval_start_epoch, "interval_start_epoch")
        end = _finite_epoch(interval_end_epoch, "interval_end_epoch")
        if end < start:
            raise ValueError("soak observation interval moved backwards")
        snapshot, reasons = self._invariant_snapshot()
        sources: dict[str, Any] = {}
        counts = {field: 0 for field in COUNT_FIELDS}
        bad_runs: Sequence[str] = ()
        expected_containers: Sequence[str] = ()
        try:
            source, derived = self._campaign_source(
                interval_start=start, interval_end=end
            )
            sources["campaign"] = source
            counts["lease_overlap_count"] = derived["lease_overlap_count"]
            counts["budget_leak_count"] = derived["budget_leak_count"]
            bad_runs = derived["bad_controller_runs"]
        except (
            AttributeError,
            OSError,
            sqlite3.Error,
            TypeError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ):
            reasons.append("CAMPAIGN_DB_UNAVAILABLE")
            sources["campaign"] = _unavailable_source(
                "campaign.sqlite3", "CAMPAIGN_DB_UNAVAILABLE"
            )
        try:
            source, derived = self._controller_source(
                bad_runs=bad_runs,
                interval_start=start,
                interval_end=end,
            )
            sources["controller"] = source
            counts["hard_fault_retry_count"] = derived[
                "hard_fault_retry_count"
            ]
            expected_containers = derived["expected_containers"]
        except (
            AttributeError,
            OSError,
            sqlite3.Error,
            TypeError,
            UnicodeError,
            ValueError,
            json.JSONDecodeError,
        ):
            reasons.append("CONTROLLER_DB_UNAVAILABLE")
            sources["controller"] = _unavailable_source(
                "controller.sqlite3", "CONTROLLER_DB_UNAVAILABLE"
            )
        try:
            if sources["controller"]["status"] != "AVAILABLE":
                raise ValueError("expected container ledger is unavailable")
            source, orphan_count = self._docker_source(
                expected_containers=expected_containers
            )
            sources["docker"] = source
            counts["orphan_container_count"] = orphan_count
        except (AttributeError, OSError, TypeError, UnicodeError, ValueError):
            reasons.append("DOCKER_INVENTORY_UNAVAILABLE")
            sources["docker"] = _unavailable_source(
                "docker-inventory", "DOCKER_INVENTORY_UNAVAILABLE"
            )
        return SoakObservation._create_trusted(
            status="UNAVAILABLE" if reasons else "AVAILABLE",
            invariant_snapshot=snapshot,
            interval_start_epoch=start,
            interval_end_epoch=end,
            counts=counts,
            sources=sources,
            unavailable_reasons=reasons,
        )


__all__ = [
    "COLLECTOR_REVISION",
    "COUNT_FIELDS",
    "OBSERVATION_SCHEMA_VERSION",
    "SoakObservation",
    "SoakObservationCollector",
]
