"""Verified disaster restore into a new, inactive runtime root.

Normal resume always uses the live databases.  This module is intentionally
separate: it validates an immutable checkpoint in place, reconstructs a new
runtime tree through a sibling staging directory, validates the copies, and
publishes that *new* directory atomically.  It never overwrites or switches an
active runtime root.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tempfile
from typing import Any, Mapping

from .autorun.store import SCHEMA_VERSION as CONTROLLER_SCHEMA_VERSION
from .campaign.store import SCHEMA_VERSION as CAMPAIGN_SCHEMA_VERSION
from .history import SCHEMA_VERSION as HISTORY_SCHEMA_VERSION
from .platform.identity import BaselineRef, ExperimentIdentity


RESTORE_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 2
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_KNOWN_TOP_LEVEL = frozenset(
    {
        "controller.sqlite3",
        "history.sqlite3",
        "campaign.sqlite3",
        "resolved-config.json",
        "artifacts",
        "state_objects",
        "controller_objects",
        "run",
    }
)
_DATABASE_TABLES = {
    "controller": (
        "runs",
        "iterations",
        "events",
        "proposal_attempts",
        "evaluation_attempts",
    ),
    "history": (
        "research_namespaces",
        "candidate_artifacts",
        "experiments",
        "case_measurements",
        "experiment_relations",
    ),
    # These are the tables covered by the checkpoint producer's schema-v2
    # Campaign summary.  New additive Campaign tables remain byte-covered by the
    # file manifest and require a checkpoint schema revision before becoming
    # part of this frozen summary contract.
    "campaign": (
        "campaigns",
        "baseline_revisions",
        "child_runs",
        "budget_actions",
        "resource_leases",
        "oj_nominations",
        "outbox",
        "soak_generations",
        "soak_violations",
    ),
}
_DATABASE_VERSIONS = {
    "controller": CONTROLLER_SCHEMA_VERSION,
    "history": HISTORY_SCHEMA_VERSION,
    "campaign": CAMPAIGN_SCHEMA_VERSION,
}
_SUMMARY_FIELDS = frozenset(
    {"sha256", "integrity_check", "sqlite_version", "table_counts"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_object(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) > _MAX_MANIFEST_BYTES:
        raise ValueError("checkpoint manifest exceeds its size limit")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError("checkpoint manifest contains duplicate keys")
            result[key] = value
        return result

    def constant(_value: str) -> Any:
        raise ValueError("checkpoint manifest contains a non-finite number")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("checkpoint manifest is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("checkpoint manifest must be a JSON object")
    return value


def _relative_file(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("checkpoint file path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ValueError("checkpoint file path must be normalized relative POSIX")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("checkpoint file path contains a forbidden component")
    if path.parts[0] not in _KNOWN_TOP_LEVEL:
        raise ValueError("checkpoint contains an unrecognized top-level object")
    return path


def _read_only_connection(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _sqlite_integrity(path: Path) -> dict[str, Any]:
    connection = _read_only_connection(path)
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity is None or str(integrity[0]).lower() != "ok":
            raise ValueError(f"SQLite integrity check failed for {path.name}")
        return {
            "user_version": int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            ),
            "integrity_check": "ok",
        }
    finally:
        connection.close()


def _json_mapping(encoded: object, *, name: str) -> dict[str, Any]:
    if not isinstance(encoded, str):
        raise ValueError(f"{name} is not stored as JSON text")
    try:
        value = json.loads(encoded)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _verify_database_summaries(
    root: Path,
    database_manifest: object,
    *,
    campaign_present: bool,
) -> dict[str, dict[str, Any]]:
    if not isinstance(database_manifest, dict):
        raise ValueError("checkpoint database summary is missing")
    required_names = {"controller", "history"}
    if campaign_present:
        required_names.add("campaign")
    if set(database_manifest) != required_names:
        raise ValueError(
            "checkpoint database summary does not exactly match included databases"
        )

    statuses: dict[str, dict[str, Any]] = {}
    for name in sorted(required_names):
        summary = database_manifest[name]
        if not isinstance(summary, dict) or set(summary) != _SUMMARY_FIELDS:
            raise ValueError(f"checkpoint {name} database summary has invalid fields")
        digest = summary["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"checkpoint {name} database summary has invalid SHA-256")
        path = root / f"{name}.sqlite3"
        if _sha256(path) != digest:
            raise ValueError(
                f"checkpoint {name} database summary SHA-256 does not match its file"
            )
        if summary["integrity_check"] != "ok":
            raise ValueError(
                f"checkpoint {name} database summary does not record integrity ok"
            )
        if not isinstance(summary["sqlite_version"], str) or not summary[
            "sqlite_version"
        ]:
            raise ValueError(
                f"checkpoint {name} database summary has no SQLite version"
            )
        expected_tables = _DATABASE_TABLES[name]
        declared_counts = summary["table_counts"]
        if not isinstance(declared_counts, dict) or set(declared_counts) != set(
            expected_tables
        ):
            raise ValueError(
                f"checkpoint {name} database table summary is incomplete"
            )
        if any(
            type(value) is not int or value < 0
            for value in declared_counts.values()
        ):
            raise ValueError(
                f"checkpoint {name} database table counts are invalid"
            )

        status = _sqlite_integrity(path)
        if status["user_version"] != _DATABASE_VERSIONS[name]:
            raise ValueError(
                f"checkpoint {name} database schema version is unsupported"
            )
        connection = _read_only_connection(path)
        try:
            actual_counts = {
                table: int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0]
                )
                for table in expected_tables
            }
        except sqlite3.DatabaseError as exc:
            raise ValueError(
                f"checkpoint {name} database table summary cannot be verified"
            ) from exc
        finally:
            connection.close()
        if actual_counts != declared_counts:
            raise ValueError(
                f"checkpoint {name} database table counts do not match the summary"
            )
        statuses[name] = {**status, "table_counts": actual_counts}
    return statuses


def _artifact_checkpoint_path(root: Path, object_path: object) -> Path:
    if not isinstance(object_path, str) or not object_path:
        raise ValueError("History candidate artifact has no object path")
    relative = PurePosixPath(object_path)
    if (
        relative.is_absolute()
        or relative.as_posix() != object_path
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("History candidate artifact has an unsafe object path")
    if relative.parts[0] == "artifacts":
        return root.joinpath(*relative.parts)
    if relative.parts[0] == "objects":
        return root.joinpath("state_objects", *relative.parts[1:])
    raise ValueError("History candidate artifact uses an unknown storage domain")


def _verify_history_identity(
    root: Path,
    *,
    run_id: str,
    namespace_id: str,
    baseline: BaselineRef,
) -> None:
    history = _read_only_connection(root / "history.sqlite3")
    controller = _read_only_connection(root / "controller.sqlite3")
    try:
        artifact = history.execute(
            """
            SELECT object_path, content_sha256 FROM candidate_artifacts
            WHERE artifact_id = ?
            """,
            (str(baseline.artifact_id),),
        ).fetchone()
        if artifact is None:
            raise ValueError(
                "History does not contain the frozen Run baseline artifact"
            )
        object_file = _artifact_checkpoint_path(root, artifact["object_path"])
        if (
            not object_file.is_file()
            or object_file.is_symlink()
            or _sha256(object_file) != artifact["content_sha256"]
        ):
            raise ValueError(
                "frozen Run baseline artifact is absent or corrupted in checkpoint"
            )

        revision = baseline.revision
        preferred_id = (
            int(revision.removeprefix("history-"))
            if revision.startswith("history-")
            and revision.removeprefix("history-").isdigit()
            else None
        )
        if preferred_id is None:
            evidence = history.execute(
                """
                SELECT id FROM experiments
                WHERE namespace_id = ? AND artifact_id = ?
                  AND status = 'SUCCESS' AND promotable = 1
                ORDER BY id DESC LIMIT 1
                """,
                (namespace_id, str(baseline.artifact_id)),
            ).fetchone()
        else:
            evidence = history.execute(
                """
                SELECT id FROM experiments
                WHERE id = ? AND namespace_id = ? AND artifact_id = ?
                  AND status = 'SUCCESS' AND promotable = 1
                """,
                (preferred_id, namespace_id, str(baseline.artifact_id)),
            ).fetchone()
        if evidence is None:
            raise ValueError(
                "History does not prove the frozen Run baseline experiment"
            )

        linked = controller.execute(
            """
            SELECT experiment_uid, candidate_artifact_id,
                   history_experiment_id
            FROM evaluation_attempts
            WHERE run_id = ? AND history_experiment_id IS NOT NULL
            ORDER BY id
            """,
            (run_id,),
        ).fetchall()
        for attempt in linked:
            experiment = history.execute(
                """
                SELECT experiment_uid, namespace_id, artifact_id
                FROM experiments WHERE id = ?
                """,
                (attempt["history_experiment_id"],),
            ).fetchone()
            if (
                experiment is None
                or experiment["experiment_uid"] != attempt["experiment_uid"]
                or experiment["namespace_id"] != namespace_id
                or (
                    attempt["candidate_artifact_id"] is not None
                    and experiment["artifact_id"]
                    != attempt["candidate_artifact_id"]
                )
            ):
                raise ValueError(
                    "Controller evaluation link differs from trusted History"
                )
    finally:
        controller.close()
        history.close()


def _verify_staged_baseline_evidence(
    root: Path,
    *,
    campaign_id: str,
    namespace_id: str,
    staged_baseline: BaselineRef,
    parent_baseline: BaselineRef,
    primary_uid: str,
    confirmation_uid: str,
) -> None:
    """Prove a staged Campaign revision across all three durable stores.

    A Campaign row is only a lineage pointer.  It is not scientific evidence by
    itself, so both UIDs must resolve to the exact immutable History records and
    to succeeded, History-linked Controller attempts.  This prevents a
    structurally plausible Campaign database from being grafted onto an
    otherwise valid Controller/History checkpoint.
    """

    history = _read_only_connection(root / "history.sqlite3")
    controller = _read_only_connection(root / "controller.sqlite3")
    try:
        rows = history.execute(
            """
            SELECT id, experiment_uid, namespace_id, artifact_id,
                   candidate_hash, backend, suite, status, promotable,
                   condition_digest, replicate_kind, replicate_index,
                   baseline_experiment_uid, identity_json, result_json
            FROM experiments
            WHERE experiment_uid IN (?, ?)
            ORDER BY id
            """,
            (primary_uid, confirmation_uid),
        ).fetchall()
        by_uid = {str(row["experiment_uid"]): row for row in rows}
        if set(by_uid) != {primary_uid, confirmation_uid}:
            raise ValueError(
                "Campaign staged baseline UIDs are not both present in History"
            )
        primary = by_uid[primary_uid]
        confirmation = by_uid[confirmation_uid]
        try:
            primary_identity = ExperimentIdentity.from_value(
                _json_mapping(
                    primary["identity_json"], name="staged primary identity"
                )
            )
            confirmation_identity = ExperimentIdentity.from_value(
                _json_mapping(
                    confirmation["identity_json"],
                    name="staged confirmation identity",
                )
            )
            primary_result = _json_mapping(
                primary["result_json"], name="staged primary result"
            )
            confirmation_result = _json_mapping(
                confirmation["result_json"],
                name="staged confirmation result",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Campaign staged baseline has invalid V2 History evidence"
            ) from exc

        primary_promotion = primary_result.get("promotion")
        confirmation_promotion = confirmation_result.get("promotion")
        primary_decision = (
            primary_promotion.get("decision")
            if isinstance(primary_promotion, Mapping)
            else None
        )
        confirmation_decision = (
            confirmation_promotion.get("decision")
            if isinstance(confirmation_promotion, Mapping)
            else None
        )
        baseline_id = (
            primary_promotion.get("baseline_experiment_id")
            if isinstance(primary_promotion, Mapping)
            else None
        )
        staged_artifact = str(staged_baseline.artifact_id)
        parent_artifact = str(parent_baseline.artifact_id)
        if (
            not staged_baseline.execution_environment.is_resolved
            or not parent_baseline.execution_environment.is_resolved
            or staged_baseline.execution_environment
            != parent_baseline.execution_environment
            or primary["namespace_id"] != namespace_id
            or confirmation["namespace_id"] != namespace_id
            or primary["artifact_id"] != staged_artifact
            or confirmation["artifact_id"] != staged_artifact
            or primary["candidate_hash"] != confirmation["candidate_hash"]
            or primary["backend"] != "c500"
            or confirmation["backend"] != "c500"
            or primary["suite"] != "full"
            or confirmation["suite"] != "full"
            or primary["status"] != "SUCCESS"
            or confirmation["status"] != "SUCCESS"
            or bool(primary["promotable"])
            or not bool(confirmation["promotable"])
            or primary["replicate_kind"] != "primary"
            or confirmation["replicate_kind"] != "confirmation"
            or primary_identity.experiment_uid != primary_uid
            or confirmation_identity.experiment_uid != confirmation_uid
            or primary_identity.namespace_id != namespace_id
            or confirmation_identity.namespace_id != namespace_id
            or str(primary_identity.candidate_artifact_id) != staged_artifact
            or str(confirmation_identity.candidate_artifact_id)
            != staged_artifact
            or str(primary_identity.parent_artifact_id) != parent_artifact
            or str(confirmation_identity.parent_artifact_id) != parent_artifact
            or primary_identity.baseline != parent_baseline
            or confirmation_identity.baseline != parent_baseline
            or primary_identity.execution_environment
            != staged_baseline.execution_environment
            or confirmation_identity.execution_environment
            != staged_baseline.execution_environment
            or primary_identity.stage != "full_primary"
            or confirmation_identity.stage != "confirmation"
            or primary_identity.suite != "full"
            or confirmation_identity.suite != "full"
            or primary_identity.replicate_kind != "primary"
            or confirmation_identity.replicate_kind != "confirmation"
            or primary_identity.replicate_index != primary["replicate_index"]
            or confirmation_identity.replicate_index
            != confirmation["replicate_index"]
            or primary_identity.condition_digest != primary["condition_digest"]
            or confirmation_identity.condition_digest
            != confirmation["condition_digest"]
            or primary_identity.campaign_id != campaign_id
            or confirmation_identity.campaign_id != campaign_id
            or primary_identity.run_id != confirmation_identity.run_id
            or primary_identity.iteration != confirmation_identity.iteration
            or primary["baseline_experiment_uid"] is None
            or confirmation["baseline_experiment_uid"]
            != primary["baseline_experiment_uid"]
            or primary_result.get("status") != "SUCCESS"
            or confirmation_result.get("status") != "SUCCESS"
            or not isinstance(primary_promotion, Mapping)
            or primary_promotion.get("phase") != "primary"
            or primary_promotion.get("reason") != "confirmation_required"
            or not isinstance(primary_decision, Mapping)
            or primary_decision.get("promoted") is not False
            or primary_decision.get("needs_confirmation") is not True
            or primary_decision.get("reason") != "confirmation_required"
            or not isinstance(confirmation_promotion, Mapping)
            or confirmation_promotion.get("phase") != "confirmation"
            or confirmation_promotion.get("reason") != "promoted"
            or confirmation_promotion.get("confirmed") is not True
            or confirmation_promotion.get("primary_experiment_id")
            != primary["id"]
            or not isinstance(confirmation_decision, Mapping)
            or confirmation_decision.get("promoted") is not True
            or confirmation_decision.get("needs_confirmation") is not False
            or confirmation_decision.get("reason") != "promoted"
            or type(baseline_id) is not int
            or confirmation_promotion.get("baseline_experiment_id")
            != baseline_id
            or primary_promotion.get("baseline_candidate_hash")
            != confirmation_promotion.get("baseline_candidate_hash")
        ):
            raise ValueError(
                "Campaign staged baseline does not match its V2 promotion evidence"
            )

        for experiment in (primary, confirmation):
            measurements = history.execute(
                """
                SELECT passed, matched_ratio FROM case_measurements
                WHERE experiment_id = ? ORDER BY id
                """,
                (experiment["id"],),
            ).fetchall()
            if not measurements or any(
                row["passed"] != 1
                or row["matched_ratio"] is None
                or not math.isfinite(float(row["matched_ratio"]))
                or float(row["matched_ratio"]) < 1.0
                for row in measurements
            ):
                raise ValueError(
                    "Campaign staged baseline has incomplete correctness evidence"
                )

        baseline = history.execute(
            """
            SELECT id, experiment_uid, namespace_id, artifact_id,
                   candidate_hash, status, promotable, identity_json
            FROM experiments WHERE id = ?
            """,
            (baseline_id,),
        ).fetchone()
        try:
            baseline_identity = (
                None
                if baseline is None
                else ExperimentIdentity.from_value(
                    _json_mapping(
                        baseline["identity_json"],
                        name="staged parent History identity",
                    )
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Campaign staged baseline parent has invalid History identity"
            ) from exc
        if (
            baseline is None
            or baseline["experiment_uid"]
            != primary["baseline_experiment_uid"]
            or baseline["namespace_id"] != namespace_id
            or baseline["artifact_id"] != parent_artifact
            or baseline["status"] != "SUCCESS"
            or not bool(baseline["promotable"])
            or baseline["candidate_hash"]
            != primary_promotion.get("baseline_candidate_hash")
            or baseline_identity is None
            or baseline_identity.namespace_id != namespace_id
            or str(baseline_identity.candidate_artifact_id) != parent_artifact
            or baseline_identity.execution_environment
            != parent_baseline.execution_environment
        ):
            raise ValueError(
                "Campaign staged baseline parent is not proven by History"
            )

        relations = history.execute(
            """
            SELECT target_experiment_uid, metadata_json
            FROM experiment_relations
            WHERE source_experiment_uid = ?
              AND relation_type = 'confirmation_of'
            ORDER BY id
            """,
            (primary_uid,),
        ).fetchall()
        if len(relations) != 1:
            raise ValueError(
                "Campaign staged baseline has no unique confirmation relation"
            )
        relation_metadata = _json_mapping(
            relations[0]["metadata_json"],
            name="staged confirmation relation metadata",
        )
        if (
            relations[0]["target_experiment_uid"] != confirmation_uid
            or relation_metadata != {"baseline_experiment_id": baseline_id}
        ):
            raise ValueError(
                "Campaign staged baseline confirmation relation is inconsistent"
            )

        for experiment, identity, expected_stage, expected_kind in (
            (primary, primary_identity, "full_primary", "primary"),
            (
                confirmation,
                confirmation_identity,
                "confirmation",
                "confirmation",
            ),
        ):
            attempt = controller.execute(
                """
                SELECT a.status, a.run_id, a.stage, a.suite,
                       a.replicate_kind, a.replicate_index,
                       a.candidate_artifact_id, a.parent_artifact_id,
                       a.baseline_ref_json, a.condition_digest,
                       a.request_json, a.result_json,
                       a.history_experiment_id, i.iteration_index,
                       r.status AS run_status, r.namespace_id AS run_namespace_id,
                       r.baseline_ref_json AS run_baseline_ref_json
                FROM evaluation_attempts AS a
                JOIN iterations AS i ON i.id = a.iteration_id
                JOIN runs AS r ON r.id = a.run_id
                WHERE a.experiment_uid = ?
                """,
                (experiment["experiment_uid"],),
            ).fetchone()
            if attempt is None:
                raise ValueError(
                    "Campaign staged baseline is not linked by Controller"
                )
            try:
                attempt_baseline = BaselineRef.from_value(
                    _json_mapping(
                        attempt["baseline_ref_json"],
                        name="staged Controller attempt BaselineRef",
                    )
                )
                run_baseline = BaselineRef.from_value(
                    _json_mapping(
                        attempt["run_baseline_ref_json"],
                        name="staged Controller Run BaselineRef",
                    )
                )
                attempt_request = _json_mapping(
                    attempt["request_json"],
                    name="staged Controller attempt request",
                )
                attempt_result = _json_mapping(
                    attempt["result_json"],
                    name="staged Controller attempt result",
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Campaign staged Controller evidence is invalid"
                ) from exc
            if (
                attempt["status"] != "SUCCEEDED"
                or attempt["run_id"] != identity.run_id
                or attempt["run_status"] != "PROMOTED"
                or attempt["run_namespace_id"] != namespace_id
                or attempt["stage"] != expected_stage
                or attempt["suite"] != "full"
                or attempt["replicate_kind"] != expected_kind
                or attempt["replicate_index"] != identity.replicate_index
                or attempt["candidate_artifact_id"] != staged_artifact
                or attempt["parent_artifact_id"] != parent_artifact
                or attempt_baseline != parent_baseline
                or run_baseline != parent_baseline
                or attempt["condition_digest"] != identity.condition_digest
                or attempt_request != identity.to_dict()
                or attempt_result
                != _json_mapping(
                    experiment["result_json"],
                    name="staged History result replay",
                )
                or attempt["history_experiment_id"] != experiment["id"]
                or attempt["iteration_index"] != identity.iteration
            ):
                raise ValueError(
                    "Campaign staged Controller/History evidence link is inconsistent"
                )
    finally:
        controller.close()
        history.close()


def _verify_cross_database_identity(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    campaign_present: bool,
) -> None:
    run_id = manifest["run_id"]
    controller = _read_only_connection(root / "controller.sqlite3")
    try:
        run = controller.execute(
            """
            SELECT id, status, namespace_id, config_json,
                   workflow_snapshot_json, baseline_ref_json
            FROM runs WHERE id = ?
            """,
            (run_id,),
        ).fetchone()
    finally:
        controller.close()
    if run is None:
        raise ValueError("Controller database does not contain the checkpoint run")
    if manifest.get("run_status") != run["status"]:
        raise ValueError("checkpoint run status differs from Controller")

    workflow = _json_mapping(
        run["workflow_snapshot_json"], name="Controller workflow snapshot"
    )
    config = _json_mapping(run["config_json"], name="Controller Run config")
    try:
        baseline = BaselineRef.from_value(
            _json_mapping(run["baseline_ref_json"], name="Controller BaselineRef")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Controller Run has an invalid BaselineRef") from exc
    namespace_id = run["namespace_id"]
    if baseline.namespace_id != namespace_id:
        raise ValueError("Controller Run baseline belongs to another namespace")
    if manifest.get("config") != config:
        raise ValueError("checkpoint config differs from the frozen Controller Run")
    resolved_config = _strict_json_object(root / "resolved-config.json")
    if resolved_config != config:
        raise ValueError("resolved checkpoint config differs from Controller Run")
    try:
        max_candidates = config["max_candidates"]
        max_hours = config["max_hours"]
        max_failures = config["max_consecutive_failures"]
        if (
            type(max_candidates) is not int
            or max_candidates <= 0
            or isinstance(max_hours, bool)
            or not isinstance(max_hours, (int, float))
            or not math.isfinite(float(max_hours))
            or float(max_hours) <= 0
            or type(max_failures) is not int
            or max_failures <= 0
        ):
            raise ValueError
        max_wall_seconds = round(float(max_hours) * 3600)
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        raise ValueError("Controller Run has invalid frozen child limits") from exc

    campaign_id = workflow.get("campaign_id")
    campaign_owned = campaign_id is not None or baseline.source == "campaign"
    if campaign_owned:
        if (
            not isinstance(campaign_id, str)
            or not campaign_id
            or baseline.source != "campaign"
        ):
            raise ValueError(
                "Controller Run has inconsistent Campaign ownership identity"
            )
        if not campaign_present or manifest.get("campaign_id") != campaign_id:
            raise ValueError(
                "Campaign checkpoint manifest differs from Controller ownership"
            )
    elif campaign_present or manifest.get("campaign_id") is not None:
        raise ValueError(
            "non-Campaign Controller Run must not contain a Campaign database"
        )

    _verify_history_identity(
        root,
        run_id=run_id,
        namespace_id=namespace_id,
        baseline=baseline,
    )
    if not campaign_owned:
        return

    child_snapshot = workflow.get("campaign_child")
    if not isinstance(child_snapshot, dict):
        raise ValueError("Campaign Controller Run has no frozen child identity")
    campaign = _read_only_connection(root / "campaign.sqlite3")
    try:
        campaign_row = campaign.execute(
            """
            SELECT id, namespace_id, mode, snapshot_digest,
                   active_baseline_revision_id
            FROM campaigns WHERE id = ?
            """,
            (campaign_id,),
        ).fetchone()
        child = campaign.execute(
            """
            SELECT id, child_index, campaign_id, controller_run_id,
                   baseline_revision_id, proposer_profile_json,
                   max_candidates, max_wall_seconds,
                   max_consecutive_failures, stop_after_promotion, status
            FROM child_runs WHERE controller_run_id = ?
            """,
            (run_id,),
        ).fetchone()
        baseline_revision_id = child_snapshot.get("baseline_revision_id")
        if baseline_revision_id is None and child is not None:
            baseline_revision_id = child["baseline_revision_id"]
        if not isinstance(baseline_revision_id, str) or not baseline_revision_id:
            raise ValueError(
                "Campaign Controller Run has no local baseline row identity"
            )
        baseline_row = campaign.execute(
            """
            SELECT namespace_id, parent_revision_id, artifact_id, baseline_ref_json,
                   revision_kind, primary_experiment_uid,
                   confirmation_experiment_uid
            FROM baseline_revisions
            WHERE id = ? AND campaign_id = ?
            """,
            (baseline_revision_id, campaign_id),
        ).fetchone()
        parent_baseline_row = (
            None
            if baseline_row is None or baseline_row["parent_revision_id"] is None
            else campaign.execute(
                """
                SELECT namespace_id, artifact_id, baseline_ref_json
                FROM baseline_revisions
                WHERE id = ? AND campaign_id = ?
                """,
                (baseline_row["parent_revision_id"], campaign_id),
            ).fetchone()
        )
    finally:
        campaign.close()
    try:
        stored_baseline = (
            None
            if baseline_row is None or baseline_row["baseline_ref_json"] is None
            else BaselineRef.from_value(
                _json_mapping(
                    baseline_row["baseline_ref_json"],
                    name="Campaign baseline reference",
                )
            )
        )
        child_proposer = (
            None
            if child is None
            else _json_mapping(
                child["proposer_profile_json"],
                name="Campaign child proposer profile",
            )
        )
        parent_baseline = (
            None
            if parent_baseline_row is None
            or parent_baseline_row["baseline_ref_json"] is None
            else BaselineRef.from_value(
                _json_mapping(
                    parent_baseline_row["baseline_ref_json"],
                    name="Campaign parent baseline reference",
                )
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Campaign checkpoint contains invalid child/baseline JSON") from exc
    expected_child = {
        "child_id": child_snapshot.get("child_id"),
        "child_index": child_snapshot.get("child_index"),
        "controller_run_id": child_snapshot.get("controller_run_id"),
    }
    if (
        campaign_row is None
        or campaign_row["namespace_id"] != namespace_id
        or campaign_row["mode"] != workflow.get("mode")
        or campaign_row["snapshot_digest"]
        != workflow.get("campaign_snapshot_digest")
        or campaign_row["active_baseline_revision_id"] != baseline_revision_id
        or baseline_row is None
        or baseline_row["namespace_id"] != namespace_id
        or baseline_row["artifact_id"] != str(baseline.artifact_id)
        or stored_baseline != baseline
        or child is None
        or child["id"] != expected_child["child_id"]
        or child["child_index"] != expected_child["child_index"]
        or child["controller_run_id"] != expected_child["controller_run_id"]
        or child["campaign_id"] != campaign_id
        or child["baseline_revision_id"] != baseline_revision_id
        or child_proposer != workflow.get("proposer_profile")
        or child["max_candidates"] != max_candidates
        or child["max_wall_seconds"] != max_wall_seconds
        or child["max_consecutive_failures"] != max_failures
        or not bool(child["stop_after_promotion"])
        or child["status"] not in {"RUNNING", "PROMOTED"}
        or (
            baseline_row["revision_kind"] == "DEPLOYMENT_SEED"
            and (
                baseline_row["primary_experiment_uid"] is not None
                or baseline_row["confirmation_experiment_uid"] is not None
            )
        )
        or (
            baseline_row["revision_kind"] == "STAGED"
            and (
                not isinstance(baseline_row["primary_experiment_uid"], str)
                or not baseline_row["primary_experiment_uid"]
                or not isinstance(
                    baseline_row["confirmation_experiment_uid"], str
                )
                or not baseline_row["confirmation_experiment_uid"]
                or baseline_row["primary_experiment_uid"]
                == baseline_row["confirmation_experiment_uid"]
            )
        )
        or baseline_row["revision_kind"] not in {"DEPLOYMENT_SEED", "STAGED"}
    ):
        raise ValueError(
            "Campaign database does not prove the frozen Controller child/baseline"
        )
    if baseline_row["revision_kind"] == "STAGED":
        if (
            parent_baseline is None
            or parent_baseline_row is None
            or parent_baseline.namespace_id != namespace_id
            or parent_baseline_row["namespace_id"] != namespace_id
            or str(parent_baseline.artifact_id)
            != parent_baseline_row["artifact_id"]
        ):
            raise ValueError(
                "Campaign staged baseline has no consistent parent revision"
            )
        _verify_staged_baseline_evidence(
            root,
            campaign_id=campaign_id,
            namespace_id=namespace_id,
            staged_baseline=baseline,
            parent_baseline=parent_baseline,
            primary_uid=str(baseline_row["primary_experiment_uid"]),
            confirmation_uid=str(
                baseline_row["confirmation_experiment_uid"]
            ),
        )


def verify_checkpoint(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Verify every checkpoint byte and both SQLite databases read-only."""

    supplied_root = Path(checkpoint_dir)
    if supplied_root.is_symlink():
        raise ValueError("checkpoint must be a real directory")
    root = supplied_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("checkpoint must be a real directory")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("checkpoint manifest is missing or is a symlink")
    manifest = _strict_json_object(manifest_path)
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint manifest schema_version must be 2")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("checkpoint manifest has no run_id")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("checkpoint manifest files must be a list")

    expected: dict[str, tuple[int, str]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "size_bytes", "sha256"}:
            raise ValueError("checkpoint file record has invalid fields")
        relative = _relative_file(row["path"])
        path_text = relative.as_posix()
        if path_text in expected:
            raise ValueError("checkpoint manifest contains duplicate file paths")
        size = row["size_bytes"]
        digest = row["sha256"]
        if type(size) is not int or size < 0:
            raise ValueError("checkpoint file size must be non-negative")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("checkpoint file digest must be lowercase SHA-256")
        expected[path_text] = (size, digest)

    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("checkpoint must not contain symlinks")
        if path.is_file() and path != manifest_path:
            actual.add(path.relative_to(root).as_posix())
    if actual != set(expected):
        raise ValueError("checkpoint manifest does not exactly cover its files")
    for relative, (size, digest) in expected.items():
        path = root.joinpath(*PurePosixPath(relative).parts)
        if path.stat().st_size != size or _sha256(path) != digest:
            raise ValueError(f"checkpoint file verification failed: {relative}")

    required = {"controller.sqlite3", "history.sqlite3"}
    if not required.issubset(actual):
        raise ValueError("checkpoint is missing one or more databases")
    campaign_id = manifest.get("campaign_id")
    campaign_present = "campaign.sqlite3" in actual
    if campaign_id is None:
        pass
    elif not isinstance(campaign_id, str) or not campaign_id:
        raise ValueError("checkpoint campaign_id must be null or non-empty")
    database_status = _verify_database_summaries(
        root,
        manifest.get("databases"),
        campaign_present=campaign_present,
    )
    _verify_cross_database_identity(
        root,
        manifest,
        campaign_present=campaign_present,
    )
    return {
        "schema_version": RESTORE_SCHEMA_VERSION,
        "checkpoint": str(root),
        "run_id": run_id,
        "file_count": len(actual),
        "databases": database_status,
        "manifest": manifest,
    }


def _destination_path(relative: PurePosixPath, *, run_id: str) -> PurePosixPath:
    parts = relative.parts
    if parts[0] == "controller.sqlite3":
        return PurePosixPath("controller/controller.sqlite3")
    if parts[0] == "history.sqlite3":
        return PurePosixPath("state/history.sqlite3")
    if parts[0] == "campaign.sqlite3":
        return PurePosixPath("campaign/campaign.sqlite3")
    if parts[0] == "resolved-config.json":
        return PurePosixPath("resolved-config.json")
    prefixes = {
        "artifacts": ("state", "artifacts"),
        "state_objects": ("state", "objects"),
        "controller_objects": ("controller", "objects"),
        "run": ("controller", "runs", run_id),
    }
    return PurePosixPath(*prefixes[parts[0]], *parts[1:])


def restore_checkpoint(
    checkpoint_dir: str | Path, destination_root: str | Path
) -> dict[str, Any]:
    """Restore to a newly created runtime root without switching it active."""

    verification = verify_checkpoint(checkpoint_dir)
    source = Path(verification["checkpoint"])
    destination = Path(destination_root)
    if not destination.is_absolute():
        raise ValueError("restore destination must be absolute")
    normalized = Path(os.path.normpath(str(destination)))
    if normalized != destination or destination.resolve(strict=False) != destination:
        raise ValueError("restore destination must be canonical")
    if destination.exists() or destination.is_symlink():
        raise ValueError("restore destination must be a new runtime root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.restore-", dir=destination.parent
        )
    )
    staging.chmod(0o700)
    try:
        manifest = verification["manifest"]
        for row in manifest["files"]:
            relative = _relative_file(row["path"])
            target_relative = _destination_path(
                relative, run_id=verification["run_id"]
            )
            target = staging.joinpath(*target_relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source.joinpath(*relative.parts), target)
            target.chmod(0o600)
        shutil.copyfile(source / "manifest.json", staging / "restore-manifest.json")
        (staging / "restore-manifest.json").chmod(0o600)
        for directory in sorted(
            (path for path in staging.rglob("*") if path.is_dir()),
            key=lambda value: len(value.parts),
            reverse=True,
        ):
            directory.chmod(0o700)
        _sqlite_integrity(staging / "controller" / "controller.sqlite3")
        _sqlite_integrity(staging / "state" / "history.sqlite3")
        campaign_database = staging / "campaign" / "campaign.sqlite3"
        if campaign_database.is_file():
            _sqlite_integrity(campaign_database)
        os.replace(staging, destination)
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "schema_version": RESTORE_SCHEMA_VERSION,
        "command": "restore-checkpoint",
        "status": "READY_FOR_MANUAL_SWITCH",
        "run_id": verification["run_id"],
        "checkpoint": str(source),
        "destination": str(destination),
        "file_count": verification["file_count"],
        "switched_active_runtime": False,
    }


__all__ = ["restore_checkpoint", "verify_checkpoint"]
