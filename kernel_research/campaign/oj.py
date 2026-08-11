"""Offline export of immutable, operator-submitted OJ packages.

This module deliberately has no HTTP client, credential handling, or submit
operation.  It turns one durable CampaignStore nomination and one validated
candidate CAS object into a self-checking directory for a human operator.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any
import unicodedata

from ..history import ExperimentRecord, HistoryStore
from ..platform.artifacts import ArtifactId
from ..platform.canonical import canonical_json_bytes, canonical_sha256
from ..platform.identity import BaselineRef, ExperimentIdentity
from ..platform.proposal import (
    CandidateBundle,
    CandidateFile,
    SOURCE_BUNDLE_V1,
)
from .store import CampaignStore
from .supervisor import PromotionEvidence


PACKAGE_SCHEMA_VERSION = 1
MAX_STORED_BUNDLE_BYTES = 32 * 1024 * 1024
_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

ArtifactReader = Callable[[str], bytes]


def _child_for_campaign(
    store: CampaignStore, campaign_id: str, child_id: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign = store.get_campaign(campaign_id)
    child = store.get_child_run(child_id)
    if child["campaign_id"] != campaign_id:
        raise ValueError("OJ child does not belong to campaign")
    if not isinstance(child.get("controller_run_id"), str):
        raise ValueError("OJ child has no durable Controller run identity")
    if child["status"] in {"PENDING", "RUNNING"}:
        raise ValueError("OJ nomination requires a terminal Campaign child")
    return campaign, child


def _identity_for_child(
    store: CampaignStore,
    record: ExperimentRecord,
    *,
    campaign: Mapping[str, Any],
    child: Mapping[str, Any],
    artifact_id: str,
) -> ExperimentIdentity:
    local_revision_id = child.get("baseline_revision_id")
    if not isinstance(local_revision_id, str) or not local_revision_id:
        raise ValueError("OJ child has no frozen local baseline revision")
    revisions = [
        revision
        for revision in store.list_baseline_revisions(str(campaign["id"]))
        if revision.get("id") == local_revision_id
    ]
    if len(revisions) != 1:
        raise ValueError(
            "OJ child baseline revision does not belong to its Campaign"
        )
    revision = revisions[0]
    if (
        revision.get("id") != local_revision_id
        or revision.get("campaign_id") != campaign["id"]
        or revision.get("namespace_id") != campaign["namespace_id"]
    ):
        raise ValueError("OJ child baseline local identity is inconsistent")
    baseline_value = revision.get("baseline_ref")
    try:
        stored_baseline = BaselineRef.from_value(
            dict(baseline_value)
            if isinstance(baseline_value, Mapping)
            else baseline_value
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "OJ child baseline revision has no valid stored BaselineRef"
        ) from exc
    if (
        stored_baseline.source != "campaign"
        or stored_baseline.namespace_id != revision["namespace_id"]
        or str(stored_baseline.artifact_id) != revision["artifact_id"]
    ):
        raise ValueError(
            "OJ child stored BaselineRef differs from its local revision"
        )
    try:
        identity = ExperimentIdentity.from_value(record.identity)
    except (TypeError, ValueError) as exc:
        raise ValueError("OJ evidence has an invalid experiment identity") from exc
    if (
        record.namespace_id != campaign["namespace_id"]
        or identity.namespace_id != campaign["namespace_id"]
        or identity.campaign_id != campaign["id"]
        or identity.run_id != child["controller_run_id"]
        or str(identity.candidate_artifact_id) != artifact_id
        or record.artifact_id != artifact_id
        or identity.baseline != stored_baseline
        or identity.baseline.namespace_id != revision["namespace_id"]
        or str(identity.baseline.artifact_id) != revision["artifact_id"]
    ):
        raise ValueError("OJ evidence differs from the frozen Campaign child")
    return identity


def _correct(record: ExperimentRecord) -> bool:
    return bool(record.case_measurements) and all(
        case.passed is True
        and case.matched_ratio is not None
        and case.matched_ratio >= 1.0
        for case in record.case_measurements
    )


def _decision(record: ExperimentRecord) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    promotion = record.result.get("promotion")
    decision = promotion.get("decision") if isinstance(promotion, Mapping) else None
    if not isinstance(promotion, Mapping) or not isinstance(decision, Mapping):
        raise ValueError("OJ evidence has no trusted promotion decision")
    return promotion, decision


def _record_summary(record: ExperimentRecord) -> dict[str, Any]:
    promotion, decision = _decision(record)
    return {
        "experiment_uid": record.experiment_uid,
        "experiment_id": record.id,
        "artifact_id": record.artifact_id,
        "aggregate_score": record.aggregate_score,
        "replicate_kind": record.replicate_kind,
        "promotion_phase": promotion.get("phase"),
        "promotion_reason": promotion.get("reason"),
        "decision": dict(decision),
        "cases": [
            {
                "name": case.name,
                "matched_ratio": case.matched_ratio,
                "passed": case.passed,
                "p50_us": case.metrics.get("p50_us"),
            }
            for case in record.case_measurements
        ],
    }


def _require_artifact(history: HistoryStore, artifact_id: str) -> None:
    ArtifactId.parse(artifact_id)
    try:
        history.read_candidate_artifact(artifact_id)
    except (KeyError, RuntimeError) as exc:
        raise ValueError("OJ candidate artifact is missing or corrupted") from exc


def nominate_primary_from_child(
    store: CampaignStore,
    history: HistoryStore,
    *,
    campaign_id: str,
    child_id: int,
) -> dict[str, Any]:
    """Nominate only a Controller/History-proven primary+confirmation pair."""

    campaign, child = _child_for_campaign(store, campaign_id, child_id)
    if child["status"] != "PROMOTED":
        raise ValueError("PRIMARY_SUBMISSION requires a promoted child")
    result = child.get("result")
    promotion_value = result.get("promotion") if isinstance(result, Mapping) else None
    if not isinstance(promotion_value, Mapping):
        raise ValueError("promoted child has no immutable promotion evidence")
    promotion = PromotionEvidence.from_mapping(promotion_value)
    if promotion.namespace_id != campaign["namespace_id"]:
        raise ValueError("promotion evidence belongs to another namespace")
    primary = history.get_experiment_by_uid(promotion.primary_experiment_uid)
    confirmation = history.get_experiment_by_uid(
        promotion.confirmation_experiment_uid
    )
    if primary is None or confirmation is None:
        raise ValueError("promotion evidence is missing from History")
    primary_identity = _identity_for_child(
        store,
        primary,
        campaign=campaign,
        child=child,
        artifact_id=promotion.artifact_id,
    )
    confirmation_identity = _identity_for_child(
        store,
        confirmation,
        campaign=campaign,
        child=child,
        artifact_id=promotion.artifact_id,
    )
    primary_promotion, _ = _decision(primary)
    confirmation_promotion, confirmation_decision = _decision(confirmation)
    if (
        primary.status != "SUCCESS"
        or confirmation.status != "SUCCESS"
        or primary_identity.replicate_kind != "primary"
        or confirmation_identity.replicate_kind != "confirmation"
        or primary_identity.stage != "full_primary"
        or confirmation_identity.stage != "confirmation"
        or primary_promotion.get("phase") != "primary"
        or confirmation_promotion.get("phase") != "confirmation"
        or confirmation_promotion.get("primary_experiment_id") != primary.id
        or confirmation_promotion.get("confirmed") is not True
        or confirmation_decision.get("promoted") is not True
        or not confirmation.promotable
        or not _correct(primary)
        or not _correct(confirmation)
    ):
        raise ValueError("PRIMARY_SUBMISSION evidence does not prove promotion")
    _require_artifact(history, promotion.artifact_id)
    local_metrics = {
        "schema_version": 1,
        "eligibility": "PRIMARY_SUBMISSION",
        "checkpoint_digest": promotion.checkpoint_digest,
        "primary": _record_summary(primary),
        "confirmation": _record_summary(confirmation),
    }
    manifest_digest = canonical_sha256(
        {
            "campaign_id": campaign_id,
            "namespace_id": campaign["namespace_id"],
            "artifact_id": promotion.artifact_id,
            "baseline_revision_id": child["baseline_revision_id"],
            "local_metrics": local_metrics,
        }
    )
    return store.create_oj_nomination(
        campaign_id,
        nomination_type="PRIMARY_SUBMISSION",
        artifact_id=promotion.artifact_id,
        bundle_object_id=promotion.artifact_id,
        manifest_digest=manifest_digest,
        local_metrics=local_metrics,
        baseline_revision_id=child["baseline_revision_id"],
    )


def _exploratory_metrics(
    record: ExperimentRecord,
) -> tuple[float, float, dict[str, float]] | None:
    if record.status != "SUCCESS" or record.promotable or not _correct(record):
        return None
    try:
        promotion, decision = _decision(record)
        aggregate = float(decision["aggregate_speedup"])
        confirmation_aggregate = decision.get("confirmation_speedup")
        if confirmation_aggregate is not None:
            aggregate = min(aggregate, float(confirmation_aggregate))
        worst = float(decision["worst_case_regression"])
        confirmation_worst = decision.get("confirmation_worst_case_regression")
        if confirmation_worst is not None:
            worst = max(worst, float(confirmation_worst))
        vector_value = (
            decision.get("confirmation_case_speedups")
            or decision.get("per_case_speedups")
        )
        if not isinstance(vector_value, Mapping):
            return None
        vector = {str(key): float(value) for key, value in vector_value.items()}
    except (KeyError, TypeError, ValueError):
        return None
    rejected_for_regression = (
        promotion.get("reason") == "per_case_regression_exceeded"
        or (
            promotion.get("phase") == "confirmation"
            and decision.get("reason") == "confirmation_failed"
            and worst > 0.03
        )
    )
    if aggregate <= 1.0 or worst <= 0.03 or not rejected_for_regression:
        return None
    return aggregate, worst, vector


def nominate_exploratory_from_child(
    store: CampaignStore,
    history: HistoryStore,
    *,
    campaign_id: str,
    child_id: int,
    artifact_id: str,
) -> dict[str, Any]:
    """Nominate a correct, beneficial, regression-rejected Pareto candidate."""

    campaign, child = _child_for_campaign(store, campaign_id, child_id)
    ArtifactId.parse(artifact_id)
    records = history.list_experiments_for_namespace(
        campaign["namespace_id"],
        artifact_id=artifact_id,
        backend="c500",
        suite="full",
        status="SUCCESS",
        newest_first=True,
    )
    selected: tuple[ExperimentRecord, ExperimentIdentity] | None = None
    for record in records:
        try:
            identity = _identity_for_child(
                store,
                record,
                campaign=campaign,
                child=child,
                artifact_id=artifact_id,
            )
        except ValueError:
            continue
        if _exploratory_metrics(record) is not None:
            selected = record, identity
            break
    if selected is None:
        raise ValueError("candidate does not satisfy exploratory OJ eligibility")
    record, identity = selected
    aggregate, worst, vector = _exploratory_metrics(record) or (0.0, 0.0, {})
    for other in history.list_experiments_for_namespace(
        campaign["namespace_id"], backend="c500", suite="full", status="SUCCESS"
    ):
        if other.artifact_id == artifact_id:
            continue
        try:
            other_identity = ExperimentIdentity.from_value(other.identity)
        except (TypeError, ValueError):
            continue
        if (
            str(other_identity.baseline.artifact_id)
            != str(identity.baseline.artifact_id)
        ):
            continue
        other_metrics = _exploratory_metrics(other)
        if other_metrics is None:
            continue
        other_vector = other_metrics[2]
        if set(other_vector) == set(vector) and all(
            other_vector[name] >= vector[name] for name in vector
        ) and any(other_vector[name] > vector[name] for name in vector):
            raise ValueError("candidate is dominated and not on the Pareto frontier")
    _require_artifact(history, artifact_id)
    local_metrics = {
        "schema_version": 1,
        "eligibility": "EXPLORATORY_SUBMISSION",
        "aggregate_speedup": aggregate,
        "worst_case_regression": worst,
        "pareto_frontier": True,
        "evidence": _record_summary(record),
    }
    manifest_digest = canonical_sha256(
        {
            "campaign_id": campaign_id,
            "namespace_id": campaign["namespace_id"],
            "artifact_id": artifact_id,
            "baseline_revision_id": child["baseline_revision_id"],
            "local_metrics": local_metrics,
        }
    )
    return store.create_oj_nomination(
        campaign_id,
        nomination_type="EXPLORATORY_SUBMISSION",
        artifact_id=artifact_id,
        bundle_object_id=artifact_id,
        manifest_digest=manifest_digest,
        local_metrics=local_metrics,
        baseline_revision_id=child["baseline_revision_id"],
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strict_json_object(data: bytes) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("candidate bundle object must be strict UTF-8 JSON") from exc

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"candidate bundle contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=object_pairs)
    except json.JSONDecodeError as exc:
        raise ValueError("candidate bundle object is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("candidate bundle object must be a JSON object")
    return value


def _candidate_bundle(object_id: str, content: bytes) -> CandidateBundle:
    if not isinstance(content, bytes):
        raise TypeError("artifact_reader must return bytes")
    if len(content) > MAX_STORED_BUNDLE_BYTES:
        raise ValueError("candidate bundle object exceeds the export hard limit")
    artifact_id = ArtifactId.parse(object_id)
    if artifact_id.is_legacy_source:
        if _sha256(content) != artifact_id.digest:
            raise ValueError("legacy source object does not match bundle_object_id")
        try:
            source = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("legacy source object must be strict UTF-8") from exc
        return CandidateBundle(
            SOURCE_BUNDLE_V1,
            "kernel.py",
            (CandidateFile("kernel.py", "text/x-python", source),),
        )

    bundle = CandidateBundle.from_value(_strict_json_object(content))
    if str(bundle.artifact_id) != object_id:
        raise ValueError("candidate bundle manifest does not match bundle_object_id")
    return bundle


def _package_name(value: str) -> str:
    if (
        not isinstance(value, str)
        or not _PACKAGE_NAME_RE.fullmatch(value)
        or value in {".", ".."}
    ):
        raise ValueError("package_name must be one safe ASCII path segment")
    return value


def _safe_candidate_paths(bundle: CandidateBundle) -> None:
    filesystem_names: set[str] = set()
    for item in bundle.files:
        path = item.path
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in path):
            raise ValueError("candidate file paths must not contain control characters")
        normalized = unicodedata.normalize("NFC", path).casefold()
        if normalized in filesystem_names:
            raise ValueError("candidate file paths collide on common filesystems")
        filesystem_names.add(normalized)


def _destination(root: Path, relative_path: str) -> Path:
    posix = PurePosixPath(relative_path)
    if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise ValueError("export file path is not a safe relative POSIX path")
    destination = root.joinpath(*posix.parts)
    try:
        destination.relative_to(root)
    except ValueError as exc:  # pragma: no cover - guarded by PurePosixPath
        raise ValueError("export file path escapes the package") from exc
    return destination


def _write_new_file(root: Path, relative_path: str, content: bytes) -> None:
    destination = _destination(root, relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with destination.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    destination.chmod(0o600)


def _fsync_directories(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    directories.append(root)
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
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


def _manual_instructions(entrypoint: str) -> bytes:
    return (
        "This is an offline export. Nothing in this package submits to an OJ.\n"
        "Review the manifest and SHA256SUMS, then submit the candidate manually.\n"
        f"Candidate entrypoint: candidate/{entrypoint}\n"
    ).encode("utf-8")


def _package_manifest(
    nomination: Mapping[str, Any], bundle: CandidateBundle
) -> dict[str, Any]:
    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "format": "oj-manual-submission-package-v1",
        "submission_policy": {
            "automatic_submission": False,
            "network_access": False,
            "operator_review_required": True,
        },
        "nomination": {
            "id": nomination["id"],
            "campaign_id": nomination["campaign_id"],
            "namespace_id": nomination["namespace_id"],
            "nomination_type": nomination["nomination_type"],
            "artifact_id": nomination["artifact_id"],
            "baseline_revision_id": nomination["baseline_revision_id"],
            "bundle_object_id": nomination["bundle_object_id"],
            "evidence_manifest_digest": nomination["manifest_digest"],
        },
        "candidate": bundle.manifest,
        "local_metrics": nomination["local_metrics"],
    }


def export_oj_submission_package(
    store: CampaignStore,
    nomination_id: str,
    *,
    destination_root: str | Path,
    artifact_reader: ArtifactReader,
    package_name: str | None = None,
) -> dict[str, Any]:
    """Atomically publish one immutable package and mark it exported.

    ``artifact_reader`` is intentionally injected.  A caller may use
    ``HistoryStore.read_candidate_artifact``; the exporter itself never gains
    database credentials beyond CampaignStore and never performs network I/O.
    """

    nomination = store.get_oj_nomination(nomination_id)
    if nomination["status"] not in {"NOMINATED", "EXPORTED"}:
        raise ValueError("OJ package can only be exported before feedback")
    selected_name = _package_name(package_name or str(nomination["id"]))

    root = Path(destination_root)
    if root.is_symlink():
        raise ValueError("destination_root must not be a symbolic link")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("destination_root must be a directory")
    target = root / selected_name
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite OJ package: {target}")

    if nomination["artifact_id"] != nomination["bundle_object_id"]:
        raise ValueError(
            "OJ nomination artifact and exported object identities differ"
        )
    raw_object = artifact_reader(str(nomination["bundle_object_id"]))
    bundle = _candidate_bundle(str(nomination["bundle_object_id"]), raw_object)
    _safe_candidate_paths(bundle)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{selected_name}.", suffix=".tmp", dir=root)
    )
    temporary.chmod(0o700)
    published = False
    try:
        payloads: dict[str, bytes] = {
            f"candidate/{item.path}": item.content_bytes for item in bundle.files
        }
        payloads["candidate.bundle.json"] = bundle.bundle_bytes
        payloads["SUBMIT_MANUALLY.txt"] = _manual_instructions(bundle.entrypoint)
        manifest = _package_manifest(nomination, bundle)
        manifest_bytes = canonical_json_bytes(manifest)
        payloads["manifest.json"] = manifest_bytes

        for relative_path, content in sorted(payloads.items()):
            _write_new_file(temporary, relative_path, content)
        checksum_lines = [
            f"{_sha256(content)}  {relative_path}\n"
            for relative_path, content in sorted(payloads.items())
        ]
        checksums = "".join(checksum_lines).encode("utf-8")
        _write_new_file(temporary, "SHA256SUMS", checksums)
        _fsync_directories(temporary)

        exported = store.mark_oj_exported(nomination_id)
        if os.path.lexists(target):
            raise FileExistsError(f"refusing to overwrite OJ package: {target}")
        os.rename(temporary, target)
        published = True
        _fsync_directory(root)
    finally:
        if not published:
            shutil.rmtree(temporary, ignore_errors=True)

    return {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "status": "EXPORTED",
        "submission_mode": "MANUAL_ONLY",
        "nomination": exported,
        "path": str(target),
        "manifest_digest": canonical_sha256(manifest),
        "sha256sums_sha256": "sha256:" + _sha256(checksums),
        "files": sorted([*payloads, "SHA256SUMS"]),
    }


__all__ = [
    "MAX_STORED_BUNDLE_BYTES",
    "PACKAGE_SCHEMA_VERSION",
    "export_oj_submission_package",
    "nominate_exploratory_from_child",
    "nominate_primary_from_child",
]
