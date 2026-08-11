"""Immutable deployment-baseline pin contract shared by admin and Controller.

The administrator is the only writer.  The Controller treats this file as a
signed-by-content snapshot: every field participates in ``evidence_digest``
and is checked again against Git, History, and the resolved runtime before a
Run starts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import stat
from typing import Any

from ..platform.canonical import canonical_sha256, require_sha256_digest
from ..platform.identity import BaselineRef, ExecutionEnvironmentDigest


DEPLOYMENT_BASELINE_FILENAME = "deployment-baseline.json"
DEPLOYMENT_BASELINE_SCHEMA_VERSION = 1
MAX_DEPLOYMENT_BASELINE_BYTES = 64 * 1024
_GIT_OBJECT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_CANDIDATE_RE = re.compile(r"^[0-9a-f]{64}$")
_FIELDS = frozenset(
    {
        "schema_version",
        "namespace_id",
        "baseline_ref",
        "candidate_hash",
        "git_commit",
        "primary_experiment_uid",
        "confirmation_experiment_uid",
        "confirmation_experiment_id",
        "parent_baseline_ref",
        "execution_environment",
        "evidence_digest",
    }
)


def deployment_runtime_root(
    *,
    state_dir: Path,
    controller_dir: Path,
    checkpoint_dir: Path,
) -> Path:
    """Derive one runtime root from the three durable bounded contexts."""

    supplied = {
        "state_dir": Path(state_dir),
        "controller_dir": Path(controller_dir),
        "checkpoint_dir": Path(checkpoint_dir),
    }
    directories: dict[str, Path] = {}
    for name, path in supplied.items():
        if not path.is_absolute():
            raise ValueError(f"{name} must be an absolute path")
        try:
            directories[name] = path.resolve(strict=False)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"{name} cannot be resolved canonically") from exc
    roots = {path.parent for path in directories.values()}
    if len(roots) != 1:
        raise ValueError(
            "state_dir, controller_dir, and checkpoint_dir must share one "
            "direct runtime root"
        )
    root = next(iter(roots))
    if root == Path(root.anchor):
        raise ValueError("deployment runtime root must not be the filesystem root")
    return root


def _uid(value: object, field: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or "\x00" in value
    ):
        raise ValueError(f"{field} must be a bounded non-empty string")
    return value


def _reject_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"deployment baseline pin repeats JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(
        f"deployment baseline pin contains invalid JSON constant {value}"
    )


def _reject_symlink_chain(path: Path) -> None:
    current = path
    while True:
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            mode = None
        if mode is not None and stat.S_ISLNK(mode):
            raise ValueError(
                f"deployment baseline pin path may not traverse a symlink: {current}"
            )
        if current.parent == current:
            return
        current = current.parent


@dataclass(frozen=True)
class DeploymentBaselinePin:
    """One manually adopted baseline and the evidence that authorized it."""

    namespace_id: str
    baseline_ref: BaselineRef
    candidate_hash: str
    git_commit: str
    primary_experiment_uid: str | None
    confirmation_experiment_uid: str
    confirmation_experiment_id: int
    parent_baseline_ref: BaselineRef | None
    execution_environment: ExecutionEnvironmentDigest
    evidence_digest: str

    def __post_init__(self) -> None:
        require_sha256_digest(self.namespace_id, field="deployment namespace_id")
        if not isinstance(self.baseline_ref, BaselineRef):
            raise ValueError("deployment baseline_ref must be a BaselineRef")
        if self.baseline_ref.source != "deployment":
            raise ValueError("deployment baseline_ref source must be deployment")
        if self.baseline_ref.namespace_id != self.namespace_id:
            raise ValueError("deployment baseline_ref belongs to another namespace")
        if not _CANDIDATE_RE.fullmatch(self.candidate_hash):
            raise ValueError(
                "deployment candidate_hash must be 64 lowercase hex digits"
            )
        if not _GIT_OBJECT_RE.fullmatch(self.git_commit):
            raise ValueError("deployment git_commit must be a full Git object ID")
        _uid(
            self.primary_experiment_uid,
            "primary_experiment_uid",
            optional=True,
        )
        _uid(self.confirmation_experiment_uid, "confirmation_experiment_uid")
        if (
            self.primary_experiment_uid is not None
            and self.primary_experiment_uid == self.confirmation_experiment_uid
        ):
            raise ValueError("deployment primary and confirmation must differ")
        if (
            type(self.confirmation_experiment_id) is not int
            or self.confirmation_experiment_id <= 0
        ):
            raise ValueError("confirmation_experiment_id must be a positive integer")
        if self.parent_baseline_ref is not None:
            if not isinstance(self.parent_baseline_ref, BaselineRef):
                raise ValueError("parent_baseline_ref must be a BaselineRef or null")
            if self.parent_baseline_ref.namespace_id != self.namespace_id:
                raise ValueError("parent baseline belongs to another namespace")
        if not isinstance(self.execution_environment, ExecutionEnvironmentDigest):
            raise ValueError(
                "deployment execution_environment must be an "
                "ExecutionEnvironmentDigest"
            )
        if self.baseline_ref.execution_environment != self.execution_environment:
            raise ValueError(
                "deployment baseline and evidence execution environments differ"
            )
        require_sha256_digest(self.evidence_digest, field="evidence_digest")
        if self.evidence_digest != canonical_sha256(self.evidence_material()):
            raise ValueError("deployment baseline evidence_digest mismatch")

    def evidence_material(self) -> dict[str, Any]:
        return {
            "schema_version": DEPLOYMENT_BASELINE_SCHEMA_VERSION,
            "namespace_id": self.namespace_id,
            "baseline_ref": self.baseline_ref.to_dict(),
            "candidate_hash": self.candidate_hash,
            "git_commit": self.git_commit,
            "primary_experiment_uid": self.primary_experiment_uid,
            "confirmation_experiment_uid": self.confirmation_experiment_uid,
            "confirmation_experiment_id": self.confirmation_experiment_id,
            "parent_baseline_ref": (
                None
                if self.parent_baseline_ref is None
                else self.parent_baseline_ref.to_dict()
            ),
            "execution_environment": self.execution_environment.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        material = self.evidence_material()
        material["evidence_digest"] = self.evidence_digest
        return material

    @classmethod
    def create(
        cls,
        *,
        namespace_id: str,
        baseline_ref: BaselineRef,
        candidate_hash: str,
        git_commit: str,
        primary_experiment_uid: str | None,
        confirmation_experiment_uid: str,
        confirmation_experiment_id: int,
        parent_baseline_ref: BaselineRef | None,
        execution_environment: ExecutionEnvironmentDigest,
    ) -> "DeploymentBaselinePin":
        material = {
            "schema_version": DEPLOYMENT_BASELINE_SCHEMA_VERSION,
            "namespace_id": namespace_id,
            "baseline_ref": baseline_ref.to_dict(),
            "candidate_hash": candidate_hash,
            "git_commit": git_commit,
            "primary_experiment_uid": primary_experiment_uid,
            "confirmation_experiment_uid": confirmation_experiment_uid,
            "confirmation_experiment_id": confirmation_experiment_id,
            "parent_baseline_ref": (
                None if parent_baseline_ref is None else parent_baseline_ref.to_dict()
            ),
            "execution_environment": execution_environment.to_dict(),
        }
        return cls(
            namespace_id=namespace_id,
            baseline_ref=baseline_ref,
            candidate_hash=candidate_hash,
            git_commit=git_commit,
            primary_experiment_uid=primary_experiment_uid,
            confirmation_experiment_uid=confirmation_experiment_uid,
            confirmation_experiment_id=confirmation_experiment_id,
            parent_baseline_ref=parent_baseline_ref,
            execution_environment=execution_environment,
            evidence_digest=canonical_sha256(material),
        )

    @classmethod
    def from_value(cls, value: object) -> "DeploymentBaselinePin":
        if not isinstance(value, dict):
            raise ValueError("deployment baseline pin must be a JSON object")
        unknown = sorted(set(value) - _FIELDS)
        missing = sorted(_FIELDS - set(value))
        if unknown:
            raise ValueError(
                "deployment baseline pin contains unknown fields: "
                + ", ".join(unknown)
            )
        if missing:
            raise ValueError(
                "deployment baseline pin is missing fields: " + ", ".join(missing)
            )
        if value["schema_version"] != DEPLOYMENT_BASELINE_SCHEMA_VERSION or type(
            value["schema_version"]
        ) is not int:
            raise ValueError("deployment baseline pin schema_version must be 1")
        parent_value = value["parent_baseline_ref"]
        return cls(
            namespace_id=value["namespace_id"],
            baseline_ref=BaselineRef.from_value(value["baseline_ref"]),
            candidate_hash=value["candidate_hash"],
            git_commit=value["git_commit"],
            primary_experiment_uid=_uid(
                value["primary_experiment_uid"],
                "primary_experiment_uid",
                optional=True,
            ),
            confirmation_experiment_uid=_uid(
                value["confirmation_experiment_uid"],
                "confirmation_experiment_uid",
            ),
            confirmation_experiment_id=value["confirmation_experiment_id"],
            parent_baseline_ref=(
                None
                if parent_value is None
                else BaselineRef.from_value(parent_value)
            ),
            execution_environment=ExecutionEnvironmentDigest.from_value(
                value["execution_environment"]
            ),
            evidence_digest=value["evidence_digest"],
        )

    @classmethod
    def load(cls, path: Path) -> "DeploymentBaselinePin":
        path = Path(path)
        _reject_symlink_chain(path)
        try:
            metadata = path.stat()
        except OSError as exc:
            raise ValueError(f"could not stat deployment baseline pin: {exc}") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                "deployment baseline pin must be a regular non-symlink file"
            )
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ValueError("deployment baseline pin mode must be exactly 0600")
        if (
            metadata.st_size <= 0
            or metadata.st_size > MAX_DEPLOYMENT_BASELINE_BYTES
        ):
            raise ValueError("deployment baseline pin exceeds its fixed size bound")
        try:
            raw = path.read_bytes()
            if len(raw) != metadata.st_size:
                raise ValueError("deployment baseline pin changed while being read")
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_object,
                parse_constant=_reject_json_constant,
            )
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read deployment baseline pin: {exc}") from exc
        return cls.from_value(value)


__all__ = [
    "DEPLOYMENT_BASELINE_FILENAME",
    "DEPLOYMENT_BASELINE_SCHEMA_VERSION",
    "MAX_DEPLOYMENT_BASELINE_BYTES",
    "DeploymentBaselinePin",
    "deployment_runtime_root",
]
