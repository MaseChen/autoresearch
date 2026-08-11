"""V2 candidate bundles and the strict ProposalV2 wire contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
import re
from typing import Any

from ..constants import (
    PROPOSAL_HYPOTHESIS_HARD_LIMIT,
    PROPOSAL_RATIONALE_HARD_LIMIT,
)
from .artifacts import ArtifactId
from .canonical import canonical_json_bytes, require_sha256_digest, sha256_hex


SOURCE_BUNDLE_V1 = "source_bundle_v1"
HARD_MAX_BUNDLE_FILES = 64
HARD_MAX_FILE_BYTES = 1024 * 1024
HARD_MAX_TOTAL_BYTES = 4 * 1024 * 1024
_MEDIA_TYPE_RE = re.compile(
    r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$"
)

_BUNDLE_FIELDS = frozenset({"format", "entrypoint", "files"})
_FILE_FIELDS = frozenset({"path", "media_type", "content"})
_PROPOSAL_V2_FIELDS = frozenset(
    {
        "schema_version",
        "proposal_context_id",
        "parent_artifact_id",
        "hypothesis",
        "rationale",
        "candidate",
    }
)


def _strict_object(
    value: object, *, fields: frozenset[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {', '.join(unknown)}")
    missing = sorted(fields - set(value))
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")
    return value


def _canonical_bundle_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative POSIX path")
    if "\x00" in value or "\\" in value:
        raise ValueError(f"{field} must be a canonical relative POSIX path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"{field} must be relative")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"{field} must not contain '.', '..' or empty segments")
    if posix.as_posix() != value:
        raise ValueError(f"{field} must already be normalized")
    return value


def _positive_limit(value: object, *, field: str, maximum: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    if value > maximum:
        raise ValueError(f"{field} exceeds the system hard limit")
    return value


def _extension_set(value: object) -> frozenset[str] | None:
    if value is None:
        return None
    try:
        result = frozenset(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("allowed_extensions must be an iterable of extensions") from exc
    if not result or any(
        not isinstance(item, str)
        or not re.fullmatch(r"\.[a-z0-9][a-z0-9.+_-]{0,15}", item)
        for item in result
    ):
        raise ValueError("allowed_extensions contains an invalid extension")
    return result


def _media_type_set(value: object) -> frozenset[str] | None:
    if value is None:
        return None
    try:
        result = frozenset(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("allowed_media_types must be an iterable") from exc
    if not result or any(
        not isinstance(item, str) or not _MEDIA_TYPE_RE.fullmatch(item)
        for item in result
    ):
        raise ValueError("allowed_media_types contains an invalid media type")
    return result


@dataclass(frozen=True)
class BundleLimits:
    """Language-owned limits bounded again by system hard maxima."""

    max_files: int = HARD_MAX_BUNDLE_FILES
    max_file_bytes: int = HARD_MAX_FILE_BYTES
    max_total_bytes: int = HARD_MAX_TOTAL_BYTES
    allowed_extensions: frozenset[str] | None = None
    allowed_media_types: frozenset[str] | None = None
    required_entrypoint: str | None = None

    def __post_init__(self) -> None:
        _positive_limit(
            self.max_files, field="max_files", maximum=HARD_MAX_BUNDLE_FILES
        )
        _positive_limit(
            self.max_file_bytes,
            field="max_file_bytes",
            maximum=HARD_MAX_FILE_BYTES,
        )
        _positive_limit(
            self.max_total_bytes,
            field="max_total_bytes",
            maximum=HARD_MAX_TOTAL_BYTES,
        )
        object.__setattr__(
            self,
            "allowed_extensions",
            _extension_set(self.allowed_extensions),
        )
        object.__setattr__(
            self,
            "allowed_media_types",
            _media_type_set(self.allowed_media_types),
        )
        if self.required_entrypoint is not None:
            object.__setattr__(
                self,
                "required_entrypoint",
                _canonical_bundle_path(
                    self.required_entrypoint, field="required_entrypoint"
                ),
            )


SYSTEM_BUNDLE_LIMITS = BundleLimits()
TRITON_PYTHON_BUNDLE_LIMITS = BundleLimits(
    max_files=1,
    max_file_bytes=256 * 1024,
    max_total_bytes=256 * 1024,
    allowed_extensions=frozenset({".py"}),
    allowed_media_types=frozenset({"text/x-python"}),
    required_entrypoint="kernel.py",
)


@dataclass(frozen=True, order=True)
class CandidateFile:
    path: str
    media_type: str
    content: str

    def __post_init__(self) -> None:
        _canonical_bundle_path(self.path, field="candidate file path")
        if not isinstance(self.media_type, str) or not _MEDIA_TYPE_RE.fullmatch(
            self.media_type
        ):
            raise ValueError("candidate file media_type is invalid")
        if not isinstance(self.content, str):
            raise ValueError("candidate file content must be a string")
        if "\x00" in self.content:
            raise ValueError("candidate file content must not contain NUL")
        try:
            self.content.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("candidate file content contains invalid Unicode") from exc

    @classmethod
    def from_value(cls, value: object) -> "CandidateFile":
        obj = _strict_object(
            value, fields=_FILE_FIELDS, name="candidate file"
        )
        return cls(
            path=obj["path"],
            media_type=obj["media_type"],
            content=obj["content"],
        )

    @property
    def content_bytes(self) -> bytes:
        return self.content.encode("utf-8")

    def to_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "path": self.path,
            "media_type": self.media_type,
        }
        if include_content:
            result["content"] = self.content
        return result


@dataclass(frozen=True)
class CandidateBundle:
    format: str
    entrypoint: str
    files: tuple[CandidateFile, ...]

    def __post_init__(self) -> None:
        if self.format != SOURCE_BUNDLE_V1:
            raise ValueError(f"candidate format must be {SOURCE_BUNDLE_V1}")
        entrypoint = _canonical_bundle_path(
            self.entrypoint, field="candidate entrypoint"
        )
        try:
            original_files = tuple(self.files)
        except TypeError as exc:
            raise ValueError("candidate files must be an iterable") from exc
        if not original_files:
            raise ValueError("candidate files must not be empty")
        if any(not isinstance(item, CandidateFile) for item in original_files):
            raise ValueError("candidate files must contain CandidateFile values")
        paths = [item.path for item in original_files]
        if len(paths) != len(set(paths)):
            raise ValueError("candidate files contain duplicate paths")
        files = tuple(sorted(original_files, key=lambda item: item.path))
        if entrypoint not in {item.path for item in files}:
            raise ValueError("candidate entrypoint must identify a bundle file")
        entrypoint_file = next(item for item in files if item.path == entrypoint)
        if not entrypoint_file.content.strip():
            raise ValueError("candidate entrypoint content must be non-empty")
        object.__setattr__(self, "entrypoint", entrypoint)
        object.__setattr__(self, "files", files)
        self.validate(SYSTEM_BUNDLE_LIMITS)

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        limits: BundleLimits = SYSTEM_BUNDLE_LIMITS,
    ) -> "CandidateBundle":
        if not isinstance(limits, BundleLimits):
            raise TypeError("limits must be BundleLimits")
        obj = _strict_object(value, fields=_BUNDLE_FIELDS, name="candidate")
        raw_files = obj["files"]
        if not isinstance(raw_files, list):
            raise ValueError("candidate files must be a JSON array")
        bundle = cls(
            format=obj["format"],
            entrypoint=obj["entrypoint"],
            files=tuple(CandidateFile.from_value(item) for item in raw_files),
        )
        return bundle.validate(limits)

    @classmethod
    def single_file(
        cls,
        *,
        content: str,
        path: str = "kernel.py",
        media_type: str = "text/x-python",
        limits: BundleLimits = TRITON_PYTHON_BUNDLE_LIMITS,
    ) -> "CandidateBundle":
        return cls(
            SOURCE_BUNDLE_V1,
            path,
            (CandidateFile(path, media_type, content),),
        ).validate(limits)

    def validate(self, limits: BundleLimits) -> "CandidateBundle":
        if not isinstance(limits, BundleLimits):
            raise TypeError("limits must be BundleLimits")
        if len(self.files) > limits.max_files:
            raise ValueError("candidate bundle exceeds max_files")
        if (
            limits.required_entrypoint is not None
            and self.entrypoint != limits.required_entrypoint
        ):
            raise ValueError("candidate entrypoint is not allowed by language profile")
        total_bytes = 0
        for file in self.files:
            size = len(file.content_bytes)
            if size > limits.max_file_bytes:
                raise ValueError(
                    f"candidate file {file.path!r} exceeds max_file_bytes"
                )
            total_bytes += size
            if (
                limits.allowed_extensions is not None
                and PurePosixPath(file.path).suffix
                not in limits.allowed_extensions
            ):
                raise ValueError(
                    f"candidate file {file.path!r} has a disallowed extension"
                )
            if (
                limits.allowed_media_types is not None
                and file.media_type not in limits.allowed_media_types
            ):
                raise ValueError(
                    f"candidate file {file.path!r} has a disallowed media_type"
                )
        if total_bytes > limits.max_total_bytes:
            raise ValueError("candidate bundle exceeds max_total_bytes")
        return self

    @property
    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "candidate-bundle-manifest-v1",
            "format": self.format,
            "entrypoint": self.entrypoint,
            "files": [
                {
                    "path": file.path,
                    "media_type": file.media_type,
                    "size_bytes": len(file.content_bytes),
                    "sha256": f"sha256:{sha256_hex(file.content_bytes)}",
                }
                for file in self.files
            ],
        }

    @property
    def artifact_id(self) -> ArtifactId:
        return ArtifactId.for_bundle_manifest(self.manifest_bytes)

    @property
    def manifest_bytes(self) -> bytes:
        return canonical_json_bytes(self.manifest)

    @property
    def bundle_bytes(self) -> bytes:
        """Canonical storage encoding; source text itself is not normalized."""

        return canonical_json_bytes(self.to_dict(include_content=True))

    def to_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        return {
            "format": self.format,
            "entrypoint": self.entrypoint,
            "files": [
                item.to_dict(include_content=include_content)
                for item in self.files
            ],
        }


def _proposal_text(
    value: object,
    *,
    field: str,
    hard_limit: int,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if len(value) > hard_limit:
        raise ValueError(
            f"{field} exceeds {hard_limit} decoded Unicode characters"
        )
    return value


@dataclass(frozen=True)
class ProposalV2:
    schema_version: int
    proposal_context_id: str
    parent_artifact_id: ArtifactId
    hypothesis: str
    rationale: str
    candidate: CandidateBundle

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("proposal schema_version must be 2")
        require_sha256_digest(
            self.proposal_context_id, field="proposal_context_id"
        )
        if not isinstance(self.parent_artifact_id, ArtifactId):
            raise ValueError("parent_artifact_id must be an ArtifactId")
        _proposal_text(
            self.hypothesis,
            field="hypothesis",
            hard_limit=PROPOSAL_HYPOTHESIS_HARD_LIMIT,
        )
        _proposal_text(
            self.rationale,
            field="rationale",
            hard_limit=PROPOSAL_RATIONALE_HARD_LIMIT,
        )
        if not isinstance(self.candidate, CandidateBundle):
            raise ValueError("candidate must be a CandidateBundle")

    @classmethod
    def create(
        cls,
        *,
        proposal_context_id: str,
        parent_artifact_id: ArtifactId | str,
        hypothesis: str,
        rationale: str,
        candidate: CandidateBundle,
        limits: BundleLimits = SYSTEM_BUNDLE_LIMITS,
    ) -> "ProposalV2":
        return cls(
            schema_version=2,
            proposal_context_id=require_sha256_digest(
                proposal_context_id, field="proposal_context_id"
            ),
            parent_artifact_id=ArtifactId.parse(parent_artifact_id),
            hypothesis=hypothesis,
            rationale=rationale,
            candidate=candidate.validate(limits),
        )

    @classmethod
    def from_value(
        cls,
        value: object,
        *,
        expected_proposal_context_id: str,
        expected_parent_artifact_id: ArtifactId | str,
        limits: BundleLimits = SYSTEM_BUNDLE_LIMITS,
    ) -> "ProposalV2":
        obj = _strict_object(
            value, fields=_PROPOSAL_V2_FIELDS, name="proposal"
        )
        if type(obj["schema_version"]) is not int or obj["schema_version"] != 2:
            raise ValueError("proposal schema_version must be 2")
        context_id = require_sha256_digest(
            obj["proposal_context_id"], field="proposal_context_id"
        )
        expected_context = require_sha256_digest(
            expected_proposal_context_id,
            field="expected_proposal_context_id",
        )
        if context_id != expected_context:
            raise ValueError("proposal_context_id does not match request")
        parent = ArtifactId.parse(obj["parent_artifact_id"])
        expected_parent = ArtifactId.parse(expected_parent_artifact_id)
        if parent != expected_parent:
            raise ValueError("proposal parent_artifact_id does not match accepted")
        return cls.create(
            proposal_context_id=context_id,
            parent_artifact_id=parent,
            hypothesis=_proposal_text(
                obj["hypothesis"],
                field="hypothesis",
                hard_limit=PROPOSAL_HYPOTHESIS_HARD_LIMIT,
            ),
            rationale=_proposal_text(
                obj["rationale"],
                field="rationale",
                hard_limit=PROPOSAL_RATIONALE_HARD_LIMIT,
            ),
            candidate=CandidateBundle.from_value(obj["candidate"], limits=limits),
            limits=limits,
        )

    @property
    def candidate_artifact_id(self) -> ArtifactId:
        return self.candidate.artifact_id

    def to_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "proposal_context_id": self.proposal_context_id,
            "parent_artifact_id": str(self.parent_artifact_id),
            "hypothesis": self.hypothesis,
            "rationale": self.rationale,
            "candidate": self.candidate.to_dict(include_content=include_content),
        }


__all__ = [
    "HARD_MAX_BUNDLE_FILES",
    "HARD_MAX_FILE_BYTES",
    "HARD_MAX_TOTAL_BYTES",
    "SOURCE_BUNDLE_V1",
    "SYSTEM_BUNDLE_LIMITS",
    "TRITON_PYTHON_BUNDLE_LIMITS",
    "BundleLimits",
    "CandidateBundle",
    "CandidateFile",
    "ProposalV2",
]
