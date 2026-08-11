"""Tagged content identifiers for legacy sources and V2 bundles."""

from __future__ import annotations

from dataclasses import dataclass

from .canonical import SHA256_HEX_RE, sha256_hex


SOURCE_SHA256_V1 = "source-sha256-v1"
BUNDLE_SHA256_V1 = "bundle-sha256-v1"
ARTIFACT_TAGS = frozenset({SOURCE_SHA256_V1, BUNDLE_SHA256_V1})


@dataclass(frozen=True, order=True)
class ArtifactId:
    """A content digest whose representation identifies its hash contract."""

    tag: str
    digest: str

    def __post_init__(self) -> None:
        if self.tag not in ARTIFACT_TAGS:
            allowed = ", ".join(sorted(ARTIFACT_TAGS))
            raise ValueError(f"artifact tag must be one of: {allowed}")
        if not isinstance(self.digest, str) or not SHA256_HEX_RE.fullmatch(
            self.digest
        ):
            raise ValueError("artifact digest must be 64 lowercase hex digits")

    @classmethod
    def parse(cls, value: object) -> "ArtifactId":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str) or value.count(":") != 1:
            raise ValueError("artifact_id must be a tagged artifact string")
        tag, digest = value.split(":", 1)
        return cls(tag=tag, digest=digest)

    @classmethod
    def source_sha256(cls, digest: str) -> "ArtifactId":
        """Tag one already-computed V1 source SHA-256."""

        return cls(SOURCE_SHA256_V1, digest)

    @classmethod
    def bundle_sha256(cls, digest: str) -> "ArtifactId":
        """Tag one already-computed canonical bundle SHA-256."""

        return cls(BUNDLE_SHA256_V1, digest)

    @classmethod
    def for_source_bytes(cls, source: bytes) -> "ArtifactId":
        return cls.source_sha256(sha256_hex(source))

    @classmethod
    def for_source_text(cls, source: str) -> "ArtifactId":
        if not isinstance(source, str):
            raise TypeError("source must be a string")
        try:
            encoded = source.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("source contains invalid Unicode") from exc
        return cls.for_source_bytes(encoded)

    @classmethod
    def for_bundle_manifest(cls, manifest_bytes: bytes) -> "ArtifactId":
        return cls.bundle_sha256(sha256_hex(manifest_bytes))

    @property
    def is_legacy_source(self) -> bool:
        return self.tag == SOURCE_SHA256_V1

    @property
    def value(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.tag}:{self.digest}"


def tag_legacy_source_hash(value: object) -> ArtifactId:
    """Map a V1 bare source SHA-256 into its explicit artifact domain."""

    if not isinstance(value, str):
        raise ValueError("legacy source hash must be 64 lowercase hex digits")
    return ArtifactId.source_sha256(value)


__all__ = [
    "ARTIFACT_TAGS",
    "BUNDLE_SHA256_V1",
    "SOURCE_SHA256_V1",
    "ArtifactId",
    "tag_legacy_source_hash",
]
