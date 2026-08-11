"""Deterministic JSON encoding used by V2 scientific identities.

The trusted host deliberately accepts only the JSON data model.  In
particular, mapping keys are never coerced to strings and non-finite numbers
are rejected before they can enter a digest.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any


SHA256_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _json_value(value: Any, *, path: str) -> Any:
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        # JSON has one zero value.  Avoid platform-specific negative-zero
        # spellings in identities while retaining ordinary float spelling.
        return 0 if value == 0 else value
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError(f"{path} contains invalid Unicode") from exc
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string object key")
            normalized[key] = _json_value(child, path=f"{path}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _json_value(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    raise TypeError(
        f"{path} contains unsupported JSON value {type(value).__name__}"
    )


def canonical_json_text(value: Any) -> str:
    """Return the platform's canonical, whitespace-free JSON spelling."""

    normalized = _json_value(value, path="value")
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return canonical JSON encoded as strict UTF-8 bytes."""

    return canonical_json_text(value).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return a tagged SHA-256 over :func:`canonical_json_bytes`."""

    digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return f"sha256:{digest}"


def sha256_hex(data: bytes) -> str:
    """Return an untagged lowercase SHA-256 for raw bytes."""

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    return hashlib.sha256(data).hexdigest()


def require_sha256_digest(value: object, *, field: str) -> str:
    """Validate and return one canonical ``sha256:<hex>`` digest."""

    if not isinstance(value, str) or not SHA256_DIGEST_RE.fullmatch(value):
        raise ValueError(f"{field} must be sha256: followed by 64 lowercase hex digits")
    return value


__all__ = [
    "SHA256_DIGEST_RE",
    "SHA256_HEX_RE",
    "canonical_json_bytes",
    "canonical_json_text",
    "canonical_sha256",
    "require_sha256_digest",
    "sha256_hex",
]
