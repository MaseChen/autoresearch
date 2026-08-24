"""Strict Mac-local Console configuration and filesystem conventions."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any


CONFIG_SCHEMA_VERSION = 1
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "kernel-research-console" / "config.json"
DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "kernel-research-console"
_FIELDS = frozenset(
    {
        "schema_version",
        "ssh_binary",
        "ssh_target",
        "remote_python",
        "remote_admin_manifest",
        "data_dir",
        "poll_interval_ms",
    }
)
_SSH_TARGET_RE = re.compile(
    r"^(?:[A-Za-z0-9][A-Za-z0-9._-]{0,63}@)?[A-Za-z0-9][A-Za-z0-9._-]{0,252}$"
)


def _reject_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"console config repeats key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"console config contains invalid constant {value}")


def _reject_symlink_chain(path: Path) -> None:
    current = path
    while True:
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            metadata = None
        if metadata is not None and stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"console path may not traverse a symlink: {current}")
        if current.parent == current:
            return
        current = current.parent


def _local_path(value: object, field_name: str, *, existing_file: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or path != Path(os.path.normpath(value)):
        raise ValueError(f"{field_name} must be a canonical absolute path")
    _reject_symlink_chain(path)
    if existing_file:
        try:
            resolved = path.resolve(strict=True)
            metadata = path.stat(follow_symlinks=False)
        except OSError as exc:
            raise ValueError(f"{field_name} is unavailable") from exc
        if resolved != path or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{field_name} must be a canonical regular file")
    return path


def _remote_path(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise ValueError(f"{field_name} must be an absolute POSIX path")
    path = PurePosixPath(value)
    if not path.is_absolute() or path.as_posix() != value or any(
        part in {"", ".", ".."} for part in value.split("/")[1:]
    ):
        raise ValueError(f"{field_name} must be a canonical absolute POSIX path")
    if not re.fullmatch(r"/[A-Za-z0-9._/-]{1,4095}", value):
        raise ValueError(f"{field_name} contains unsupported characters")
    return value


@dataclass(frozen=True)
class GatewayConfig:
    ssh_binary: Path
    ssh_target: str
    remote_python: str
    remote_admin_manifest: str
    data_dir: Path
    poll_interval_ms: int = 2000

    def __post_init__(self) -> None:
        _local_path(str(self.ssh_binary), "ssh_binary", existing_file=True)
        if not isinstance(self.ssh_target, str) or not _SSH_TARGET_RE.fullmatch(
            self.ssh_target
        ):
            raise ValueError("ssh_target must be one fixed SSH host or user@host")
        _remote_path(self.remote_python, "remote_python")
        _remote_path(self.remote_admin_manifest, "remote_admin_manifest")
        _local_path(str(self.data_dir), "data_dir")
        if (
            type(self.poll_interval_ms) is not int
            or not 1000 <= self.poll_interval_ms <= 30000
        ):
            raise ValueError("poll_interval_ms must be between 1000 and 30000")

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "GatewayConfig":
        path = _local_path(str(path), "console config", existing_file=True)
        metadata = path.stat(follow_symlinks=False)
        if stat.S_IMODE(metadata.st_mode) != 0o600:
            raise ValueError("console config mode must be exactly 0600")
        if metadata.st_size <= 0 or metadata.st_size > 64 * 1024:
            raise ValueError("console config exceeds its fixed size bound")
        raw = path.read_bytes()
        if len(raw) != metadata.st_size:
            raise ValueError("console config changed while being read")
        try:
            value = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_reject_pairs,
                parse_constant=_reject_constant,
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("console config is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("console config must be a JSON object")
        unknown = sorted(set(value) - _FIELDS)
        missing = sorted(_FIELDS - set(value))
        if unknown or missing:
            raise ValueError(
                "console config fields mismatch: "
                f"unknown={unknown!r} missing={missing!r}"
            )
        if value["schema_version"] != CONFIG_SCHEMA_VERSION or type(
            value["schema_version"]
        ) is not int:
            raise ValueError("console config schema_version must be 1")
        return cls(
            ssh_binary=_local_path(value["ssh_binary"], "ssh_binary", existing_file=True),
            ssh_target=value["ssh_target"],
            remote_python=_remote_path(value["remote_python"], "remote_python"),
            remote_admin_manifest=_remote_path(
                value["remote_admin_manifest"], "remote_admin_manifest"
            ),
            data_dir=_local_path(value["data_dir"], "data_dir"),
            poll_interval_ms=value["poll_interval_ms"],
        )

    def ensure_data_dir(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        _reject_symlink_chain(self.data_dir)
        resolved = self.data_dir.resolve(strict=True)
        metadata = self.data_dir.stat(follow_symlinks=False)
        if resolved != self.data_dir or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("console data_dir must be a canonical directory")
        if metadata.st_uid != os.getuid():
            raise ValueError("console data_dir must be owned by the current user")
        os.chmod(self.data_dir, 0o700)
        return self.data_dir


__all__ = ["CONFIG_SCHEMA_VERSION", "DEFAULT_CONFIG_PATH", "DEFAULT_DATA_DIR", "GatewayConfig"]
