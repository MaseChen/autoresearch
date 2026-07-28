"""Strict, standard-library-only controller models and configuration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from ..constants import (
    MAX_AUTORESEARCH_CANDIDATES,
    MAX_AUTORESEARCH_HOURS,
    MAX_CONSECUTIVE_FAILURES,
    EVALUATOR_MEMORY_LIMIT,
    MAX_EVALUATOR_CPUS,
    MAX_EVALUATOR_TIMEOUT_SEC,
    PROPOSER_MEMORY_LIMIT,
    MAX_PROPOSER_CPUS,
    MAX_PROPOSER_OUTPUT_BYTES,
    MAX_PROPOSER_TIMEOUT_SEC,
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_RE = re.compile(r"^.+@sha256:[0-9a-f]{64}$")
GPU1_DEVICES = (
    Path("/dev/mxcd"),
    Path("/dev/dri/card2"),
    Path("/dev/dri/renderD129"),
)
PROPOSAL_FIELDS = frozenset(
    {
        "schema_version",
        "parent_candidate_hash",
        "hypothesis",
        "rationale",
        "kernel_source",
    }
)
CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "repository_dir",
        "state_dir",
        "controller_dir",
        "checkpoint_dir",
        "docker_binary",
        "proposer_image",
        "evaluator_image",
        "deepseek_key_file",
        "gpu_devices",
        "evaluator_cache_dir",
        "expected_git_commit",
        "expected_kernel_hash",
        "opencode_model",
        "max_candidates",
        "max_hours",
        "max_consecutive_failures",
        "proposer_timeout_sec",
        "proposer_max_output_bytes",
        "evaluator_timeout_sec",
        "stop_after_promotion",
        "container_uid",
        "container_gid",
        "video_gid",
        "proposer_cpus",
        "proposer_memory",
        "evaluator_cpus",
        "evaluator_memory",
        "acknowledge_gpu_passthrough_risk",
    }
)


def _strict_object(value: Any, fields: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {', '.join(unknown)}")
    return value


def _canonical_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field} must be absolute")
    normalized = Path(os.path.normpath(value))
    if normalized != path or path.resolve(strict=False) != path:
        raise ValueError(f"{field} must be an absolute canonical path")
    return path


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a positive number")
    number = float(value)
    if number <= 0:
        raise ValueError(f"{field} must be a positive number")
    return number


def _positive_int(value: Any, field: str) -> int:
    number = _positive_number(value, field)
    if int(number) != number:
        raise ValueError(f"{field} must be an integer")
    return int(number)


@dataclass(frozen=True)
class ProposalV1:
    schema_version: int
    parent_candidate_hash: str
    hypothesis: str
    rationale: str
    kernel_source: str

    @property
    def candidate_hash(self) -> str:
        return hashlib.sha256(self.kernel_source.encode("utf-8")).hexdigest()

    @classmethod
    def from_value(
        cls, value: Any, *, expected_parent_hash: str
    ) -> "ProposalV1":
        obj = _strict_object(value, PROPOSAL_FIELDS, "proposal")
        missing = sorted(PROPOSAL_FIELDS - set(obj))
        if missing:
            raise ValueError(f"proposal is missing fields: {', '.join(missing)}")
        if type(obj["schema_version"]) is not int or obj["schema_version"] != 1:
            raise ValueError("proposal schema_version must be 1")
        parent = obj["parent_candidate_hash"]
        if not isinstance(parent, str) or not SHA256_RE.fullmatch(parent):
            raise ValueError("parent_candidate_hash must be 64 lowercase hex digits")
        if parent != expected_parent_hash:
            raise ValueError("proposal parent_candidate_hash does not match accepted")
        hypothesis = obj["hypothesis"]
        rationale = obj["rationale"]
        source = obj["kernel_source"]
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError("hypothesis must be a non-empty string")
        if len(hypothesis) > 1000:
            raise ValueError("hypothesis exceeds 1000 characters")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale must be a non-empty string")
        if len(rationale) > 8000:
            raise ValueError("rationale exceeds 8000 characters")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("kernel_source must be a non-empty string")
        if len(source.encode("utf-8")) > 256 * 1024:
            raise ValueError("kernel_source exceeds 256 KiB")
        return cls(1, parent, hypothesis, rationale, source)

    def to_dict(self, *, include_source: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": 1,
            "parent_candidate_hash": self.parent_candidate_hash,
            "hypothesis": self.hypothesis,
            "rationale": self.rationale,
            "candidate_hash": self.candidate_hash,
        }
        if include_source:
            result["kernel_source"] = self.kernel_source
        return result


@dataclass(frozen=True)
class ControllerConfig:
    repository_dir: Path
    state_dir: Path
    controller_dir: Path
    checkpoint_dir: Path
    docker_binary: Path
    proposer_image: str
    evaluator_image: str
    deepseek_key_file: Path
    gpu_devices: tuple[Path, ...]
    evaluator_cache_dir: Path
    expected_git_commit: str
    expected_kernel_hash: str
    opencode_model: str = "deepseek/deepseek-v4-pro"
    max_candidates: int = 5
    max_hours: float = 6.0
    max_consecutive_failures: int = 3
    proposer_timeout_sec: float = 1200.0
    proposer_max_output_bytes: int = 2 * 1024 * 1024
    evaluator_timeout_sec: float = 2700.0
    stop_after_promotion: bool = True
    container_uid: int = 1000
    container_gid: int = 1000
    video_gid: int = 44
    proposer_cpus: float = 2.0
    proposer_memory: str = PROPOSER_MEMORY_LIMIT
    evaluator_cpus: float = 8.0
    evaluator_memory: str = EVALUATOR_MEMORY_LIMIT
    acknowledge_gpu_passthrough_risk: bool = False

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "ControllerConfig":
        config_path = Path(path)
        try:
            value = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read controller config: {exc}") from exc
        obj = _strict_object(value, CONFIG_FIELDS, "config")
        if (
            type(obj.get("schema_version")) is not int
            or obj.get("schema_version") != 1
        ):
            raise ValueError("config schema_version must be 1")
        required = {
            "repository_dir",
            "state_dir",
            "controller_dir",
            "checkpoint_dir",
            "docker_binary",
            "proposer_image",
            "evaluator_image",
            "deepseek_key_file",
            "gpu_devices",
            "evaluator_cache_dir",
            "expected_git_commit",
            "expected_kernel_hash",
            "acknowledge_gpu_passthrough_risk",
        }
        missing = sorted(required - set(obj))
        if missing:
            raise ValueError(f"config is missing fields: {', '.join(missing)}")
        images = {}
        for field in ("proposer_image", "evaluator_image"):
            image = obj[field]
            if not isinstance(image, str) or not IMAGE_RE.fullmatch(image):
                raise ValueError(f"{field} must use an immutable @sha256: reference")
            images[field] = image
        devices = obj["gpu_devices"]
        if not isinstance(devices, list) or len(devices) != 3:
            raise ValueError("gpu_devices must contain exactly three device paths")
        commit = obj["expected_git_commit"]
        if (
            not isinstance(commit, str)
            or not re.fullmatch(r"[0-9a-f]{40}", commit)
        ):
            raise ValueError("expected_git_commit must be 40 lowercase hex digits")
        kernel_hash = obj["expected_kernel_hash"]
        if not isinstance(kernel_hash, str) or not SHA256_RE.fullmatch(kernel_hash):
            raise ValueError("expected_kernel_hash must be 64 lowercase hex digits")
        stop_after = obj.get("stop_after_promotion", True)
        if not isinstance(stop_after, bool):
            raise ValueError("stop_after_promotion must be boolean")
        if not stop_after:
            raise ValueError(
                "the MVP requires stop_after_promotion to remain true"
            )
        acknowledge_gpu_risk = obj["acknowledge_gpu_passthrough_risk"]
        if not isinstance(acknowledge_gpu_risk, bool):
            raise ValueError("acknowledge_gpu_passthrough_risk must be boolean")
        memory_fields: dict[str, str] = {}
        for field, default in (
            ("proposer_memory", PROPOSER_MEMORY_LIMIT),
            ("evaluator_memory", EVALUATOR_MEMORY_LIMIT),
        ):
            memory = obj.get(field, default)
            if not isinstance(memory, str) or not re.fullmatch(
                r"[1-9][0-9]*(?:[kKmMgG])?", memory
            ):
                raise ValueError(f"{field} must be a Docker memory value")
            memory_fields[field] = memory
        repository_dir = _canonical_path(obj["repository_dir"], "repository_dir")
        state_dir = _canonical_path(obj["state_dir"], "state_dir")
        controller_dir = _canonical_path(obj["controller_dir"], "controller_dir")
        checkpoint_dir = _canonical_path(obj["checkpoint_dir"], "checkpoint_dir")
        evaluator_cache_dir = _canonical_path(
            obj["evaluator_cache_dir"], "evaluator_cache_dir"
        )
        deepseek_key_file = _canonical_path(
            obj["deepseek_key_file"], "deepseek_key_file"
        )
        gpu_devices = tuple(
            _canonical_path(device, "gpu_devices") for device in devices
        )
        if gpu_devices != GPU1_DEVICES:
            raise ValueError(
                "gpu_devices must be exactly /dev/mxcd, /dev/dri/card2 and "
                "/dev/dri/renderD129 in that order"
            )
        roots = {
            "repository_dir": repository_dir,
            "state_dir": state_dir,
            "controller_dir": controller_dir,
            "checkpoint_dir": checkpoint_dir,
            "evaluator_cache_dir": evaluator_cache_dir,
        }
        for name, root in roots.items():
            if root == Path("/"):
                raise ValueError(f"{name} may not be the filesystem root")
        for name, root in roots.items():
            if name == "repository_dir":
                continue
            if root == repository_dir or root.is_relative_to(repository_dir):
                raise ValueError(f"{name} must be outside repository_dir")
        if (
            deepseek_key_file.is_relative_to(repository_dir)
            or deepseek_key_file.is_relative_to(state_dir)
            or deepseek_key_file.is_relative_to(controller_dir)
        ):
            raise ValueError(
                "deepseek_key_file must be outside repository, state and controller"
            )
        opencode_model = str(
            obj.get("opencode_model", "deepseek/deepseek-v4-pro")
        )
        if opencode_model != "deepseek/deepseek-v4-pro":
            raise ValueError("opencode_model must be deepseek/deepseek-v4-pro")
        max_candidates = _positive_int(
            obj.get("max_candidates", 5), "max_candidates"
        )
        max_hours = _positive_number(obj.get("max_hours", 6), "max_hours")
        max_failures = _positive_int(
            obj.get("max_consecutive_failures", 3),
            "max_consecutive_failures",
        )
        proposer_timeout = _positive_number(
            obj.get("proposer_timeout_sec", 1200),
            "proposer_timeout_sec",
        )
        proposer_output = _positive_int(
            obj.get("proposer_max_output_bytes", 2 * 1024 * 1024),
            "proposer_max_output_bytes",
        )
        evaluator_timeout = _positive_number(
            obj.get("evaluator_timeout_sec", 2700),
            "evaluator_timeout_sec",
        )
        proposer_cpus = _positive_number(
            obj.get("proposer_cpus", 2), "proposer_cpus"
        )
        evaluator_cpus = _positive_number(
            obj.get("evaluator_cpus", 8), "evaluator_cpus"
        )
        for field, value, maximum in (
            ("max_candidates", max_candidates, MAX_AUTORESEARCH_CANDIDATES),
            ("max_hours", max_hours, MAX_AUTORESEARCH_HOURS),
            (
                "max_consecutive_failures",
                max_failures,
                MAX_CONSECUTIVE_FAILURES,
            ),
            (
                "proposer_timeout_sec",
                proposer_timeout,
                MAX_PROPOSER_TIMEOUT_SEC,
            ),
            (
                "proposer_max_output_bytes",
                proposer_output,
                MAX_PROPOSER_OUTPUT_BYTES,
            ),
            (
                "evaluator_timeout_sec",
                evaluator_timeout,
                MAX_EVALUATOR_TIMEOUT_SEC,
            ),
            ("proposer_cpus", proposer_cpus, MAX_PROPOSER_CPUS),
            ("evaluator_cpus", evaluator_cpus, MAX_EVALUATOR_CPUS),
        ):
            if value > maximum:
                raise ValueError(f"{field} exceeds the MVP safety maximum")
        if (
            memory_fields["proposer_memory"].lower()
            != PROPOSER_MEMORY_LIMIT
        ):
            raise ValueError(
                f"proposer_memory must remain {PROPOSER_MEMORY_LIMIT}"
            )
        if (
            memory_fields["evaluator_memory"].lower()
            != EVALUATOR_MEMORY_LIMIT
        ):
            raise ValueError(
                f"evaluator_memory must remain {EVALUATOR_MEMORY_LIMIT}"
            )
        integer_fields = {}
        for field, default in (
            ("container_uid", 1000),
            ("container_gid", 1000),
            ("video_gid", 44),
        ):
            value = obj.get(field, default)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
            integer_fields[field] = value
        return cls(
            repository_dir=repository_dir,
            state_dir=state_dir,
            controller_dir=controller_dir,
            checkpoint_dir=checkpoint_dir,
            docker_binary=_canonical_path(obj["docker_binary"], "docker_binary"),
            proposer_image=images["proposer_image"],
            evaluator_image=images["evaluator_image"],
            deepseek_key_file=deepseek_key_file,
            gpu_devices=gpu_devices,
            evaluator_cache_dir=evaluator_cache_dir,
            expected_git_commit=commit,
            expected_kernel_hash=kernel_hash,
            opencode_model=opencode_model,
            max_candidates=max_candidates,
            max_hours=max_hours,
            max_consecutive_failures=max_failures,
            proposer_timeout_sec=proposer_timeout,
            proposer_max_output_bytes=proposer_output,
            evaluator_timeout_sec=evaluator_timeout,
            stop_after_promotion=stop_after,
            container_uid=integer_fields["container_uid"],
            container_gid=integer_fields["container_gid"],
            video_gid=integer_fields["video_gid"],
            proposer_cpus=proposer_cpus,
            proposer_memory=memory_fields["proposer_memory"],
            evaluator_cpus=evaluator_cpus,
            evaluator_memory=memory_fields["evaluator_memory"],
            acknowledge_gpu_passthrough_risk=acknowledge_gpu_risk,
        )

    def validate_host(self, *, require_secret: bool = True) -> list[str]:
        errors: list[str] = []
        for path, kind in (
            (self.repository_dir, "directory"),
            (self.state_dir, "directory"),
            (self.docker_binary, "file"),
            (self.evaluator_cache_dir, "directory"),
        ):
            if kind == "directory" and not path.is_dir():
                errors.append(f"{path} is not a directory")
            if kind == "file" and not path.is_file():
                errors.append(f"{path} is not a file")
            if (
                kind == "file"
                and path.is_file()
                and path == self.docker_binary
                and not os.access(path, os.X_OK)
            ):
                errors.append(f"{path} is not executable")
        if not (self.state_dir / "history.sqlite3").is_file():
            errors.append("state_dir does not contain history.sqlite3")
        if self.container_uid != os.getuid():
            errors.append("container_uid must match the trusted controller uid")
        if self.container_gid != os.getgid():
            errors.append("container_gid must match the trusted controller gid")
        for device in self.gpu_devices:
            if not device.exists():
                errors.append(f"GPU device is missing: {device}")
            elif not stat.S_ISCHR(device.stat().st_mode):
                errors.append(f"GPU device is not a character device: {device}")
        if require_secret:
            if not self.deepseek_key_file.is_file():
                errors.append(f"DeepSeek key file is missing: {self.deepseek_key_file}")
            else:
                mode = stat.S_IMODE(self.deepseek_key_file.stat().st_mode)
                if mode != 0o600:
                    errors.append("DeepSeek key file mode must be exactly 0600")
                if self.deepseek_key_file.stat().st_uid != os.getuid():
                    errors.append("DeepSeek key file must be owned by the controller uid")
                if self.deepseek_key_file.stat().st_size == 0:
                    errors.append("DeepSeek key file must not be empty")
                elif self.deepseek_key_file.stat().st_size > 512:
                    errors.append("DeepSeek key file is unexpectedly large")
                else:
                    key_bytes = self.deepseek_key_file.read_bytes()
                    if any(byte in b" \t\r\n" for byte in key_bytes):
                        errors.append(
                            "DeepSeek key file must contain one token without whitespace"
                        )
                    elif not all(33 <= byte <= 126 for byte in key_bytes):
                        errors.append(
                            "DeepSeek key file must contain printable ASCII only"
                        )
        return errors

    def redacted_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "repository_dir": str(self.repository_dir),
            "state_dir": str(self.state_dir),
            "controller_dir": str(self.controller_dir),
            "checkpoint_dir": str(self.checkpoint_dir),
            "docker_binary": str(self.docker_binary),
            "proposer_image": self.proposer_image,
            "evaluator_image": self.evaluator_image,
            "deepseek_key_file": str(self.deepseek_key_file),
            "gpu_devices": [str(path) for path in self.gpu_devices],
            "evaluator_cache_dir": str(self.evaluator_cache_dir),
            "expected_git_commit": self.expected_git_commit,
            "expected_kernel_hash": self.expected_kernel_hash,
            "opencode_model": self.opencode_model,
            "max_candidates": self.max_candidates,
            "max_hours": self.max_hours,
            "max_consecutive_failures": self.max_consecutive_failures,
            "proposer_timeout_sec": self.proposer_timeout_sec,
            "proposer_max_output_bytes": self.proposer_max_output_bytes,
            "evaluator_timeout_sec": self.evaluator_timeout_sec,
            "stop_after_promotion": self.stop_after_promotion,
            "container_uid": self.container_uid,
            "container_gid": self.container_gid,
            "video_gid": self.video_gid,
            "proposer_cpus": self.proposer_cpus,
            "proposer_memory": self.proposer_memory,
            "evaluator_cpus": self.evaluator_cpus,
            "evaluator_memory": self.evaluator_memory,
            "acknowledge_gpu_passthrough_risk": (
                self.acknowledge_gpu_passthrough_risk
            ),
        }
