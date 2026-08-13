"""Trusted, standard-library-only server administration for autoresearch.

This module is intentionally separate from the research state machine.  It
never proposes kernels or starts a research run; it only maintains pinned
configuration and verifies the trusted host deployment.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sqlite3
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence
import uuid

from ..constants import OPENCODE_PROPOSER_STEPS
from ..campaign.paths import (
    campaign_maintenance_fence,
    conventional_campaign_database,
    verify_inherited_campaign_maintenance_fence,
)
from ..history import ExperimentRecord, HistoryStore
from ..platform.artifacts import ArtifactId
from ..platform.canonical import canonical_json_bytes
from ..platform.identity import (
    BaselineRef,
    ExecutionEnvironmentDigest,
    ExperimentIdentity,
)
from ..platform.profiles import (
    CURRENT_RESEARCH_NAMESPACE,
    LEGACY_RESEARCH_NAMESPACE,
    ResearchNamespace,
)
from ..platform.proposal import CandidateBundle, TRITON_PYTHON_BUNDLE_LIMITS
from .controller import ResearchController, gpu_lock
from .deployment import (
    DEPLOYMENT_BASELINE_FILENAME,
    DeploymentBaselinePin,
    deployment_runtime_root,
)
from .errors import ControlledRuntimeError
from .model_catalog import (
    OPENCODE_MODEL_SPECS,
    OPENCODE_OUTPUT_TOKEN_MAX_ENV,
)
from .models import CONFIG_FIELDS, ControllerConfig
from .runtime import CommandRunner, proposer_argv, write_opencode_config
from .states import TERMINAL_RUN_STATUSES
from .store import ControllerStore


ADMIN_SCHEMA_VERSION = 1
PRO_MODEL = "deepseek/deepseek-v4-pro"
FLASH_MODEL = "deepseek/deepseek-v4-flash"
ADMIN_REPORT_LIMIT = 8 * 1024 * 1024
ADMIN_CPU_TIMEOUT_SEC = 1200.0

MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "repository_dir",
        "runtime_root",
        "base_config",
        "pro_config",
        "flash_config",
        "pro_canary_config",
        "flash_canary_config",
        "environment_file",
        "expected_branch",
        "expected_upstream",
        "host_python",
    }
)


def _canonical_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field} must be absolute")
    normalized = Path(os.path.normpath(value))
    if normalized != path or path.resolve(strict=False) != path:
        raise ValueError(f"{field} must be an absolute canonical path")
    return path


def _reject_symlink_path(path: Path) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"administration path may not be a symlink: {current}")
        if current.parent == current:
            return
        current = current.parent


@dataclass(frozen=True)
class AdminManifest:
    repository_dir: Path
    runtime_root: Path
    base_config: Path
    pro_config: Path
    flash_config: Path
    pro_canary_config: Path
    flash_canary_config: Path
    environment_file: Path
    expected_branch: str
    expected_upstream: str
    host_python: Path

    @classmethod
    def load(cls, path: Path) -> "AdminManifest":
        path = _canonical_path(str(path), "manifest")
        _reject_symlink_path(path)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read admin manifest: {exc}") from exc
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise ValueError("admin manifest mode must be exactly 0600")
        if not isinstance(value, dict):
            raise ValueError("admin manifest must be a JSON object")
        unknown = sorted(set(value) - MANIFEST_FIELDS)
        missing = sorted(MANIFEST_FIELDS - set(value))
        if unknown:
            raise ValueError(
                "admin manifest contains unknown fields: " + ", ".join(unknown)
            )
        if missing:
            raise ValueError(
                "admin manifest is missing fields: " + ", ".join(missing)
            )
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("admin manifest schema_version must be 1")
        strings: dict[str, str] = {}
        for field in ("expected_branch", "expected_upstream"):
            item = value[field]
            if not isinstance(item, str) or not item or any(
                character.isspace() for character in item
            ):
                raise ValueError(f"{field} must be a non-empty ref without whitespace")
            strings[field] = item
        paths = {
            field: _canonical_path(value[field], field)
            for field in (
                "repository_dir",
                "runtime_root",
                "base_config",
                "pro_config",
                "flash_config",
                "pro_canary_config",
                "flash_canary_config",
                "environment_file",
                "host_python",
            )
        }
        for candidate in paths.values():
            _reject_symlink_path(candidate)
        if len({paths[name] for name in paths if name.endswith("config")}) != 5:
            raise ValueError("all generated config paths must be distinct")
        for name in (
            "base_config",
            "pro_config",
            "flash_config",
            "pro_canary_config",
            "flash_canary_config",
            "environment_file",
        ):
            if paths[name].parent != paths["runtime_root"]:
                raise ValueError(f"{name} must be directly inside runtime_root")
        if path != paths["runtime_root"] / "admin.json":
            raise ValueError("manifest must be runtime_root/admin.json")
        return cls(
            repository_dir=paths["repository_dir"],
            runtime_root=paths["runtime_root"],
            base_config=paths["base_config"],
            pro_config=paths["pro_config"],
            flash_config=paths["flash_config"],
            pro_canary_config=paths["pro_canary_config"],
            flash_canary_config=paths["flash_canary_config"],
            environment_file=paths["environment_file"],
            expected_branch=strings["expected_branch"],
            expected_upstream=strings["expected_upstream"],
            host_python=paths["host_python"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ADMIN_SCHEMA_VERSION,
            "repository_dir": str(self.repository_dir),
            "runtime_root": str(self.runtime_root),
            "base_config": str(self.base_config),
            "pro_config": str(self.pro_config),
            "flash_config": str(self.flash_config),
            "pro_canary_config": str(self.pro_canary_config),
            "flash_canary_config": str(self.flash_canary_config),
            "environment_file": str(self.environment_file),
            "expected_branch": self.expected_branch,
            "expected_upstream": self.expected_upstream,
            "host_python": str(self.host_python),
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")


def _read_json(path: Path, name: str) -> dict[str, Any]:
    _reject_symlink_path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _run(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout: float = 60.0,
    check: bool = True,
    env: Mapping[str, str] | None = None,
    pass_fds: Sequence[int] = (),
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [str(item) for item in argv],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
        env=None if env is None else dict(env),
        pass_fds=tuple(pass_fds),
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ControlledRuntimeError(
            f"command failed ({completed.returncode}): {argv[0]}: {detail}"
        )
    return completed


def _git(repository: Path, *arguments: str, check: bool = True) -> str:
    return _run(
        ["git", *arguments], cwd=repository, timeout=120, check=check
    ).stdout.strip()


def _git_blob(repository: Path, revision_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", revision_path],
        cwd=repository,
        check=False,
        capture_output=True,
        timeout=30,
        shell=False,
    )
    if completed.returncode != 0:
        raise ControlledRuntimeError(
            "could not read committed kernel.py: "
            + completed.stderr.decode("utf-8", errors="replace").strip()
        )
    return completed.stdout


def _repo_state(manifest: AdminManifest, *, require_clean: bool = True) -> dict[str, str]:
    repository = manifest.repository_dir
    if not repository.is_dir():
        raise ControlledRuntimeError("repository_dir is not a directory")
    branch = _git(repository, "symbolic-ref", "--short", "HEAD")
    upstream = _git(
        repository, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"
    )
    if branch != manifest.expected_branch:
        raise ControlledRuntimeError(
            f"branch {branch} does not match manifest {manifest.expected_branch}"
        )
    if upstream != manifest.expected_upstream:
        raise ControlledRuntimeError(
            f"upstream {upstream} does not match manifest {manifest.expected_upstream}"
        )
    status = _git(repository, "status", "--porcelain")
    if require_clean and status:
        raise ControlledRuntimeError("repository worktree is not clean")
    head = _git(repository, "rev-parse", "HEAD")
    kernel_path = repository / "kernel.py"
    if kernel_path.is_symlink() or not kernel_path.is_file():
        raise ControlledRuntimeError("kernel.py must be a regular non-symlink file")
    return {
        "branch": branch,
        "upstream": upstream,
        "head": head,
        "kernel_hash": _sha256_file(kernel_path),
        "kernel_blob_hash": _sha256_bytes(_git_blob(repository, "HEAD:kernel.py")),
    }


def _load_base(manifest: AdminManifest) -> dict[str, Any]:
    base = _read_json(manifest.base_config, "base config")
    unknown = sorted(set(base) - CONFIG_FIELDS)
    if unknown:
        raise ValueError("base config contains unknown fields: " + ", ".join(unknown))
    if "opencode_model" in base:
        raise ValueError("base config must not contain opencode_model")
    if base.get("max_candidates", 5) != 5:
        raise ValueError("base config max_candidates must be 5")
    return base


def _render_config_values(
    base: Mapping[str, Any], *, commit: str, kernel_hash: str
) -> dict[str, dict[str, Any]]:
    normalized = dict(base)
    framework_commit = normalized.get(
        "framework_git_commit", normalized.get("expected_git_commit", commit)
    )
    normalized["expected_git_commit"] = commit
    normalized["framework_git_commit"] = framework_commit
    normalized["expected_kernel_hash"] = kernel_hash
    normalized["max_candidates"] = 5
    normalized.pop("opencode_model", None)

    def selected(model: str, candidates: int) -> dict[str, Any]:
        value = dict(normalized)
        value["opencode_model"] = model
        value["max_candidates"] = candidates
        return value

    return {
        "base": normalized,
        "pro": selected(PRO_MODEL, 5),
        "flash": selected(FLASH_MODEL, 5),
        "pro_canary": selected(PRO_MODEL, 1),
        "flash_canary": selected(FLASH_MODEL, 1),
    }


def _config_targets(manifest: AdminManifest) -> dict[str, Path]:
    return {
        "base": manifest.base_config,
        "pro": manifest.pro_config,
        "flash": manifest.flash_config,
        "pro_canary": manifest.pro_canary_config,
        "flash_canary": manifest.flash_canary_config,
    }


def _validate_generated(values: Mapping[str, Mapping[str, Any]]) -> None:
    with tempfile.TemporaryDirectory(prefix="kar-admin-config-") as temporary:
        root = Path(temporary)
        loaded: dict[str, ControllerConfig] = {}
        for name, value in values.items():
            path = root / f"{name}.json"
            path.write_bytes(_json_bytes(value))
            loaded[name] = ControllerConfig.load(path)
    pro = loaded["pro"].redacted_dict()
    flash = loaded["flash"].redacted_dict()
    pro_canary = loaded["pro_canary"].redacted_dict()
    flash_canary = loaded["flash_canary"].redacted_dict()
    changed = {key for key in pro if pro.get(key) != flash.get(key)}
    if changed != {"opencode_model"}:
        raise ValueError("formal Pro/Flash configs differ outside opencode_model")
    for formal, canary in ((pro, pro_canary), (flash, flash_canary)):
        changed = {key for key in formal if formal.get(key) != canary.get(key)}
        if changed != {"max_candidates"} or canary["max_candidates"] != 1:
            raise ValueError("canary config may differ only by max_candidates=1")


def _env_bytes(
    manifest: AdminManifest,
    commit: str,
    kernel_hash: str,
    framework_commit: str,
) -> bytes:
    exports = {
        "AUTORESEARCH_REPO": str(manifest.repository_dir),
        "AUTORESEARCH_RUNTIME": str(manifest.runtime_root),
        "AUTORESEARCH_MANIFEST": str(manifest.runtime_root / "admin.json"),
        "AUTORESEARCH_PRO_CONFIG": str(manifest.pro_config),
        "AUTORESEARCH_FLASH_CONFIG": str(manifest.flash_config),
        "AUTORESEARCH_PRO_CANARY_CONFIG": str(manifest.pro_canary_config),
        "AUTORESEARCH_FLASH_CANARY_CONFIG": str(manifest.flash_canary_config),
        "AUTORESEARCH_COMMIT": commit,
        "AUTORESEARCH_FRAMEWORK_COMMIT": framework_commit,
        "AUTORESEARCH_KERNEL_HASH": kernel_hash,
    }
    lines = ["# Generated by kernel-autoresearch-admin; contains no secrets."]
    lines.extend(
        f"export {name}={shlex.quote(value)}" for name, value in exports.items()
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _atomic_publish(files: Mapping[Path, tuple[bytes, int]]) -> None:
    staged: dict[Path, Path] = {}
    previous: dict[Path, tuple[bytes, int] | None] = {}
    for target in files:
        _reject_symlink_path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            mode = stat.S_IMODE(target.stat().st_mode)
            previous[target] = (target.read_bytes(), mode)
        else:
            previous[target] = None
    try:
        for target, (data, mode) in files.items():
            descriptor, name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
            )
            temporary = Path(name)
            staged[target] = temporary
            try:
                os.fchmod(descriptor, mode)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        replaced: list[Path] = []
        try:
            for target, temporary in staged.items():
                os.replace(temporary, target)
                replaced.append(target)
            parent_descriptors = {
                os.open(target.parent, os.O_RDONLY) for target in files
            }
            try:
                for descriptor in parent_descriptors:
                    os.fsync(descriptor)
            finally:
                for descriptor in parent_descriptors:
                    os.close(descriptor)
        except BaseException:
            for target in reversed(replaced):
                old = previous[target]
                if old is None:
                    target.unlink(missing_ok=True)
                else:
                    data, mode = old
                    descriptor, name = tempfile.mkstemp(
                        prefix=f".{target.name}.restore.", dir=target.parent
                    )
                    with os.fdopen(descriptor, "wb") as handle:
                        os.fchmod(handle.fileno(), mode)
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(name, target)
            raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)


def _trusted_namespace(namespace_id: str | None) -> ResearchNamespace:
    """Resolve only exact built-in scientific namespaces.

    Omitting the namespace is the one-cycle V1 compatibility path.  It is
    deliberately equivalent to naming the legacy namespace; no alias can be
    used to relabel evidence.
    """

    selected = (
        LEGACY_RESEARCH_NAMESPACE.namespace_id
        if namespace_id is None
        else namespace_id
    )
    trusted = {
        LEGACY_RESEARCH_NAMESPACE.namespace_id: LEGACY_RESEARCH_NAMESPACE,
        CURRENT_RESEARCH_NAMESPACE.namespace_id: CURRENT_RESEARCH_NAMESPACE,
    }.get(selected)
    if trusted is None:
        raise ValueError(
            "namespace must be an exact built-in LEGACY or CURRENT namespace_id"
        )
    return trusted


def _deployment_pin_path(config: ControllerConfig) -> Path:
    try:
        runtime_root = deployment_runtime_root(
            state_dir=config.state_dir,
            controller_dir=config.controller_dir,
            checkpoint_dir=config.checkpoint_dir,
        )
    except ValueError as exc:
        raise ControlledRuntimeError(
            f"formal config has no unique deployment runtime root: {exc}"
        ) from exc
    return runtime_root / DEPLOYMENT_BASELINE_FILENAME


def _configured_namespace_id(config: ControllerConfig) -> str:
    path = _deployment_pin_path(config)
    if not path.exists() and not path.is_symlink():
        return LEGACY_RESEARCH_NAMESPACE.namespace_id
    try:
        pin = DeploymentBaselinePin.load(path)
        _trusted_namespace(pin.namespace_id)
    except ValueError as exc:
        raise ControlledRuntimeError(
            f"deployment baseline pin is invalid: {exc}"
        ) from exc
    if pin.candidate_hash != config.expected_kernel_hash:
        raise ControlledRuntimeError(
            "deployment baseline pin candidate differs from formal config"
        )
    if pin.git_commit != config.expected_git_commit:
        raise ControlledRuntimeError(
            "deployment baseline pin Git commit differs from formal config"
        )
    return pin.namespace_id


def _deployment_pin_for_commit(
    config: ControllerConfig,
    *,
    git_commit: str,
    framework_git_commit: str,
) -> DeploymentBaselinePin | None:
    """Reissue a pin only when its scientific framework stays byte-frozen.

    The full deployment commit may advance for control-plane fixes or an
    operator-reviewed ``kernel.py``.  The evaluator framework is independently
    materialized from ``framework_git_commit``; a resolved pin may cross only
    that non-scientific deployment transition.
    """

    path = _deployment_pin_path(config)
    if not path.exists() and not path.is_symlink():
        return None
    try:
        pin = DeploymentBaselinePin.load(path)
        _trusted_namespace(pin.namespace_id)
    except ValueError as exc:
        raise ControlledRuntimeError(
            f"deployment baseline pin is invalid: {exc}"
        ) from exc
    if pin.candidate_hash != config.expected_kernel_hash:
        raise ControlledRuntimeError(
            "deployment baseline pin candidate differs from formal config"
        )
    if pin.execution_environment.is_resolved:
        if framework_git_commit != config.resolved_framework_git_commit:
            raise ControlledRuntimeError(
                "resolved deployment evidence cannot cross a framework commit"
            )
        namespace = _trusted_namespace(pin.namespace_id)
        try:
            ResearchController(config)._resolved_execution_environment(
                namespace
            ).require_match(
                pin.execution_environment,
                context="deployment update",
            )
        except ValueError as exc:
            raise ControlledRuntimeError(str(exc)) from exc
    if pin.git_commit == git_commit:
        return pin
    return DeploymentBaselinePin.create(
        namespace_id=pin.namespace_id,
        baseline_ref=pin.baseline_ref,
        candidate_hash=pin.candidate_hash,
        git_commit=git_commit,
        primary_experiment_uid=pin.primary_experiment_uid,
        confirmation_experiment_uid=pin.confirmation_experiment_uid,
        confirmation_experiment_id=pin.confirmation_experiment_id,
        parent_baseline_ref=pin.parent_baseline_ref,
        execution_environment=pin.execution_environment,
    )


def _add_deployment_pin_publish(
    files: dict[Path, tuple[bytes, int]],
    hashes: dict[str, str],
    *,
    manifest: AdminManifest,
    pin: DeploymentBaselinePin | None,
) -> None:
    if pin is None:
        return
    path = manifest.runtime_root / DEPLOYMENT_BASELINE_FILENAME
    data = _json_bytes(pin.to_dict())
    files[path] = (data, 0o600)
    hashes["deployment_baseline"] = _sha256_bytes(data)


def _checked_history_path(
    config: ControllerConfig, relative_path: str, *, name: str
) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ControlledRuntimeError(f"{name} has no object path")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ControlledRuntimeError(f"{name} has an unsafe object path")
    raw = config.state_dir / relative
    try:
        _reject_symlink_path(raw)
    except ValueError as exc:
        raise ControlledRuntimeError(f"{name} uses a symlinked object path") from exc
    root = config.state_dir.resolve(strict=False)
    resolved = raw.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ControlledRuntimeError(f"{name} escapes state_dir")
    if not raw.is_file():
        raise ControlledRuntimeError(f"{name} is missing")
    return raw


def _bundle_entrypoint_artifact(
    history: HistoryStore,
    config: ControllerConfig,
    best: ExperimentRecord,
) -> Path:
    """Verify bundle object, manifest, entrypoint source CAS, and exact bytes."""

    bundle_record = history.get_candidate_artifact(best.artifact_id)
    if bundle_record is None or bundle_record.artifact_kind != "source_bundle_v1":
        raise ControlledRuntimeError(
            "accepted bundle has no authoritative bundle CAS record"
        )
    bundle_path = _checked_history_path(
        config, bundle_record.object_path, name="accepted bundle CAS object"
    )
    try:
        bundle_bytes = history.read_candidate_artifact(best.artifact_id)
    except (KeyError, RuntimeError) as exc:
        raise ControlledRuntimeError(
            "accepted bundle CAS object is missing or corrupted"
        ) from exc
    if bundle_path.read_bytes() != bundle_bytes or bundle_record.byte_size != len(
        bundle_bytes
    ):
        raise ControlledRuntimeError("accepted bundle CAS metadata is inconsistent")
    try:
        decoded = json.loads(bundle_bytes.decode("utf-8"))
        bundle = CandidateBundle.from_value(
            decoded, limits=TRITON_PYTHON_BUNDLE_LIMITS
        )
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ControlledRuntimeError(
            "accepted bundle CAS payload is not a valid trusted-language bundle"
        ) from exc
    if (
        bundle.bundle_bytes != bundle_bytes
        or str(bundle.artifact_id) != best.artifact_id
        or dict(bundle_record.manifest) != bundle.manifest
        or canonical_json_bytes(bundle_record.manifest) != bundle.manifest_bytes
    ):
        raise ControlledRuntimeError(
            "accepted bundle artifact ID, manifest, and payload disagree"
        )
    entrypoint = next(
        item for item in bundle.files if item.path == bundle.entrypoint
    )
    source_id = str(ArtifactId.source_sha256(best.candidate_hash))
    source_record = history.get_candidate_artifact(source_id)
    if source_record is None or source_record.artifact_kind != "source_text_v1":
        raise ControlledRuntimeError(
            "accepted bundle has no verified entrypoint source artifact"
        )
    source_path = _checked_history_path(
        config, source_record.object_path, name="accepted entrypoint source CAS object"
    )
    try:
        source_bytes = history.read_candidate_artifact(source_id)
    except (KeyError, RuntimeError) as exc:
        raise ControlledRuntimeError(
            "accepted entrypoint source CAS object is missing or corrupted"
        ) from exc
    if (
        source_path.read_bytes() != source_bytes
        or source_record.byte_size != len(source_bytes)
        or source_bytes != entrypoint.content_bytes
        or _sha256_bytes(source_bytes) != best.candidate_hash
    ):
        raise ControlledRuntimeError(
            "accepted bundle entrypoint, source CAS, and History hash disagree"
        )
    return source_path


def _artifact_for_best(
    config: ControllerConfig,
    *,
    candidate_hash: str | None = None,
    namespace_id: str | None = None,
) -> tuple[ExperimentRecord, Path]:
    """Resolve explicit deployment/adoption evidence, never a global best."""

    selected_hash = candidate_hash or config.expected_kernel_hash
    selected_namespace = (
        _configured_namespace_id(config)
        if namespace_id is None
        else _trusted_namespace(namespace_id).namespace_id
    )
    with HistoryStore(
        config.state_dir / "history.sqlite3", state_dir=config.state_dir
    ) as history:
        records = history.find_by_candidate_hash(
            selected_hash,
            namespace_id=selected_namespace,
        )
        best = next(
            (
                record
                for record in reversed(records)
                if record.backend == "c500"
                and record.suite == "full"
                and record.status == "SUCCESS"
                and record.promotable
                and record.result.get("promotion", {}).get("phase")
                in {"confirmation", "baseline"}
            ),
            None,
        )
        if best is None:
            raise ControlledRuntimeError(
                "candidate has no accepted C500 full evidence in the deployment namespace"
            )
        artifact = _artifact_for_experiment(history, config, best)
    return best, artifact


def _artifact_for_experiment(
    history: HistoryStore,
    config: ControllerConfig,
    experiment: ExperimentRecord,
) -> Path:
    """Verify the authoritative artifact for one already-selected experiment."""

    try:
        artifact_id = ArtifactId.parse(experiment.artifact_id)
    except ValueError as exc:
        raise ControlledRuntimeError(
            "accepted History artifact ID is invalid"
        ) from exc
    if artifact_id.tag == "bundle-sha256-v1":
        artifact = _bundle_entrypoint_artifact(history, config, experiment)
    else:
        artifact = _checked_history_path(
            config,
            experiment.artifact_path,
            name="accepted History artifact",
        )
        if artifact_id.digest != experiment.candidate_hash:
            raise ControlledRuntimeError(
                "accepted source artifact ID differs from candidate hash"
            )
    if _sha256_file(artifact) != experiment.candidate_hash:
        raise ControlledRuntimeError("accepted History artifact hash mismatch")
    return artifact


def _artifact_for_pinned_confirmation(
    config: ControllerConfig,
    pin: DeploymentBaselinePin,
) -> tuple[ExperimentRecord, Path]:
    """Resolve the immutable History row named by a deployment pin.

    A newer promotable row may legitimately reuse the same artifact, for
    example a resolved baseline-qualification experiment.  Such a row must not
    replace the confirmation ID and UID that the administrator actually
    published.
    """

    namespace = _trusted_namespace(pin.namespace_id)
    if pin.baseline_ref.namespace_id != namespace.namespace_id:
        raise ControlledRuntimeError(
            "deployment baseline pin has inconsistent scientific coordinates"
        )
    with HistoryStore(
        config.state_dir / "history.sqlite3", state_dir=config.state_dir
    ) as history:
        confirmation = history.get_experiment(
            pin.confirmation_experiment_id
        )
        if confirmation is None:
            raise ControlledRuntimeError(
                "deployment baseline pin confirmation experiment is missing"
            )
        if (
            confirmation.experiment_uid
            != pin.confirmation_experiment_uid
            or confirmation.namespace_id != pin.namespace_id
            or confirmation.candidate_hash != pin.candidate_hash
            or confirmation.artifact_id
            != str(pin.baseline_ref.artifact_id)
        ):
            raise ControlledRuntimeError(
                "deployment baseline pin confirmation ID/UID or artifact "
                "coordinates disagree"
            )
        artifact = _artifact_for_experiment(
            history, config, confirmation
        )
    return confirmation, artifact


@dataclass(frozen=True)
class _AdoptionProof:
    confirmation: ExperimentRecord
    primary: ExperimentRecord | None
    artifact: Path
    namespace: ResearchNamespace
    parent_baseline: BaselineRef | None
    execution_environment: ExecutionEnvironmentDigest


def _correct_full_evidence(record: ExperimentRecord) -> bool:
    return bool(record.case_measurements) and all(
        case.passed is True
        and case.matched_ratio is not None
        and float(case.matched_ratio) >= 1.0
        for case in record.case_measurements
    )


def _legacy_adoption_proof(
    config: ControllerConfig,
    *,
    candidate_hash: str,
    selected_confirmation: tuple[ExperimentRecord, Path] | None = None,
) -> _AdoptionProof:
    """Grandfather the genuine V1 confirmation shape for one compatibility cycle."""

    best, artifact = (
        _artifact_for_best(
            config,
            candidate_hash=candidate_hash,
            namespace_id=LEGACY_RESEARCH_NAMESPACE.namespace_id,
        )
        if selected_confirmation is None
        else selected_confirmation
    )
    promotion = best.result.get("promotion")
    if (
        best.namespace_id != LEGACY_RESEARCH_NAMESPACE.namespace_id
        or best.replicate_kind != "legacy"
        or best.backend != "c500"
        or best.suite != "full"
        or best.status != "SUCCESS"
        or not best.promotable
        or not _correct_full_evidence(best)
        or not isinstance(promotion, Mapping)
        or promotion.get("phase") != "confirmation"
        or promotion.get("reason") != "promoted"
        or promotion.get("confirmed") is not True
    ):
        raise ControlledRuntimeError(
            "legacy adoption requires genuine V1 accepted confirmation evidence"
        )
    try:
        ExperimentIdentity.from_value(dict(best.identity))
    except (TypeError, ValueError):
        pass
    else:
        raise ControlledRuntimeError(
            "V2-shaped evidence cannot use the V1 grandfather adoption path"
        )
    environment = ExecutionEnvironmentDigest.legacy_unknown(
        scope={
            "namespace_id": LEGACY_RESEARCH_NAMESPACE.namespace_id,
            "artifact_id": best.artifact_id,
            "confirmation_experiment_uid": best.experiment_uid,
        }
    )
    return _AdoptionProof(
        confirmation=best,
        primary=None,
        artifact=artifact,
        namespace=LEGACY_RESEARCH_NAMESPACE,
        parent_baseline=None,
        execution_environment=environment,
    )


def _strict_v2_adoption_proof(
    config: ControllerConfig,
    *,
    candidate_hash: str,
    namespace: ResearchNamespace,
    selected_confirmation: tuple[ExperimentRecord, Path] | None = None,
) -> _AdoptionProof:
    confirmation, artifact = (
        _artifact_for_best(
            config,
            candidate_hash=candidate_hash,
            namespace_id=namespace.namespace_id,
        )
        if selected_confirmation is None
        else selected_confirmation
    )
    with HistoryStore(
        config.state_dir / "history.sqlite3", state_dir=config.state_dir
    ) as history:
        namespace_record = history.get_namespace(namespace.namespace_id)
        if (
            namespace_record is None
            or dict(namespace_record.identity) != namespace.to_dict()
        ):
            raise ControlledRuntimeError(
                "History namespace snapshot does not match the trusted built-in namespace"
            )
        # Re-read the exact selected row inside this proof and reject replacement
        # or cross-namespace selection between the artifact and lineage checks.
        current_confirmation = history.get_experiment(confirmation.id)
        if current_confirmation != confirmation:
            raise ControlledRuntimeError(
                "confirmation evidence changed during adoption proof"
            )
        confirmation_promotion = confirmation.result.get("promotion")
        primary_id = (
            confirmation_promotion.get("primary_experiment_id")
            if isinstance(confirmation_promotion, Mapping)
            else None
        )
        if type(primary_id) is not int:
            raise ControlledRuntimeError(
                "confirmation evidence has no linked primary experiment"
            )
        primary = history.get_experiment(primary_id)
        if primary is None:
            raise ControlledRuntimeError("linked primary experiment is missing")
        try:
            primary_identity = ExperimentIdentity.from_value(dict(primary.identity))
            confirmation_identity = ExperimentIdentity.from_value(
                dict(confirmation.identity)
            )
        except (TypeError, ValueError) as exc:
            raise ControlledRuntimeError(
                "promotion evidence has an invalid V2 scientific identity"
            ) from exc
        if (
            not primary_identity.is_scientifically_comparable
            or not confirmation_identity.is_scientifically_comparable
        ):
            raise ControlledRuntimeError(
                "LEGACY_UNKNOWN evidence cannot be adopted as a V2 baseline"
            )
        primary_promotion = primary.result.get("promotion")
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
        if (
            primary.namespace_id != namespace.namespace_id
            or confirmation.namespace_id != namespace.namespace_id
            or primary_identity.namespace != namespace
            or confirmation_identity.namespace != namespace
            or primary.backend != "c500"
            or confirmation.backend != "c500"
            or primary.suite != "full"
            or confirmation.suite != "full"
            or primary.status != "SUCCESS"
            or confirmation.status != "SUCCESS"
            or primary.promotable
            or not confirmation.promotable
            or not _correct_full_evidence(primary)
            or not _correct_full_evidence(confirmation)
            or primary.artifact_id != confirmation.artifact_id
            or primary.artifact_id != confirmation_identity.candidate_artifact_id.value
            or primary.artifact_id != primary_identity.candidate_artifact_id.value
            or primary.candidate_hash != candidate_hash
            or confirmation.candidate_hash != candidate_hash
            or primary_identity.stage != "full_primary"
            or confirmation_identity.stage != "confirmation"
            or primary_identity.suite != "full"
            or confirmation_identity.suite != "full"
            or primary_identity.replicate_kind != "primary"
            or confirmation_identity.replicate_kind != "confirmation"
            or primary.replicate_kind != "primary"
            or confirmation.replicate_kind != "confirmation"
            or primary_identity.replicate_index != primary.replicate_index
            or confirmation_identity.replicate_index != confirmation.replicate_index
            or primary_identity.experiment_uid != primary.experiment_uid
            or confirmation_identity.experiment_uid
            != confirmation.experiment_uid
            or primary_identity.run_id != confirmation_identity.run_id
            or primary_identity.iteration != confirmation_identity.iteration
            or primary_identity.baseline != confirmation_identity.baseline
            or primary_identity.parent_artifact_id
            != primary_identity.baseline.artifact_id
            or confirmation_identity.parent_artifact_id
            != confirmation_identity.baseline.artifact_id
            or primary_identity.execution_environment
            != confirmation_identity.execution_environment
            or primary_identity.condition_digest
            == confirmation_identity.condition_digest
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
            or not isinstance(confirmation_decision, Mapping)
            or confirmation_decision.get("promoted") is not True
            or confirmation_decision.get("needs_confirmation") is not False
            or confirmation_decision.get("reason") != "promoted"
        ):
            raise ControlledRuntimeError(
                "primary/confirmation evidence does not prove one V2 promotion"
            )
        parent = primary_identity.baseline
        baseline_id = primary_promotion.get("baseline_experiment_id")
        if (
            type(baseline_id) is not int
            or confirmation_promotion.get("baseline_experiment_id") != baseline_id
            or primary_promotion.get("baseline_candidate_hash")
            != confirmation_promotion.get("baseline_candidate_hash")
            or primary.baseline_experiment_uid is None
            or confirmation.baseline_experiment_uid
            != primary.baseline_experiment_uid
        ):
            raise ControlledRuntimeError(
                "primary and confirmation do not share one frozen baseline"
            )
        baseline = history.get_experiment(baseline_id)
        if (
            baseline is None
            or baseline.experiment_uid != primary.baseline_experiment_uid
            or baseline.namespace_id != namespace.namespace_id
            or baseline.artifact_id != str(parent.artifact_id)
            or baseline.candidate_hash
            != primary_promotion.get("baseline_candidate_hash")
            or baseline.status != "SUCCESS"
            or not baseline.promotable
        ):
            raise ControlledRuntimeError(
                "promotion parent baseline cannot be proven from History"
            )
        try:
            baseline_identity = ExperimentIdentity.from_value(dict(baseline.identity))
            if (
                baseline_identity.experiment_uid != baseline.experiment_uid
                or baseline_identity.namespace != namespace
                or str(baseline_identity.candidate_artifact_id)
                != baseline.artifact_id
            ):
                raise ValueError(
                    "baseline identity does not match its History record"
                )
            parent.require_environment(baseline_identity.execution_environment)
            parent.require_environment(primary_identity.execution_environment)
        except (TypeError, ValueError) as exc:
            raise ControlledRuntimeError(
                "promotion baseline or execution environment is unresolved/mismatched"
            ) from exc
        relations = history.list_experiment_relations(
            primary.experiment_uid,
            direction="source",
            relation_type="confirmation_of",
        )
        matching_relations = [
            relation
            for relation in relations
            if relation.target_experiment_uid == confirmation.experiment_uid
            and dict(relation.metadata) == {"baseline_experiment_id": baseline.id}
        ]
        if len(matching_relations) != 1:
            raise ControlledRuntimeError(
                "promotion has no unique immutable primary-to-confirmation link"
            )
    return _AdoptionProof(
        confirmation=confirmation,
        primary=primary,
        artifact=artifact,
        namespace=namespace,
        parent_baseline=parent,
        execution_environment=primary_identity.execution_environment,
    )


def _prove_adoption(
    config: ControllerConfig,
    *,
    candidate_hash: str,
    namespace_id: str | None,
    selected_confirmation: tuple[ExperimentRecord, Path] | None = None,
) -> _AdoptionProof:
    namespace = _trusted_namespace(namespace_id)
    if namespace == LEGACY_RESEARCH_NAMESPACE:
        selected = (
            _artifact_for_best(
                config,
                candidate_hash=candidate_hash,
                namespace_id=namespace.namespace_id,
            )
            if selected_confirmation is None
            else selected_confirmation
        )
        if selected[0].replicate_kind == "legacy":
            return _legacy_adoption_proof(
                config,
                candidate_hash=candidate_hash,
                selected_confirmation=selected,
            )
    return _strict_v2_adoption_proof(
        config,
        candidate_hash=candidate_hash,
        namespace=namespace,
        selected_confirmation=selected_confirmation,
    )


def _identity(
    manifest: AdminManifest,
    *,
    require_config_commit: bool = True,
    expected_kernel_hash: str | None = None,
) -> dict[str, Any]:
    state = _repo_state(manifest)
    if state["kernel_hash"] != state["kernel_blob_hash"]:
        raise ControlledRuntimeError("working kernel.py differs from HEAD Git blob")
    base = _load_base(manifest)
    pin = base.get("expected_kernel_hash")
    if not isinstance(pin, str):
        raise ControlledRuntimeError("base config has no expected_kernel_hash")
    required_hash = expected_kernel_hash or pin
    if state["kernel_hash"] != required_hash:
        raise ControlledRuntimeError("kernel.py does not match the pinned baseline")
    if require_config_commit and base.get("expected_git_commit") != state["head"]:
        raise ControlledRuntimeError("base config commit does not match repository HEAD")
    config = ControllerConfig.load(manifest.pro_config)
    pin_path = _deployment_pin_path(config)
    deployment_pin = (
        None
        if not pin_path.exists() and not pin_path.is_symlink()
        else DeploymentBaselinePin.load(pin_path)
    )
    if deployment_pin is None:
        best, artifact = _artifact_for_best(config)
    else:
        selected_confirmation = _artifact_for_pinned_confirmation(
            config, deployment_pin
        )
        proof = _prove_adoption(
            config,
            candidate_hash=required_hash,
            namespace_id=deployment_pin.namespace_id,
            selected_confirmation=selected_confirmation,
        )
        if (
            proof.confirmation.id
            != deployment_pin.confirmation_experiment_id
            or proof.confirmation.experiment_uid
            != deployment_pin.confirmation_experiment_uid
            or proof.confirmation.artifact_id
            != str(deployment_pin.baseline_ref.artifact_id)
            or (
                proof.primary is None
                and deployment_pin.primary_experiment_uid is not None
            )
            or (
                proof.primary is not None
                and proof.primary.experiment_uid
                != deployment_pin.primary_experiment_uid
            )
            or proof.execution_environment.is_resolved
            != deployment_pin.execution_environment.is_resolved
            or (
                proof.execution_environment.is_resolved
                and proof.execution_environment
                != deployment_pin.execution_environment
            )
            or proof.parent_baseline != deployment_pin.parent_baseline_ref
        ):
            raise ControlledRuntimeError(
                "deployment baseline pin differs from its immutable History proof"
            )
        best = proof.confirmation
        artifact = proof.artifact
    if best.candidate_hash != required_hash:
        raise ControlledRuntimeError("accepted History baseline does not match kernel.py")
    return {
        **state,
        "baseline_experiment_id": best.id,
        "baseline_hash": best.candidate_hash,
        "baseline_artifact": str(artifact),
    }


def _write_report(manifest: AdminManifest, command: str, report: Mapping[str, Any]) -> Path:
    report_dir = manifest.runtime_root / "admin-reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = report_dir / f"{stamp}-{command}-{uuid.uuid4().hex[:8]}.json"
    _atomic_publish({path: (_json_bytes(report), 0o600)})
    return path


def _active_run_error(config: ControllerConfig) -> str | None:
    database = config.controller_dir / "controller.sqlite3"
    if not database.is_file():
        return None
    with ControllerStore(database) as store:
        latest = store.latest_run()
    if latest is not None and latest["status"] not in TERMINAL_RUN_STATUSES:
        return f"controller run {latest['id']} is still {latest['status']}"
    return None


def _active_campaign_error(runtime_root: Path) -> str | None:
    """Reject deployment mutation while any campaign can still resume.

    The check is read-only and uses the sole production
    ``campaign/campaign.sqlite3`` convention.  A malformed campaign database
    is itself a data-integrity reason to fail closed.
    """

    candidates = (conventional_campaign_database(runtime_root),)
    active: list[tuple[str, str]] = []
    for database in candidates:
        try:
            _reject_symlink_path(database)
        except ValueError as exc:
            raise ControlledRuntimeError(
                "campaign database path failed the non-symlink boundary"
            ) from exc
        if database.exists() and not database.is_file():
            raise ControlledRuntimeError(
                f"campaign database {database} is not a regular file"
            )
        if not database.is_file():
            continue
        try:
            connection = sqlite3.connect(
                database.resolve().as_uri() + "?mode=ro", uri=True
            )
            connection.row_factory = sqlite3.Row
            try:
                version = int(
                    connection.execute("PRAGMA user_version").fetchone()[0]
                )
                if version != 1:
                    raise ControlledRuntimeError(
                        f"campaign database {database} has unsupported schema {version}"
                    )
                rows = connection.execute(
                    """
                    SELECT id, status FROM campaigns
                    WHERE status NOT IN ('COMPLETED', 'CANCELLED')
                    ORDER BY created_at, id
                    """
                ).fetchall()
            finally:
                connection.close()
        except sqlite3.DatabaseError as exc:
            raise ControlledRuntimeError(
                f"campaign database {database} failed integrity/read checks: {exc}"
            ) from exc
        active.extend((str(row["id"]), str(row["status"])) for row in rows)
    if active:
        rendered = ", ".join(f"{campaign_id}={status}" for campaign_id, status in active)
        return "deployment mutation is forbidden while campaigns are active: " + rendered
    return None


def _require_no_active_campaign(runtime_root: Path) -> None:
    error = _active_campaign_error(runtime_root)
    if error:
        raise ControlledRuntimeError(error)


@contextmanager
def _deployment_mutation_guard(manifest: AdminManifest) -> Iterator[int]:
    """Hold the canonical Campaign fence, then the existing GPU lock.

    Every deployment publisher uses this exact ordering.  Campaign lifecycle
    commands take only the first lock, so they cannot introduce a resumable
    Campaign between the initial guard and the final publication check.
    """

    with campaign_maintenance_fence(manifest.runtime_root) as descriptor:
        with gpu_lock(manifest.runtime_root / "controller" / "gpu1.lock"):
            _require_no_active_campaign(manifest.runtime_root)
            yield descriptor


def _active_containers(config: ControllerConfig) -> list[str]:
    completed = _run(
        [
            str(config.docker_binary),
            "ps",
            "-a",
            "--filter",
            "label=kernel-autoresearch.project=fused-moe",
            "--format",
            "{{.Names}}",
        ],
        check=True,
        timeout=30,
    )
    names = [line for line in completed.stdout.splitlines() if line]
    legacy = _run(
        [
            str(config.docker_binary),
            "ps",
            "-a",
            "--filter",
            "name=kar-",
            "--format",
            "{{.Names}}",
        ],
        check=True,
        timeout=30,
    )
    return sorted(set(names + [line for line in legacy.stdout.splitlines() if line]))


def _static_checks(manifest: AdminManifest) -> dict[str, Any]:
    identity = _identity(manifest)
    if not manifest.host_python.is_file() or not os.access(
        manifest.host_python, os.X_OK
    ):
        raise ControlledRuntimeError("manifest host_python is not executable")
    configs = {
        name: ControllerConfig.load(path)
        for name, path in _config_targets(manifest).items()
    }
    base = _load_base(manifest)
    values = _render_config_values(
        base,
        commit=identity["head"],
        kernel_hash=identity["kernel_hash"],
    )
    _validate_generated(values)
    for name, path in _config_targets(manifest).items():
        actual = _read_json(path, f"{name} config")
        if actual != values[name]:
            raise ControlledRuntimeError(f"generated {name} config has drifted")
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise ControlledRuntimeError(f"{name} config mode must be 0600")
    expected_environment = _env_bytes(
        manifest,
        identity["head"],
        identity["kernel_hash"],
        str(values["base"]["framework_git_commit"]),
    )
    if manifest.environment_file.read_bytes() != expected_environment:
        raise ControlledRuntimeError("generated environment file has drifted")
    if stat.S_IMODE(manifest.environment_file.stat().st_mode) != 0o600:
        raise ControlledRuntimeError("environment file mode must be 0600")
    for config in configs.values():
        errors = config.validate_host(require_secret=True)
        if errors:
            raise ControlledRuntimeError("host validation failed: " + "; ".join(errors))
    images: dict[str, Any] = {}
    primary = configs["pro"]
    for role, image in (
        ("proposer", primary.proposer_image),
        ("evaluator", primary.evaluator_image),
    ):
        result = _run(
            [str(primary.docker_binary), "image", "inspect", image],
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            raise ControlledRuntimeError(f"pinned {role} image is not present locally")
        images[role] = image
    generated_models: dict[str, str] = {}
    generated_model_settings: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory(prefix="kar-admin-opencode-") as temporary:
        for name in ("pro", "flash"):
            path = Path(temporary) / f"{name}.json"
            write_opencode_config(path, configs[name])
            value = json.loads(path.read_text(encoding="utf-8"))
            model = configs[name].opencode_model
            spec = OPENCODE_MODEL_SPECS[model]
            if value["model"] != model:
                raise ControlledRuntimeError("OpenCode root model identity mismatch")
            if value["agent"]["kernel-proposer"]["model"] != model:
                raise ControlledRuntimeError("OpenCode agent model identity mismatch")
            if list(value["provider"]["deepseek"]["models"]) != [spec.provider_id]:
                raise ControlledRuntimeError("OpenCode provider model identity mismatch")
            provider_model = value["provider"]["deepseek"]["models"][
                spec.provider_id
            ]
            if provider_model["limit"] != {
                "context": spec.context_tokens,
                "output": spec.output_tokens,
            }:
                raise ControlledRuntimeError("OpenCode model limits mismatch")
            agent = value["agent"]["kernel-proposer"]
            if (
                value["permission"] != {"*": "deny"}
                or agent["permission"] != {"*": "deny"}
                or agent["steps"] != OPENCODE_PROPOSER_STEPS
                or agent["reasoningEffort"] != spec.reasoning_effort
                or agent["thinking"] != {"type": "enabled"}
                or not value["tools"]
                or any(value["tools"].values())
            ):
                raise ControlledRuntimeError("OpenCode proposer boundary mismatch")
            argv = proposer_argv(
                configs[name],
                name=f"kar-admin-{name}",
                run_id=f"admin-{name}",
                opencode_config=path,
            )
            docker_env = [
                argv[index + 1]
                for index, item in enumerate(argv[:-1])
                if item == "--env"
            ]
            output_cap_env = [
                item
                for item in docker_env
                if item.startswith(f"{OPENCODE_OUTPUT_TOKEN_MAX_ENV}=")
            ]
            expected_output_cap_env = (
                f"{OPENCODE_OUTPUT_TOKEN_MAX_ENV}="
                f"{spec.request_output_token_cap}"
            )
            if output_cap_env != [expected_output_cap_env]:
                raise ControlledRuntimeError(
                    "OpenCode request output token cap mismatch"
                )
            if argv[argv.index("--model") + 1] != model:
                raise ControlledRuntimeError(
                    "OpenCode Docker model identity mismatch"
                )
            generated_models[name] = model
            generated_model_settings[name] = {
                "model": model,
                "reasoning_effort": spec.reasoning_effort,
                "output_tokens": spec.output_tokens,
                "declared_output_tokens": spec.output_tokens,
                "request_output_token_cap": (
                    spec.request_output_token_cap
                ),
            }
    active = _active_run_error(primary)
    if active:
        raise ControlledRuntimeError(active)
    containers = _active_containers(primary)
    if containers:
        raise ControlledRuntimeError(
            "project containers still exist: " + ", ".join(containers)
        )
    return {
        "identity": identity,
        "models": generated_models,
        "model_settings": generated_model_settings,
        "images": images,
        "secret": {
            "path": str(primary.deepseek_key_file),
            "owner_uid": primary.deepseek_key_file.stat().st_uid,
            "mode": oct(stat.S_IMODE(primary.deepseek_key_file.stat().st_mode)),
            "size_bytes": primary.deepseek_key_file.stat().st_size,
        },
        "containers": [],
    }


def _cpu_tests(config: ControllerConfig, *, run_tag: str) -> dict[str, Any]:
    name = f"kar-admin-cpu-{run_tag}"
    argv = [
        str(config.docker_binary),
        "run",
        "--rm",
        "--name",
        name,
        "--label",
        "kernel-autoresearch.project=fused-moe",
        "--init",
        "--network",
        "none",
        "--read-only",
        "--user",
        f"{config.container_uid}:{config.container_gid}",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "256",
        "--memory",
        "4g",
        "--cpus",
        "4",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=1g,mode=1777",
        "--mount",
        f"type=bind,src={config.repository_dir},dst=/workspace,readonly",
        "--env",
        "HOME=/tmp",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--workdir",
        "/workspace",
        "--entrypoint",
        "/opt/conda/bin/python",
        config.evaluator_image,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-v",
    ]
    result = CommandRunner().run(
        argv,
        input_text=None,
        timeout_sec=ADMIN_CPU_TIMEOUT_SEC,
        max_output_bytes=ADMIN_REPORT_LIMIT,
        container_name=name,
        docker_binary=config.docker_binary,
    )
    if result.timed_out:
        raise ControlledRuntimeError("fixed evaluator CPU test timed out")
    if result.output_limited:
        raise ControlledRuntimeError("fixed evaluator CPU test output exceeded limit")
    if result.returncode != 0:
        tail = (result.stdout + "\n" + result.stderr)[-8000:]
        raise ControlledRuntimeError("fixed evaluator CPU tests failed: " + tail)
    return {
        "status": "PASSED",
        "image": config.evaluator_image,
        "returncode": result.returncode,
    }


def _doctor_configs(
    configs: Mapping[str, ControllerConfig],
    *,
    deployment_pin: DeploymentBaselinePin | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name in ("pro", "flash"):
        payload = ResearchController(
            configs[name], _deployment_pin_override=deployment_pin
        ).doctor()
        if payload.get("status") != "SUCCESS":
            raise ControlledRuntimeError(
                f"{name} controller doctor failed: "
                + "; ".join(payload.get("errors", []))
            )
        if payload.get("proposer_model") != configs[name].opencode_model:
            raise ControlledRuntimeError(f"{name} doctor model identity mismatch")
        spec = OPENCODE_MODEL_SPECS[configs[name].opencode_model]
        if payload.get("proposer_reasoning_effort") != spec.reasoning_effort:
            raise ControlledRuntimeError(
                f"{name} doctor reasoning effort identity mismatch"
            )
        if payload.get("proposer_declared_output_tokens") != spec.output_tokens:
            raise ControlledRuntimeError(
                f"{name} doctor declared output identity mismatch"
            )
        if (
            payload.get("proposer_request_output_token_cap")
            != spec.request_output_token_cap
        ):
            raise ControlledRuntimeError(
                f"{name} doctor request output cap identity mismatch"
            )
        probe = payload.get("c500_probe") or {}
        if probe.get("environment", {}).get("compile_probe_status") != "PASSED":
            raise ControlledRuntimeError(f"{name} C500 compile probe did not pass")
        results[name] = {
            "status": payload["status"],
            "proposer_model": payload["proposer_model"],
            "proposer_reasoning_effort": payload[
                "proposer_reasoning_effort"
            ],
            "proposer_declared_output_tokens": payload[
                "proposer_declared_output_tokens"
            ],
            "proposer_request_output_token_cap": payload[
                "proposer_request_output_token_cap"
            ],
            "environment": probe.get("environment"),
        }
    return results


def bootstrap(manifest_path: Path, pro_path: Path, flash_path: Path) -> dict[str, Any]:
    manifest_path = _canonical_path(str(manifest_path), "manifest")
    pro_path = _canonical_path(str(pro_path), "pro_config")
    flash_path = _canonical_path(str(flash_path), "flash_config")
    if manifest_path.exists() or manifest_path.is_symlink():
        raise ValueError("bootstrap refuses to overwrite an existing manifest")
    pro_value = _read_json(pro_path, "Pro config")
    flash_value = _read_json(flash_path, "Flash config")
    pro_config = ControllerConfig.load(pro_path)
    flash_config = ControllerConfig.load(flash_path)
    if pro_config.opencode_model != PRO_MODEL or flash_config.opencode_model != FLASH_MODEL:
        raise ValueError("bootstrap requires one Pro config and one Flash config")
    without_model_pro = dict(pro_value)
    without_model_flash = dict(flash_value)
    without_model_pro.pop("opencode_model", None)
    without_model_flash.pop("opencode_model", None)
    if without_model_pro != without_model_flash:
        changed = sorted(
            key
            for key in set(without_model_pro) | set(without_model_flash)
            if without_model_pro.get(key) != without_model_flash.get(key)
        )
        raise ValueError(
            "Pro and Flash configs differ outside opencode_model: " + ", ".join(changed)
        )
    repository = pro_config.repository_dir
    runtime_root = pro_config.controller_dir.parent
    if manifest_path != runtime_root / "admin.json":
        raise ValueError("bootstrap manifest must be runtime_root/admin.json")
    if flash_config.repository_dir != repository or flash_config.controller_dir.parent != runtime_root:
        raise ValueError("Pro and Flash configs do not share repository/runtime root")
    branch = _git(repository, "symbolic-ref", "--short", "HEAD")
    upstream = _git(
        repository, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"
    )
    manifest = AdminManifest(
        repository_dir=repository,
        runtime_root=runtime_root,
        base_config=runtime_root / "autorun.base.json",
        pro_config=pro_path,
        flash_config=flash_path,
        pro_canary_config=runtime_root / "autorun.pro.canary.json",
        flash_canary_config=runtime_root / "autorun.flash.canary.json",
        environment_file=runtime_root / "env.sh",
        expected_branch=branch,
        expected_upstream=upstream,
        host_python=Path(sys.executable).resolve(),
    )
    repository_state = _repo_state(manifest)
    if repository_state["kernel_hash"] != repository_state["kernel_blob_hash"]:
        raise ControlledRuntimeError("working kernel.py differs from HEAD Git blob")
    if repository_state["kernel_hash"] != pro_config.expected_kernel_hash:
        raise ControlledRuntimeError(
            "bootstrap input kernel pin does not match current kernel.py"
        )
    best, _ = _artifact_for_best(pro_config)
    if best.candidate_hash != repository_state["kernel_hash"]:
        raise ControlledRuntimeError(
            "bootstrap accepted History baseline does not match kernel.py"
        )
    for path in (
        manifest.base_config,
        manifest.pro_canary_config,
        manifest.flash_canary_config,
        manifest.environment_file,
    ):
        if path.exists() or path.is_symlink():
            raise ValueError(f"bootstrap refuses to overwrite existing generated file: {path}")
    base = dict(without_model_pro)
    base["max_candidates"] = 5
    values = _render_config_values(
        base,
        commit=repository_state["head"],
        kernel_hash=pro_config.expected_kernel_hash,
    )
    _validate_generated(values)
    files, hashes = _prepared_files_from_base(manifest, values)
    files[manifest_path] = (_json_bytes(manifest.to_dict()), 0o600)
    _atomic_publish(files)
    loaded = AdminManifest.load(manifest_path)
    identity = _identity(loaded)
    report = {
        "schema_version": 1,
        "command": "bootstrap",
        "status": "SUCCESS",
        "manifest": str(manifest_path),
        "identity": identity,
        "config_sha256": hashes,
    }
    _write_report(loaded, "bootstrap", report)
    return report


def _prepared_files_from_base(
    manifest: AdminManifest, values: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[Path, tuple[bytes, int]], dict[str, str]]:
    files = {
        path: (_json_bytes(values[name]), 0o600)
        for name, path in _config_targets(manifest).items()
    }
    commit = str(values["base"]["expected_git_commit"])
    framework_commit = str(values["base"]["framework_git_commit"])
    kernel_hash = str(values["base"]["expected_kernel_hash"])
    files[manifest.environment_file] = (
        _env_bytes(manifest, commit, kernel_hash, framework_commit),
        0o600,
    )
    hashes = {name: _sha256_bytes(_json_bytes(value)) for name, value in values.items()}
    hashes["environment"] = _sha256_bytes(files[manifest.environment_file][0])
    return files, hashes


def _sync_locked(manifest: AdminManifest, *, publish: bool) -> dict[str, Any]:
    identity = _identity(manifest, require_config_commit=False)
    current_config = ControllerConfig.load(manifest.pro_config)
    active = _active_run_error(current_config)
    if active:
        raise ControlledRuntimeError(active)
    containers = _active_containers(current_config)
    if containers:
        raise ControlledRuntimeError(
            "project containers still exist: " + ", ".join(containers)
        )
    base = _load_base(manifest)
    pin = str(base["expected_kernel_hash"])
    previous_commit = base.get("expected_git_commit")
    values = _render_config_values(base, commit=identity["head"], kernel_hash=pin)
    _validate_generated(values)
    files, hashes = _prepared_files_from_base(manifest, values)
    deployment_pin = _deployment_pin_for_commit(
        current_config,
        git_commit=identity["head"],
        framework_git_commit=str(values["base"]["framework_git_commit"]),
    )
    _add_deployment_pin_publish(
        files,
        hashes,
        manifest=manifest,
        pin=deployment_pin,
    )
    changed = [
        str(path)
        for path, (data, _) in files.items()
        if not path.exists() or path.read_bytes() != data
    ]
    if publish:
        _require_no_active_campaign(manifest.runtime_root)
        _atomic_publish(files)
    return {
        "schema_version": 1,
        "command": "sync",
        "status": "SUCCESS",
        "published": publish,
        "identity": identity,
        "changed_files": changed,
        "changed_fields": (
            {}
            if previous_commit == identity["head"]
            else {
                "expected_git_commit": {
                    "old": previous_commit,
                    "new": identity["head"],
                }
            }
        ),
        "config_sha256": hashes,
        "models": {"pro": PRO_MODEL, "flash": FLASH_MODEL},
    }


def sync(manifest: AdminManifest, *, publish: bool = True) -> dict[str, Any]:
    with _deployment_mutation_guard(manifest):
        return _sync_locked(manifest, publish=publish)


def verify(manifest: AdminManifest, level: str) -> dict[str, Any]:
    if level not in {"static", "cpu", "doctor"}:
        raise ValueError("verify level must be static, cpu or doctor")
    with gpu_lock(manifest.runtime_root / "controller" / "gpu1.lock"):
        static = _static_checks(manifest)
        payload: dict[str, Any] = {
            "schema_version": 1,
            "command": "verify",
            "level": level,
            "status": "SUCCESS",
            "static": static,
        }
        configs = {
            name: ControllerConfig.load(path)
            for name, path in _config_targets(manifest).items()
        }
        if level in {"cpu", "doctor"}:
            payload["cpu"] = _cpu_tests(configs["pro"], run_tag=uuid.uuid4().hex[:12])
        if level == "doctor":
            payload["doctor"] = _doctor_configs(configs)
    _write_report(manifest, "verify", payload)
    return payload


def _post_update_locked(
    manifest: AdminManifest, *, doctor: bool
) -> dict[str, Any]:
    identity = _identity(manifest, require_config_commit=False)
    base = _load_base(manifest)
    pin = str(base["expected_kernel_hash"])
    values = _render_config_values(base, commit=identity["head"], kernel_hash=pin)
    _validate_generated(values)
    files, hashes = _prepared_files_from_base(manifest, values)
    pro = ControllerConfig.load(manifest.pro_config)
    deployment_pin = _deployment_pin_for_commit(
        pro,
        git_commit=identity["head"],
        framework_git_commit=str(values["base"]["framework_git_commit"]),
    )
    _add_deployment_pin_publish(
        files,
        hashes,
        manifest=manifest,
        pin=deployment_pin,
    )
    cpu = _cpu_tests(pro, run_tag=uuid.uuid4().hex[:12])
    _run(
        [
            str(manifest.host_python),
            "-m",
            "pip",
            "install",
            "--user",
            "--no-deps",
            "-e",
            str(manifest.repository_dir),
        ],
        cwd=manifest.repository_dir,
        timeout=600,
    )
    doctor_payload = None
    if doctor:
        with tempfile.TemporaryDirectory(
            prefix="kar-admin-doctor-", dir=manifest.runtime_root
        ) as temporary:
            temporary_root = Path(temporary)
            configs: dict[str, ControllerConfig] = {}
            for name in ("pro", "flash"):
                path = temporary_root / f"{name}.json"
                path.write_bytes(_json_bytes(values[name]))
                configs[name] = ControllerConfig.load(path)
            doctor_payload = _doctor_configs(
                configs, deployment_pin=deployment_pin
            )
    _require_no_active_campaign(manifest.runtime_root)
    _atomic_publish(files)
    report = {
        "schema_version": 1,
        "command": "post-update",
        "status": "SUCCESS",
        "identity": identity,
        "cpu": cpu,
        "doctor": doctor_payload,
        "config_sha256": hashes,
    }
    _write_report(manifest, "post-update", report)
    return report


def _post_update(
    manifest: AdminManifest,
    *,
    doctor: bool,
    maintenance_lock_fd: int | None = None,
) -> dict[str, Any]:
    """Validate and publish an update only inside the maintenance fence.

    Normal direct use acquires both locks itself.  ``update`` passes its
    already-locked descriptor into the post-update child so the parent keeps
    one uninterrupted fence across the Git fast-forward and config/pin
    publication without making the child recursively wait on that same lock.
    """

    if maintenance_lock_fd is None:
        with _deployment_mutation_guard(manifest):
            return _post_update_locked(manifest, doctor=doctor)
    verify_inherited_campaign_maintenance_fence(
        manifest.runtime_root, maintenance_lock_fd
    )
    _require_no_active_campaign(manifest.runtime_root)
    return _post_update_locked(manifest, doctor=doctor)


def update(manifest: AdminManifest, *, doctor: bool) -> dict[str, Any]:
    with _deployment_mutation_guard(manifest) as maintenance_lock_fd:
        state = _repo_state(manifest)
        # Permit resuming post-update after Git has already advanced, but still
        # prove that the pinned kernel and accepted History artifact agree
        # before any network or Git mutation.
        _identity(manifest, require_config_commit=False)
        current_config = ControllerConfig.load(manifest.pro_config)
        active = _active_run_error(current_config)
        if active:
            raise ControlledRuntimeError(active)
        containers = _active_containers(current_config)
        if containers:
            raise ControlledRuntimeError(
                "project containers still exist: " + ", ".join(containers)
            )
        remote, separator, branch = manifest.expected_upstream.partition("/")
        if not separator or not remote or not branch:
            raise ValueError("expected_upstream must be a remote/branch tracking ref")
        _git(manifest.repository_dir, "fetch", "--no-tags", remote, branch)
        ancestor_check = _run(
            ["git", "merge-base", "--is-ancestor", state["head"], "FETCH_HEAD"],
            cwd=manifest.repository_dir,
            check=False,
        )
        if ancestor_check.returncode != 0:
            raise ControlledRuntimeError("upstream update is not a fast-forward descendant")
        _require_no_active_campaign(manifest.runtime_root)
        _git(manifest.repository_dir, "merge", "--ff-only", "FETCH_HEAD")
        environment = dict(os.environ)
        current_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(manifest.repository_dir) + (
            os.pathsep + current_pythonpath if current_pythonpath else ""
        )
        argv = [
            str(manifest.host_python),
            "-m",
            "kernel_research.autorun.admin",
            "_post-update",
            "--manifest",
            str(manifest.runtime_root / "admin.json"),
            "--maintenance-lock-fd",
            str(maintenance_lock_fd),
        ]
        if doctor:
            argv.append("--doctor")
        completed = _run(
            argv,
            cwd=manifest.repository_dir,
            timeout=2400,
            check=False,
            env=environment,
            pass_fds=(maintenance_lock_fd,),
        )
        if completed.returncode != 0:
            raise ControlledRuntimeError(
                "post-update validation failed; formal configs remain pinned: "
                + (completed.stderr.strip() or completed.stdout.strip())
            )
        try:
            child = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ControlledRuntimeError("post-update child emitted invalid JSON") from exc
        new_commit = _git(manifest.repository_dir, "rev-parse", "HEAD")
    report = {
        "schema_version": 1,
        "command": "update",
        "status": "SUCCESS",
        "old_commit": state["head"],
        "new_commit": new_commit,
        "post_update": child,
    }
    _write_report(manifest, "update", report)
    return report


def _require_candidate_only_commit(
    repository: Path, *, deployed_commit: str, candidate_commit: str
) -> None:
    """Prove the operator commit is one direct, kernel-only transition."""

    if candidate_commit == deployed_commit:
        return
    ancestry = _git(
        repository, "rev-list", "--parents", "-n", "1", candidate_commit
    ).split()
    if len(ancestry) != 2 or ancestry[1] != deployed_commit:
        raise ControlledRuntimeError(
            "candidate commit must directly descend from the deployed commit"
        )
    changed = _git(
        repository,
        "diff",
        "--name-only",
        "--no-renames",
        deployed_commit,
        candidate_commit,
    ).splitlines()
    if changed != ["kernel.py"]:
        raise ControlledRuntimeError(
            "candidate commit must change exactly kernel.py"
        )


def adopt_baseline(
    manifest: AdminManifest,
    *,
    candidate_hash: str,
    doctor: bool,
    namespace_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(candidate_hash, str) or len(candidate_hash) != 64 or any(
        character not in "0123456789abcdef" for character in candidate_hash
    ):
        raise ValueError("candidate hash must be 64 lowercase hex digits")
    with _deployment_mutation_guard(manifest):
        state = _repo_state(manifest)
        if state["kernel_hash"] != candidate_hash or state["kernel_blob_hash"] != candidate_hash:
            raise ControlledRuntimeError("candidate hash must match kernel.py and HEAD blob")
        current = ControllerConfig.load(manifest.pro_config)
        _require_candidate_only_commit(
            manifest.repository_dir,
            deployed_commit=current.expected_git_commit,
            candidate_commit=state["head"],
        )
        active = _active_run_error(current)
        if active:
            raise ControlledRuntimeError(active)
        containers = _active_containers(current)
        if containers:
            raise ControlledRuntimeError(
                "project containers still exist: " + ", ".join(containers)
            )
        proof = _prove_adoption(
            current,
            candidate_hash=candidate_hash,
            namespace_id=namespace_id,
        )
        best = proof.confirmation
        artifact = proof.artifact
        base = _load_base(manifest)
        values = _render_config_values(
            base, commit=state["head"], kernel_hash=candidate_hash
        )
        _validate_generated(values)
        if proof.execution_environment.is_resolved:
            with tempfile.TemporaryDirectory(
                prefix="kar-admin-target-", dir=manifest.runtime_root
            ) as temporary:
                target_path = Path(temporary) / "pro.json"
                target_path.write_bytes(_json_bytes(values["pro"]))
                target_config = ControllerConfig.load(target_path)
            target_environment = ResearchController(
                target_config
            )._resolved_execution_environment(proof.namespace)
            try:
                proof.execution_environment.require_match(
                    target_environment,
                    context="adoption target",
                )
            except ValueError as exc:
                raise ControlledRuntimeError(str(exc)) from exc
        files, hashes = _prepared_files_from_base(manifest, values)
        baseline_ref = BaselineRef.create(
            namespace=proof.namespace,
            artifact_id=best.artifact_id,
            source="deployment",
            revision=f"history-{best.id}",
            execution_environment=(
                proof.execution_environment
                if proof.execution_environment.is_resolved
                else None
            ),
        )
        deployment_pin = DeploymentBaselinePin.create(
            namespace_id=proof.namespace.namespace_id,
            baseline_ref=baseline_ref,
            candidate_hash=candidate_hash,
            git_commit=state["head"],
            primary_experiment_uid=(
                None if proof.primary is None else proof.primary.experiment_uid
            ),
            confirmation_experiment_uid=best.experiment_uid,
            confirmation_experiment_id=best.id,
            parent_baseline_ref=proof.parent_baseline,
            execution_environment=baseline_ref.execution_environment,
        )
        pin_path = manifest.runtime_root / DEPLOYMENT_BASELINE_FILENAME
        files[pin_path] = (_json_bytes(deployment_pin.to_dict()), 0o600)
        hashes["deployment_baseline"] = _sha256_bytes(files[pin_path][0])
        doctor_payload = None
        if doctor:
            with tempfile.TemporaryDirectory(
                prefix="kar-admin-adopt-", dir=manifest.runtime_root
            ) as temporary:
                configs: dict[str, ControllerConfig] = {}
                for name in ("pro", "flash"):
                    path = Path(temporary) / f"{name}.json"
                    path.write_bytes(_json_bytes(values[name]))
                    configs[name] = ControllerConfig.load(path)
                doctor_payload = _doctor_configs(
                    configs, deployment_pin=deployment_pin
                )
        _require_no_active_campaign(manifest.runtime_root)
        _atomic_publish(files)
    report = {
        "schema_version": 1,
        "command": "adopt-baseline",
        "status": "SUCCESS",
        "candidate_hash": candidate_hash,
        "deployment_git_commit": state["head"],
        "framework_git_commit": str(
            values["base"]["framework_git_commit"]
        ),
        "namespace_id": proof.namespace.namespace_id,
        "experiment_id": best.id,
        "artifact": str(artifact),
        "baseline_ref": baseline_ref.to_dict(),
        "evidence_digest": deployment_pin.evidence_digest,
        "doctor": doctor_payload,
        "config_sha256": hashes,
    }
    _write_report(manifest, "adopt-baseline", report)
    return report


def requalify_adoption(
    manifest: AdminManifest,
    *,
    candidate_hash: str,
    candidate_path: str | Path,
    namespace_id: str,
) -> dict[str, Any]:
    """Produce fresh resolved evidence without publishing deployment state."""

    with campaign_maintenance_fence(manifest.runtime_root):
        _require_no_active_campaign(manifest.runtime_root)
        config = ControllerConfig.load(manifest.pro_config)
        active = _active_run_error(config)
        if active:
            raise ControlledRuntimeError(active)
        containers = _active_containers(config)
        if containers:
            raise ControlledRuntimeError(
                "project containers still exist: " + ", ".join(containers)
            )
        result = ResearchController(config).requalify_candidate_for_adoption(
            candidate_path=candidate_path,
            candidate_hash=candidate_hash,
            namespace_id=namespace_id,
        )
        _require_no_active_campaign(manifest.runtime_root)
    report = {
        "schema_version": 1,
        "command": "requalify-adoption",
        **result,
    }
    _write_report(manifest, "requalify-adoption", report)
    return report


def bootstrap_current_baseline(
    manifest: AdminManifest,
    *,
    candidate_hash: str,
) -> dict[str, Any]:
    """Produce CURRENT evidence for the exact deployed improvement chain."""

    if not isinstance(candidate_hash, str) or re.fullmatch(
        r"[0-9a-f]{64}", candidate_hash
    ) is None:
        raise ValueError("candidate hash must be 64 lowercase hex digits")
    with campaign_maintenance_fence(manifest.runtime_root):
        _require_no_active_campaign(manifest.runtime_root)
        config = ControllerConfig.load(manifest.pro_config)
        if candidate_hash != config.expected_kernel_hash:
            raise ControlledRuntimeError(
                "CURRENT bootstrap hash must equal the formal deployed kernel hash"
            )
        active = _active_run_error(config)
        if active:
            raise ControlledRuntimeError(active)
        containers = _active_containers(config)
        if containers:
            raise ControlledRuntimeError(
                "project containers still exist: " + ", ".join(containers)
            )
        result = ResearchController(config).bootstrap_current_baseline(
            candidate_path=config.repository_dir / "kernel.py",
            candidate_hash=candidate_hash,
        )
        _require_no_active_campaign(manifest.runtime_root)
    report = {
        "schema_version": 1,
        "command": "bootstrap-current-baseline",
        **result,
    }
    _write_report(manifest, "bootstrap-current-baseline", report)
    return report


def _print(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2))


def _manifest_argument(value: str) -> AdminManifest:
    return AdminManifest.load(_canonical_path(value, "manifest"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kernel-autoresearch-admin",
        description="Maintain and verify a pinned autoresearch server deployment.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    bootstrap_parser = subparsers.add_parser("bootstrap")
    bootstrap_parser.add_argument("--manifest", required=True)
    bootstrap_parser.add_argument("--pro-config", required=True)
    bootstrap_parser.add_argument("--flash-config", required=True)
    for command in (
        "sync",
        "verify",
        "update",
        "requalify-adoption",
        "bootstrap-current-baseline",
        "adopt-baseline",
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--manifest", required=True)
        if command == "verify":
            subparser.add_argument(
                "--level", choices=("static", "cpu", "doctor"), required=True
            )
        if command in {"update", "adopt-baseline"}:
            subparser.add_argument("--doctor", action="store_true")
        if command == "adopt-baseline":
            subparser.add_argument("--candidate-hash", required=True)
            subparser.add_argument(
                "--namespace",
                dest="namespace_id",
                help=(
                    "exact built-in ResearchNamespace namespace_id; omitting "
                    "it uses the one-cycle legacy V1 compatibility path"
                ),
            )
        if command == "requalify-adoption":
            subparser.add_argument("--candidate-hash", required=True)
            subparser.add_argument("--candidate-path", required=True)
            subparser.add_argument("--namespace", dest="namespace_id", required=True)
        if command == "bootstrap-current-baseline":
            subparser.add_argument("--candidate-hash", required=True)
    internal = subparsers.add_parser(
        "_post-update", help="internal locked post-update continuation"
    )
    internal.add_argument("--manifest", required=True)
    internal.add_argument("--doctor", action="store_true")
    internal.add_argument(
        "--maintenance-lock-fd", type=int, help=argparse.SUPPRESS
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "bootstrap":
            payload = bootstrap(
                _canonical_path(args.manifest, "manifest"),
                _canonical_path(args.pro_config, "pro_config"),
                _canonical_path(args.flash_config, "flash_config"),
            )
        else:
            manifest = _manifest_argument(args.manifest)
            if args.command == "sync":
                payload = sync(manifest)
                _write_report(manifest, "sync", payload)
            elif args.command == "verify":
                payload = verify(manifest, args.level)
            elif args.command == "update":
                payload = update(manifest, doctor=args.doctor)
            elif args.command == "adopt-baseline":
                payload = adopt_baseline(
                    manifest,
                    candidate_hash=args.candidate_hash,
                    doctor=args.doctor,
                    namespace_id=args.namespace_id,
                )
            elif args.command == "requalify-adoption":
                payload = requalify_adoption(
                    manifest,
                    candidate_hash=args.candidate_hash,
                    candidate_path=args.candidate_path,
                    namespace_id=args.namespace_id,
                )
            elif args.command == "bootstrap-current-baseline":
                payload = bootstrap_current_baseline(
                    manifest,
                    candidate_hash=args.candidate_hash,
                )
            elif args.command == "_post-update":
                payload = _post_update(
                    manifest,
                    doctor=args.doctor,
                    maintenance_lock_fd=args.maintenance_lock_fd,
                )
            else:
                raise AssertionError("unknown admin command")
        _print(payload)
        return 0
    except (
        OSError,
        ValueError,
        ControlledRuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
