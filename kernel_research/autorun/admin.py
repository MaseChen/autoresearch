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
import shlex
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence
import uuid

from ..constants import OPENCODE_PROPOSER_STEPS
from ..history import HistoryStore
from .controller import ResearchController, gpu_lock
from .errors import ControlledRuntimeError
from .model_catalog import OPENCODE_MODEL_SPECS
from .models import CONFIG_FIELDS, ControllerConfig
from .runtime import CommandRunner, write_opencode_config
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
    normalized["expected_git_commit"] = commit
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


def _env_bytes(manifest: AdminManifest, commit: str, kernel_hash: str) -> bytes:
    exports = {
        "AUTORESEARCH_REPO": str(manifest.repository_dir),
        "AUTORESEARCH_RUNTIME": str(manifest.runtime_root),
        "AUTORESEARCH_MANIFEST": str(manifest.runtime_root / "admin.json"),
        "AUTORESEARCH_PRO_CONFIG": str(manifest.pro_config),
        "AUTORESEARCH_FLASH_CONFIG": str(manifest.flash_config),
        "AUTORESEARCH_PRO_CANARY_CONFIG": str(manifest.pro_canary_config),
        "AUTORESEARCH_FLASH_CANARY_CONFIG": str(manifest.flash_canary_config),
        "AUTORESEARCH_COMMIT": commit,
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


def _artifact_for_best(config: ControllerConfig) -> tuple[Any, Path]:
    with HistoryStore(
        config.state_dir / "history.sqlite3", state_dir=config.state_dir
    ) as history:
        best = history.get_best(backend="c500", suite="full")
    if best is None:
        raise ControlledRuntimeError("no accepted C500 full baseline exists")
    artifact = (config.state_dir / best.artifact_path).resolve(strict=False)
    if not artifact.is_relative_to(config.state_dir.resolve()):
        raise ControlledRuntimeError("accepted artifact escapes state_dir")
    if artifact.is_symlink() or not artifact.is_file():
        raise ControlledRuntimeError("accepted History artifact is missing")
    if _sha256_file(artifact) != best.candidate_hash:
        raise ControlledRuntimeError("accepted History artifact hash mismatch")
    return best, artifact


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
    best, artifact = _artifact_for_best(config)
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
        manifest, identity["head"], identity["kernel_hash"]
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
            generated_models[name] = model
            generated_model_settings[name] = {
                "model": model,
                "reasoning_effort": spec.reasoning_effort,
                "output_tokens": spec.output_tokens,
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


def _doctor_configs(configs: Mapping[str, ControllerConfig]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for name in ("pro", "flash"):
        payload = ResearchController(configs[name]).doctor()
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
        probe = payload.get("c500_probe") or {}
        if probe.get("environment", {}).get("compile_probe_status") != "PASSED":
            raise ControlledRuntimeError(f"{name} C500 compile probe did not pass")
        results[name] = {
            "status": payload["status"],
            "proposer_model": payload["proposer_model"],
            "proposer_reasoning_effort": payload[
                "proposer_reasoning_effort"
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
    kernel_hash = str(values["base"]["expected_kernel_hash"])
    files[manifest.environment_file] = (
        _env_bytes(manifest, commit, kernel_hash),
        0o600,
    )
    hashes = {name: _sha256_bytes(_json_bytes(value)) for name, value in values.items()}
    hashes["environment"] = _sha256_bytes(files[manifest.environment_file][0])
    return files, hashes


def sync(manifest: AdminManifest, *, publish: bool = True) -> dict[str, Any]:
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
    changed = [
        str(path)
        for path, (data, _) in files.items()
        if not path.exists() or path.read_bytes() != data
    ]
    if publish:
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


def _post_update(manifest: AdminManifest, *, doctor: bool) -> dict[str, Any]:
    identity = _identity(manifest, require_config_commit=False)
    base = _load_base(manifest)
    pin = str(base["expected_kernel_hash"])
    values = _render_config_values(base, commit=identity["head"], kernel_hash=pin)
    _validate_generated(values)
    files, hashes = _prepared_files_from_base(manifest, values)
    pro = ControllerConfig.load(manifest.pro_config)
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
            doctor_payload = _doctor_configs(configs)
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


def update(manifest: AdminManifest, *, doctor: bool) -> dict[str, Any]:
    lock_path = manifest.runtime_root / "controller" / "gpu1.lock"
    with gpu_lock(lock_path):
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
        ]
        if doctor:
            argv.append("--doctor")
        completed = _run(
            argv,
            cwd=manifest.repository_dir,
            timeout=2400,
            check=False,
            env=environment,
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
    report = {
        "schema_version": 1,
        "command": "update",
        "status": "SUCCESS",
        "old_commit": state["head"],
        "new_commit": _git(manifest.repository_dir, "rev-parse", "HEAD"),
        "post_update": child,
    }
    _write_report(manifest, "update", report)
    return report


def adopt_baseline(
    manifest: AdminManifest, *, candidate_hash: str, doctor: bool
) -> dict[str, Any]:
    if not isinstance(candidate_hash, str) or len(candidate_hash) != 64 or any(
        character not in "0123456789abcdef" for character in candidate_hash
    ):
        raise ValueError("candidate hash must be 64 lowercase hex digits")
    with gpu_lock(manifest.runtime_root / "controller" / "gpu1.lock"):
        state = _repo_state(manifest)
        if state["kernel_hash"] != candidate_hash or state["kernel_blob_hash"] != candidate_hash:
            raise ControlledRuntimeError("candidate hash must match kernel.py and HEAD blob")
        current = ControllerConfig.load(manifest.pro_config)
        active = _active_run_error(current)
        if active:
            raise ControlledRuntimeError(active)
        containers = _active_containers(current)
        if containers:
            raise ControlledRuntimeError(
                "project containers still exist: " + ", ".join(containers)
            )
        best, artifact = _artifact_for_best(current)
        promotion = best.result.get("promotion", {})
        if (
            best.candidate_hash != candidate_hash
            or not best.promotable
            or not isinstance(promotion, Mapping)
            or promotion.get("phase") != "confirmation"
            or promotion.get("reason") != "promoted"
        ):
            raise ControlledRuntimeError(
                "candidate is not the accepted C500 full confirmation"
            )
        base = _load_base(manifest)
        values = _render_config_values(
            base, commit=state["head"], kernel_hash=candidate_hash
        )
        _validate_generated(values)
        files, hashes = _prepared_files_from_base(manifest, values)
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
                doctor_payload = _doctor_configs(configs)
        _atomic_publish(files)
    report = {
        "schema_version": 1,
        "command": "adopt-baseline",
        "status": "SUCCESS",
        "candidate_hash": candidate_hash,
        "experiment_id": best.id,
        "artifact": str(artifact),
        "doctor": doctor_payload,
        "config_sha256": hashes,
    }
    _write_report(manifest, "adopt-baseline", report)
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
    for command in ("sync", "verify", "update", "adopt-baseline"):
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
    internal = subparsers.add_parser(
        "_post-update", help="internal locked post-update continuation"
    )
    internal.add_argument("--manifest", required=True)
    internal.add_argument("--doctor", action="store_true")
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
                with gpu_lock(manifest.runtime_root / "controller" / "gpu1.lock"):
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
                )
            elif args.command == "_post-update":
                payload = _post_update(manifest, doctor=args.doctor)
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
