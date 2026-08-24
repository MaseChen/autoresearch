"""Durable, fixed-shape Console operation preparation and reconciliation."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time
from typing import Any, Iterator, Mapping

from ..autorun.store import ControllerStore
from ..platform.canonical import canonical_json_bytes, canonical_sha256
from .protocol import (
    OPERATION_KINDS,
    ManualEvaluationRequestV1,
    OperationReceiptV1,
    PreparedOperationV1,
    strict_json_loads,
)
from .read_model import ConsolePaths, ConsoleReadModel


PREPARE_TTL_SECONDS = 300.0
MAX_INTENT_BYTES = 512 * 1024
MAX_CAMPAIGN_CANDIDATES = 1000
MAX_CAMPAIGN_MILLISECONDS = 30 * 24 * 60 * 60 * 1000
MAX_CAMPAIGN_TOKENS = 10**12
MAX_CAMPAIGN_COST_MICROUSD = 10**12
_PHRASES = {
    "RUN_START": "启动自治 Run",
    "RUN_STOP": "停止 Run",
    "RUN_RESUME": "恢复 Run",
    "MANUAL_EVALUATION_START": "启动手工 CURRENT 评测",
    "CAMPAIGN_CREATE": "创建 Campaign",
    "CAMPAIGN_START": "启动 Campaign",
    "CAMPAIGN_PAUSE": "暂停 Campaign",
    "CAMPAIGN_RESUME": "恢复 Campaign",
    "CAMPAIGN_CHILD_EXECUTE": "执行 Campaign Child",
    "CAMPAIGN_LINEAGE_ADVANCE": "推进科学谱系",
    "BENCHMARK_INIT": "初始化 Benchmark",
    "BENCHMARK_EXECUTE": "执行 Benchmark",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _operation_root(paths: ConsolePaths, *, create: bool) -> Path:
    root = paths.controller_dir / "console" / "operations"
    if create:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
    elif not root.exists():
        raise ValueError("Console operation root does not exist")
    if root.is_symlink() or root.resolve(strict=True) != root:
        raise ValueError("Console operation root is not canonical")
    metadata = root.stat(follow_symlinks=False)
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.getuid():
        raise ValueError("Console operation root has an unsafe owner or type")
    os.chmod(root, 0o700)
    return root


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    if len(encoded) > MAX_INTENT_BYTES:
        raise ValueError("Console operation record exceeds its size bound")
    descriptor, temporary = tempfile.mkstemp(prefix=".operation-", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Console operation record is missing or unsafe")
    metadata = path.stat(follow_symlinks=False)
    if stat.S_IMODE(metadata.st_mode) != 0o600 or not 0 < metadata.st_size <= MAX_INTENT_BYTES:
        raise ValueError("Console operation record mode or size is invalid")
    raw = path.read_bytes()
    if len(raw) != metadata.st_size:
        raise ValueError("Console operation record changed while being read")
    value = strict_json_loads(raw, max_bytes=MAX_INTENT_BYTES)
    if not isinstance(value, dict):
        raise ValueError("Console operation record is not an object")
    return value


@contextmanager
def _operation_lock(root: Path, operation_id: str) -> Iterator[None]:
    path = root / f"{operation_id}.lock"
    descriptor = os.open(
        path,
        os.O_CREAT
        | os.O_RDWR
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _manual_run_id(request: ManualEvaluationRequestV1) -> str:
    import uuid

    selected = uuid.uuid5(
        uuid.NAMESPACE_URL,
        "kernel-research/console-manual-evaluation/v1/"
        f"{request.operation_id}/{request.candidate.artifact_id}",
    )
    return "console-manual-" + selected.hex


class ConsoleOperationService:
    """Remote authority for prepare/execute/reconcile over fixed DTOs."""

    def __init__(self, paths: ConsolePaths, model: ConsoleReadModel) -> None:
        if not isinstance(paths, ConsolePaths) or not isinstance(model, ConsoleReadModel):
            raise TypeError("ConsoleOperationService requires trusted paths/model")
        self.paths = paths
        self.model = model
        self.root = paths.controller_dir / "console" / "operations"

    def _path(self, operation_id: str) -> Path:
        import uuid

        try:
            parsed = uuid.UUID(operation_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("operation_id must be a canonical UUID") from exc
        if str(parsed) != operation_id:
            raise ValueError("operation_id must be a canonical lowercase UUID")
        return self.root / f"{operation_id}.json"

    def _validate_parameters(
        self, kind: str, operation_id: str, parameters: object, runtime_digest: str
    ) -> dict[str, Any]:
        if kind == "MANUAL_EVALUATION_START":
            if not isinstance(parameters, Mapping):
                raise ValueError("manual evaluation parameters must be an object")
            value = dict(parameters)
            value.update(
                {
                    "schema_version": 1,
                    "operation_id": operation_id,
                    "runtime_identity_digest": runtime_digest,
                }
            )
            return ManualEvaluationRequestV1.from_value(value).to_dict(
                include_content=True
            )
        return _validate_domain_parameters(kind, parameters)

    def prepare(self, payload: Mapping[str, Any]) -> PreparedOperationV1:
        if set(payload) != {
            "operation_id",
            "kind",
            "runtime_identity_digest",
            "parameters",
        }:
            raise ValueError("prepare payload fields do not match the schema")
        kind = payload["kind"]
        if not isinstance(kind, str) or kind not in OPERATION_KINDS:
            raise ValueError("operation kind is not allowlisted")
        operation_id = str(payload["operation_id"])
        self._path(operation_id)
        identity = self.model.runtime_identity()
        if payload["runtime_identity_digest"] != identity.digest:
            raise ValueError("prepare runtime identity is stale")
        health = self.model.health(deep=True)
        if health.get("status") != "READY":
            raise ValueError("remote runtime is not READY")
        parameters = self._validate_parameters(
            kind, operation_id, payload["parameters"], identity.digest
        )
        material = {
            "schema_version": 1,
            "operation_id": operation_id,
            "kind": kind,
            "runtime_identity_digest": identity.digest,
            "parameters": parameters,
        }
        digest = canonical_sha256(material)
        prepared = PreparedOperationV1(
            operation_id=operation_id,
            kind=kind,
            operation_digest=digest,
            runtime_identity_digest=identity.digest,
            prepared_at=_utc_now(),
            expires_epoch=time.time() + PREPARE_TTL_SECONDS,
            confirmation_phrase=_PHRASES[kind],
            impact=_impact(kind, parameters),
        )
        record = {
            "schema_version": 1,
            "prepared": prepared.to_dict(),
            "parameters": parameters,
            "state": "PREPARED",
            "pid": None,
            "receipt": None,
        }
        self.root = _operation_root(self.paths, create=True)
        path = self._path(operation_id)
        with _operation_lock(self.root, operation_id):
            if path.exists():
                existing = _read_record(path)
                if existing.get("prepared") != prepared.to_dict():
                    persisted = PreparedOperationV1.from_value(existing.get("prepared"))
                    if (
                        persisted.operation_digest != digest
                        or persisted.kind != kind
                        or persisted.runtime_identity_digest != identity.digest
                    ):
                        raise ValueError("operation_id is already bound to another intent")
                    return persisted
            else:
                _atomic_json(path, record)
        return prepared

    def execute(self, payload: Mapping[str, Any]) -> OperationReceiptV1:
        if set(payload) != {
            "operation_id",
            "operation_digest",
            "confirmation_phrase",
        }:
            raise ValueError("execute payload fields do not match the schema")
        operation_id = str(payload["operation_id"])
        self.root = _operation_root(self.paths, create=False)
        path = self._path(operation_id)
        with _operation_lock(self.root, operation_id):
            record = _read_record(path)
            prepared = PreparedOperationV1.from_value(record.get("prepared"))
            if (
                payload["operation_digest"] != prepared.operation_digest
                or payload["confirmation_phrase"] != prepared.confirmation_phrase
            ):
                raise ValueError("operation confirmation identity mismatch")
            if self.model.runtime_identity().digest != prepared.runtime_identity_digest:
                raise ValueError("runtime identity changed after prepare")
            if time.time() > prepared.expires_epoch and record.get("state") == "PREPARED":
                receipt = OperationReceiptV1(
                    operation_id=operation_id,
                    kind=prepared.kind,
                    operation_digest=prepared.operation_digest,
                    status="EXPIRED",
                    observed_at=_utc_now(),
                    domain_identity={},
                    problem={"code": "PREPARE_EXPIRED"},
                )
                record.update({"state": "EXPIRED", "receipt": receipt.to_dict()})
                _atomic_json(path, record)
                return receipt
            if record.get("receipt") is not None:
                return _receipt_from_value(record["receipt"])
            if record.get("state") == "EXECUTING":
                return self._reconcile_record(record, prepared)
            if record.get("state") != "PREPARED":
                raise ValueError("operation is not executable")
            stdout = self.root / f"{operation_id}.stdout"
            stderr = self.root / f"{operation_id}.stderr"
            output_descriptors: list[int] = []
            try:
                for output in (stdout, stderr):
                    descriptor = os.open(
                        output,
                        os.O_CREAT
                        | os.O_EXCL
                        | os.O_WRONLY
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                    )
                    os.fchmod(descriptor, 0o600)
                    output_descriptors.append(descriptor)
                with os.fdopen(
                    output_descriptors.pop(0), "ab", buffering=0
                ) as output_handle, os.fdopen(
                    output_descriptors.pop(0), "ab", buffering=0
                ) as error_handle:
                    process = subprocess.Popen(
                        [
                            str(_host_python(self.paths.manifest_path)),
                            "-m",
                            "kernel_research.console.worker",
                            "--admin-manifest",
                            str(self.paths.manifest_path),
                            "--operation-id",
                            operation_id,
                        ],
                        cwd=self.paths.repository_dir,
                        stdin=subprocess.DEVNULL,
                        stdout=output_handle,
                        stderr=error_handle,
                        close_fds=True,
                        start_new_session=True,
                        shell=False,
                    )
            except BaseException as exc:
                for descriptor in output_descriptors:
                    os.close(descriptor)
                receipt = OperationReceiptV1(
                    operation_id=operation_id,
                    kind=prepared.kind,
                    operation_digest=prepared.operation_digest,
                    status="FAILED",
                    observed_at=_utc_now(),
                    domain_identity=_domain_identity(
                        prepared.kind, record["parameters"]
                    ),
                    problem={
                        "code": "DETACHED_EXECUTOR_START_FAILED",
                        "detail": type(exc).__name__,
                    },
                )
                record.update({"state": "FAILED", "receipt": receipt.to_dict()})
                _atomic_json(path, record)
                return receipt
            record.update({"state": "EXECUTING", "pid": process.pid})
            _atomic_json(path, record)
            return OperationReceiptV1(
                operation_id=operation_id,
                kind=prepared.kind,
                operation_digest=prepared.operation_digest,
                status="EXECUTING",
                observed_at=_utc_now(),
                domain_identity=_domain_identity(prepared.kind, record["parameters"]),
            )

    def reconcile(self, payload: Mapping[str, Any]) -> OperationReceiptV1:
        if set(payload) != {"operation_id"}:
            raise ValueError("reconcile payload fields do not match the schema")
        operation_id = str(payload["operation_id"])
        self.root = _operation_root(self.paths, create=False)
        path = self._path(operation_id)
        with _operation_lock(self.root, operation_id):
            record = _read_record(path)
            prepared = PreparedOperationV1.from_value(record.get("prepared"))
            if record.get("receipt") is not None:
                return _receipt_from_value(record["receipt"])
            return self._reconcile_record(record, prepared)

    def _reconcile_record(
        self, record: dict[str, Any], prepared: PreparedOperationV1
    ) -> OperationReceiptV1:
        if prepared.kind == "MANUAL_EVALUATION_START":
            request = ManualEvaluationRequestV1.from_value(record["parameters"])
            run_id = _manual_run_id(request)
            if self.paths.controller_db.exists():
                with ControllerStore(self.paths.controller_db) as store:
                    try:
                        run = store.get_run(run_id)
                    except ValueError:
                        run = None
                if run is not None and run["status"] in {
                    "PROMOTED",
                    "STOPPED",
                    "FAILED",
                    "HARD_FAILED",
                    "BUDGET_EXHAUSTED",
                }:
                    status = (
                        "SUCCEEDED"
                        if run["status"] in {"PROMOTED", "STOPPED"}
                        else (
                            "UNKNOWN_OUTCOME"
                            if run["status"] == "HARD_FAILED"
                            and "unknown" in str(run.get("stop_reason", "")).lower()
                            else "FAILED"
                        )
                    )
                    receipt = OperationReceiptV1(
                        operation_id=prepared.operation_id,
                        kind=prepared.kind,
                        operation_digest=prepared.operation_digest,
                        status=status,
                        observed_at=_utc_now(),
                        domain_identity={"run_id": run_id},
                        result={"run": run},
                    )
                    record.update({"state": status, "receipt": receipt.to_dict()})
                    _atomic_json(self._path(prepared.operation_id), record)
                    return receipt
        pid = record.get("pid")
        if type(pid) is int and pid > 1:
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = True
            if alive:
                return OperationReceiptV1(
                    operation_id=prepared.operation_id,
                    kind=prepared.kind,
                    operation_digest=prepared.operation_digest,
                    status="EXECUTING",
                    observed_at=_utc_now(),
                    domain_identity=_domain_identity(
                        prepared.kind, record.get("parameters", {})
                    ),
                )
        receipt = OperationReceiptV1(
            operation_id=prepared.operation_id,
            kind=prepared.kind,
            operation_digest=prepared.operation_digest,
            status="UNKNOWN_OUTCOME",
            observed_at=_utc_now(),
            domain_identity=_domain_identity(
                prepared.kind, record.get("parameters", {})
            ),
            problem={"code": "DETACHED_EXECUTOR_OUTCOME_UNKNOWN"},
        )
        record.update({"state": "UNKNOWN_OUTCOME", "receipt": receipt.to_dict()})
        _atomic_json(self._path(prepared.operation_id), record)
        return receipt


def _host_python(manifest_path: Path) -> Path:
    from ..autorun.admin import AdminManifest

    manifest = AdminManifest.load(manifest_path)
    if manifest.host_python.is_symlink() or not manifest.host_python.is_file():
        raise ValueError("Admin host Python is unavailable")
    return manifest.host_python


def _validate_domain_parameters(kind: str, parameters: object) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        raise ValueError("operation parameters must be an object")
    allowed: dict[str, frozenset[str]] = {
        "RUN_START": frozenset({"profile", "proposal_only"}),
        "RUN_STOP": frozenset({"profile", "run_id"}),
        "RUN_RESUME": frozenset({"profile", "run_id"}),
        "CAMPAIGN_CREATE": frozenset({"mode", "profile", "budget"}),
        "CAMPAIGN_START": frozenset({"campaign_id"}),
        "CAMPAIGN_PAUSE": frozenset({"campaign_id", "reason"}),
        "CAMPAIGN_RESUME": frozenset({"campaign_id", "profile"}),
        "CAMPAIGN_CHILD_EXECUTE": frozenset(
            {"campaign_id", "profile", "max_candidates", "max_wall_seconds"}
        ),
        "CAMPAIGN_LINEAGE_ADVANCE": frozenset({"campaign_id", "child_id"}),
        "BENCHMARK_INIT": frozenset({"profile", "arms", "repetitions", "budget"}),
        "BENCHMARK_EXECUTE": frozenset({"campaign_id", "profile"}),
    }
    if kind not in allowed or set(parameters) != allowed[kind]:
        raise ValueError(f"{kind} parameters do not match the fixed DTO")
    normalized = json.loads(canonical_json_bytes(parameters).decode("utf-8"))
    profile = normalized.get("profile")
    if profile is not None and profile not in {"pro", "flash"}:
        raise ValueError("profile must be pro or flash")
    for key in ("run_id", "campaign_id"):
        if key in normalized and (
            not isinstance(normalized[key], str)
            or not normalized[key]
            or len(normalized[key]) > 256
            or not normalized[key].replace("-", "").replace("_", "").isalnum()
        ):
            raise ValueError(f"{key} is not a stable identifier")
    if "proposal_only" in normalized and not isinstance(
        normalized["proposal_only"], bool
    ):
        raise ValueError("proposal_only must be boolean")
    if kind == "CAMPAIGN_CREATE" and normalized["mode"] != "DISCOVERY":
        raise ValueError("Console Campaign creation supports Discovery only")
    if kind in {"CAMPAIGN_CREATE", "BENCHMARK_INIT"}:
        _validate_budget(normalized["budget"])
    if kind == "BENCHMARK_INIT":
        arms = normalized["arms"]
        if (
            not isinstance(arms, list)
            or not 2 <= len(arms) <= 4
            or any(not isinstance(item, str) for item in arms)
            or len(set(arms)) != len(arms)
            or any(item not in {"pro", "flash"} for item in arms)
        ):
            raise ValueError("benchmark arms must be 2-4 unique active profiles")
        repetitions = normalized["repetitions"]
        if type(repetitions) is not int or not 1 <= repetitions <= 100:
            raise ValueError("benchmark repetitions must be between 1 and 100")
    if kind == "CAMPAIGN_CHILD_EXECUTE":
        _bounded_integer(
            normalized["max_candidates"],
            name="max_candidates",
            maximum=MAX_CAMPAIGN_CANDIDATES,
        )
        _bounded_integer(
            normalized["max_wall_seconds"],
            name="max_wall_seconds",
            maximum=MAX_CAMPAIGN_MILLISECONDS // 1000,
        )
    if kind == "CAMPAIGN_PAUSE":
        reason = normalized["reason"]
        if (
            not isinstance(reason, str)
            or not reason
            or len(reason) > 256
            or any(ord(character) < 0x20 for character in reason)
        ):
            raise ValueError("Campaign pause reason must be bounded printable text")
    if "child_id" in normalized:
        _stable_identifier(normalized["child_id"], "child_id")
    return normalized


def _stable_identifier(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 256
        or not value.replace("-", "").replace("_", "").isalnum()
    ):
        raise ValueError(f"{name} is not a stable identifier")
    return value


def _bounded_integer(value: object, *, name: str, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} is outside the Console bound")
    return value


def _validate_budget(value: object) -> None:
    fields = {
        "candidates": MAX_CAMPAIGN_CANDIDATES,
        "wall_ms": MAX_CAMPAIGN_MILLISECONDS,
        "gpu_ms": MAX_CAMPAIGN_MILLISECONDS,
        "tokens": MAX_CAMPAIGN_TOKENS,
        "cost_microusd": MAX_CAMPAIGN_COST_MICROUSD,
    }
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("budget must contain the exact five bounded axes")
    for name, maximum in fields.items():
        _bounded_integer(value[name], name=f"budget {name}", maximum=maximum)


def _impact(kind: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": kind,
        "gpu_possible": kind
        in {
            "RUN_START",
            "RUN_RESUME",
            "MANUAL_EVALUATION_START",
            "CAMPAIGN_RESUME",
            "CAMPAIGN_CHILD_EXECUTE",
            "BENCHMARK_EXECUTE",
        },
        "deployment_authority": False,
        "lineage_authority": kind == "CAMPAIGN_LINEAGE_ADVANCE",
        "unknown_outcome_replay": "FORBIDDEN",
        "parameter_digest": canonical_sha256(parameters),
    }


def _domain_identity(kind: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if kind == "MANUAL_EVALUATION_START":
        request = ManualEvaluationRequestV1.from_value(parameters)
        result["run_id"] = _manual_run_id(request)
        result["candidate_artifact_id"] = str(request.candidate.artifact_id)
    for key in ("run_id", "campaign_id", "child_id"):
        if key in parameters:
            result[key] = parameters[key]
    return result


def _receipt_from_value(value: object) -> OperationReceiptV1:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "operation_id",
        "kind",
        "operation_digest",
        "status",
        "observed_at",
        "domain_identity",
        "result",
        "problem",
    }:
        raise ValueError("operation receipt fields do not match the schema")
    if value["schema_version"] != 1 or type(value["schema_version"]) is not int:
        raise ValueError("operation receipt schema_version must be 1")
    return OperationReceiptV1(
        **{key: value[key] for key in value if key != "schema_version"}
    )


__all__ = ["ConsoleOperationService", "PREPARE_TTL_SECONDS"]
