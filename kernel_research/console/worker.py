"""Detached remote executor for one already-prepared Console operation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import resource
import time
from typing import Any, Mapping

from ..autorun.admin import AdminManifest
from ..autorun.controller import ResearchController
from ..autorun.models import ControllerConfig
from .operations import (
    ConsoleOperationService,
    _atomic_json,
    _operation_lock,
    _operation_root,
    _read_record,
)
from .protocol import ManualEvaluationRequestV1, OperationReceiptV1, PreparedOperationV1
from .read_model import ConsolePaths, ConsoleReadModel


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kernel-autoresearch-console-worker")
    parser.add_argument("--admin-manifest", type=Path, required=True)
    parser.add_argument("--operation-id", required=True)
    return parser


def _observed_at() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _receipt(
    prepared: PreparedOperationV1,
    *,
    status: str,
    domain_identity: Mapping[str, Any],
    result: Mapping[str, Any] | None = None,
    problem: Mapping[str, Any] | None = None,
) -> OperationReceiptV1:
    return OperationReceiptV1(
        operation_id=prepared.operation_id,
        kind=prepared.kind,
        operation_digest=prepared.operation_digest,
        status=status,
        observed_at=_observed_at(),
        domain_identity=domain_identity,
        result=result,
        problem=problem,
    )


def _execute_manual(
    manifest: AdminManifest,
    parameters: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    request = ManualEvaluationRequestV1.from_value(parameters)
    config = ControllerConfig.load(manifest.pro_config)
    result = ResearchController(config).evaluate_manual_candidate(
        candidate=request.candidate,
        operation_id=request.operation_id,
    )
    return {"run_id": result["run_id"]}, result


def _wait_for_parent_publish(path: Path, pid: int) -> dict[str, Any]:
    deadline = time.monotonic() + 5.0
    while True:
        record = _read_record(path)
        if record.get("state") == "EXECUTING" and record.get("pid") == pid:
            return record
        if time.monotonic() >= deadline:
            raise RuntimeError("detached executor identity was not durably published")
        time.sleep(0.01)


def run_worker(manifest_path: Path, operation_id: str) -> int:
    # Bound accidental Python diagnostics even if a future exception handler
    # regresses. Domain/Docker diagnostics remain in their existing private CAS.
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024, 64 * 1024))
    paths = ConsolePaths.from_admin_manifest(manifest_path)
    model = ConsoleReadModel(paths)
    service = ConsoleOperationService(paths, model)
    service.root = _operation_root(paths, create=False)
    path = service._path(operation_id)
    record = _wait_for_parent_publish(path, os.getpid())
    prepared = PreparedOperationV1.from_value(record.get("prepared"))
    if model.runtime_identity().digest != prepared.runtime_identity_digest:
        raise RuntimeError("runtime identity changed before detached execution")
    manifest = AdminManifest.load(manifest_path)
    domain_identity: Mapping[str, Any] = {}
    try:
        if prepared.kind == "MANUAL_EVALUATION_START":
            domain_identity, result = _execute_manual(
                manifest, record["parameters"]
            )
        else:
            from .write_service import execute_domain_operation

            domain_identity, result = execute_domain_operation(
                manifest=manifest,
                kind=prepared.kind,
                operation_id=operation_id,
                parameters=record["parameters"],
            )
        domain_status = str(result.get("status", ""))
        stop_reason = ""
        run_value = result.get("run")
        if isinstance(run_value, Mapping):
            stop_reason = str(run_value.get("stop_reason", ""))
        status = (
            "UNKNOWN_OUTCOME"
            if domain_status in {"UNKNOWN_GPU_OUTCOME", "PAUSED_UNKNOWN_OUTCOME"}
            or (domain_status == "HARD_FAILED" and "unknown" in stop_reason.lower())
            else (
                "FAILED"
                if domain_status in {"FAILED", "HARD_FAILED", "DATA_INTEGRITY"}
                else "SUCCEEDED"
            )
        )
        receipt = _receipt(
            prepared,
            status=status,
            domain_identity=domain_identity,
            result=result,
        )
    except Exception as exc:
        receipt = _receipt(
            prepared,
            status="FAILED",
            domain_identity=domain_identity,
            problem={"code": type(exc).__name__, "detail": str(exc)[:4096]},
        )
    except BaseException as exc:
        receipt = _receipt(
            prepared,
            status="UNKNOWN_OUTCOME",
            domain_identity=domain_identity,
            problem={
                "code": "DETACHED_EXECUTOR_INTERRUPTED",
                "detail": type(exc).__name__,
            },
        )
    with _operation_lock(service.root, operation_id):
        current = _read_record(path)
        if current.get("prepared") != record.get("prepared"):
            raise RuntimeError("operation record identity changed during execution")
        if current.get("receipt") is not None:
            existing = current["receipt"]
            if existing != receipt.to_dict():
                raise RuntimeError("operation already has a conflicting receipt")
            return 0
        current.update({"state": receipt.status, "receipt": receipt.to_dict()})
        _atomic_json(path, current)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return run_worker(args.admin_manifest, args.operation_id)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "run_worker"]
