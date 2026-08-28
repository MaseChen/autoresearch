"""Detached remote executor for one already-prepared Console operation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import Any, Iterator, Mapping

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


MAX_WORKER_OUTPUT_BYTES = 64 * 1024
_TRUNCATION_MARKER = b"\n[console worker output truncated]\n"


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("could not write bounded Console worker output")
        view = view[written:]


def _drain_output(read_descriptor: int, output_descriptor: int, maximum: int) -> None:
    content_limit = maximum - len(_TRUNCATION_MARKER)
    written = 0
    truncated = False
    try:
        while True:
            chunk = os.read(read_descriptor, 16 * 1024)
            if not chunk:
                break
            remaining = max(0, content_limit - written)
            if remaining:
                selected = chunk[:remaining]
                _write_all(output_descriptor, selected)
                written += len(selected)
            if len(chunk) > remaining and not truncated:
                _write_all(output_descriptor, _TRUNCATION_MARKER)
                truncated = True
    except OSError:
        # The audit stream is advisory. Never recurse through threading's
        # exception hook into the stderr pipe that this thread may be draining.
        pass
    finally:
        os.close(read_descriptor)
        os.close(output_descriptor)


def _install_output_drain(target: int, maximum: int) -> threading.Thread:
    destination = os.dup(target)
    read_descriptor = -1
    write_descriptor = -1
    installed = False
    try:
        read_descriptor, write_descriptor = os.pipe()
        os.dup2(write_descriptor, target, inheritable=True)
        installed = True
        os.close(write_descriptor)
        write_descriptor = -1
        thread = threading.Thread(
            target=_drain_output,
            args=(read_descriptor, destination, maximum),
            name=f"console-output-drain-{target}",
            daemon=True,
        )
        thread.start()
        return thread
    except BaseException:
        if installed:
            os.dup2(destination, target, inheritable=True)
        for descriptor in (read_descriptor, write_descriptor, destination):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        raise


@contextmanager
def _bounded_output_descriptors(
    stdout_descriptor: int,
    stderr_descriptor: int,
    *,
    maximum: int = MAX_WORKER_OUTPUT_BYTES,
) -> Iterator[None]:
    """Bound inherited worker output without constraining scientific files."""

    if type(maximum) is not int or maximum <= len(_TRUNCATION_MARKER):
        raise ValueError("Console worker output bound is invalid")
    targets = (stdout_descriptor, stderr_descriptor)
    threads: list[threading.Thread] = []
    installed: list[int] = []
    try:
        for target in targets:
            thread = _install_output_drain(target, maximum)
            threads.append(thread)
            installed.append(target)
        yield
    finally:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (OSError, ValueError):
                pass
        null_descriptor = os.open(os.devnull, os.O_WRONLY)
        try:
            for target in installed:
                os.dup2(null_descriptor, target, inheritable=True)
        finally:
            os.close(null_descriptor)
        for thread in threads:
            thread.join(timeout=5.0)


@contextmanager
def _bounded_process_output() -> Iterator[None]:
    with _bounded_output_descriptors(sys.stdout.fileno(), sys.stderr.fileno()):
        yield


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
        # This stream is a private, mode-0600 Controller diagnostic bounded by
        # the worker output drainer. Do not include locals or expose it through HTTP.
        traceback.print_exc()
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
    with _bounded_process_output():
        return run_worker(args.admin_manifest, args.operation_id)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["MAX_WORKER_OUTPUT_BYTES", "main", "run_worker"]
