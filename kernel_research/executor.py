"""Crash-isolated candidate execution.

This module provides fault isolation, not a security boundary. Candidate code is
copied to a temporary working directory and only imported by the child process
for real C500 evaluation. Mock evaluation never imports candidate code.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import tempfile
import time
import traceback
from typing import Any, Mapping

from .constants import (
    CURRENT_C500_EVALUATION_PROTOCOL_ID,
    LEGACY_C500_EVALUATION_PROTOCOL_ID,
    XPUOJ_C500_EVALUATION_PROTOCOL_ID,
)
from .contract import validate_candidate


DEFAULT_MOCK_TIMEOUT_SEC = 5.0
DEFAULT_C500_COMPILE_TIMEOUT_SEC = 180.0
DEFAULT_C500_CASE_TIMEOUT_SEC = 300.0

_SUPPORTED_EVALUATION_PROTOCOL_IDS = frozenset(
    {
        LEGACY_C500_EVALUATION_PROTOCOL_ID,
        CURRENT_C500_EVALUATION_PROTOCOL_ID,
        XPUOJ_C500_EVALUATION_PROTOCOL_ID,
    }
)


def _resolve_evaluation_protocol_id(value: str | None) -> str:
    """Select the current compatibility protocol or fail closed."""

    if value is None:
        return CURRENT_C500_EVALUATION_PROTOCOL_ID
    if (
        not isinstance(value, str)
        or value not in _SUPPORTED_EVALUATION_PROTOCOL_IDS
    ):
        raise ValueError(f"unsupported evaluation protocol: {value!r}")
    return value


def _isolate_process_group() -> int | None:
    """Place the worker and any compiler children in a disposable process group."""

    if hasattr(os, "setsid"):
        try:
            os.setsid()
        except OSError:
            # Fault isolation still works without a dedicated group, but only
            # the direct multiprocessing child can then be terminated.
            pass
    if hasattr(os, "getpgrp") and os.getpgrp() == os.getpid():
        return os.getpid()
    return None


def _publish_process_group(ownership: Any, group_id: int | None) -> None:
    """Synchronously hand verified process-group ownership to the parent."""

    try:
        ownership.send(group_id)
    finally:
        ownership.close()


def _redirect_child_stdout() -> None:
    try:
        os.dup2(2, 1)
    except OSError:
        pass


def _doctor_child_entry(
    output: "mp.Queue[dict[str, Any]]",
    ownership: Any,
    backend_name: str,
) -> None:
    group_id = _isolate_process_group()
    _publish_process_group(ownership, group_id)
    _redirect_child_stdout()
    try:
        from .backends import C500Backend, MockBackend

        result = (
            MockBackend().doctor()
            if backend_name == "mock"
            else C500Backend().doctor(compile_probe=True)
        )
        output.put(result.to_dict())
    except BaseException as exc:
        output.put(
            _result_dict(
                "CRASH",
                error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
            )
        )


def _result_dict(
    status: str,
    *,
    candidate_hash: str = "",
    error: str | None = None,
    eligible_for_promotion: bool = False,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "eligible_for_promotion": eligible_for_promotion,
        "candidate_hash": candidate_hash,
        "cases": [],
        "aggregate_score": None,
        "error": error,
        "environment": dict(environment or {}),
        "benchmark_config": {},
    }


def _child_entry(
    output: "mp.Queue[dict[str, Any]]",
    progress: "mp.Queue[dict[str, Any]]",
    ownership: Any,
    backend_name: str,
    candidate_path: str,
    baseline_path: str | None,
    suite: str,
    evaluation_protocol_id: str,
    test_fault: str | None,
) -> None:
    """Top-level spawn target; all exceptions become serializable results."""

    try:
        group_id = _isolate_process_group()
        _publish_process_group(ownership, group_id)
        # Preserve stdout as a machine-readable channel owned by the parent.
        # Candidate imports, compilers, and vendor runtimes are redirected to
        # stderr at the file-descriptor level, including native-library writes.
        _redirect_child_stdout()
        # The candidate runs from an ephemeral working directory. This reduces
        # accidental writes into the repository, but is not a security sandbox.
        os.chdir(Path(candidate_path).parent)
        if test_fault == "crash":
            os._exit(17)
        if test_fault == "timeout":
            time.sleep(60.0)
        if test_fault == "print":
            print("test fixture: child stdout noise", flush=True)

        from .backends import C500Backend, MockBackend

        backend = MockBackend() if backend_name == "mock" else C500Backend()
        if backend_name == "mock":
            result = backend.evaluate(
                Path(candidate_path),
                suite=suite,
                evaluation_protocol_id=evaluation_protocol_id,
            )
        else:

            def report_progress(
                phase: str, case_id: str | None, stage: str
            ) -> None:
                try:
                    progress.put_nowait(
                        {
                            "phase": phase,
                            "case_id": case_id,
                            "stage": stage,
                        }
                    )
                except (queue.Full, OSError, ValueError):
                    # Watchdog updates must never stall a GPU worker. The queue
                    # is generously sized for the finite number of phase events.
                    pass

            report_progress("compile", None, "backend-initialization")
            result = backend.evaluate(
                Path(candidate_path),
                suite=suite,
                baseline_path=(Path(baseline_path) if baseline_path else None),
                progress_callback=report_progress,
                evaluation_protocol_id=evaluation_protocol_id,
            )
        output.put(result.to_dict())
    except BaseException as exc:  # child must report compiler/runtime failures
        output.put(
            _result_dict(
                "CRASH",
                error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
                environment={"traceback": traceback.format_exc(limit=20)},
            )
        )


def _signal_worker_group(process: mp.Process, signal_number: int) -> bool:
    """Signal a worker-owned POSIX process group when it is safe to do so."""

    if process.pid is None or not hasattr(os, "killpg"):
        return False
    try:
        if os.getpgid(process.pid) != process.pid:
            return False
        os.killpg(process.pid, signal_number)
        return True
    except (OSError, ProcessLookupError):
        return False


def _stop_process(
    process: mp.Process,
    *,
    owned_group_id: int | None = None,
) -> None:
    owned_group = (
        owned_group_id is not None
        and process.pid is not None
        and owned_group_id == process.pid
    )
    group_id = process.pid
    if process.is_alive():
        if owned_group and group_id is not None:
            try:
                os.killpg(group_id, signal.SIGTERM)
            except OSError:
                pass
        else:
            owned_group = _signal_worker_group(process, signal.SIGTERM)
        if not owned_group:
            process.terminate()
        process.join(timeout=2.0)

    # The group leader can exit on SIGTERM while a compiler grandchild ignores
    # it. Once ownership was established, clean the original group even when
    # multiprocessing already considers the direct worker dead.
    if owned_group and group_id is not None and hasattr(os, "killpg"):
        try:
            os.killpg(group_id, signal.SIGKILL)
        except OSError:
            pass
        process.join(timeout=2.0)
    elif process.is_alive() and hasattr(process, "kill"):
        try:
            process.kill()
        except OSError:
            pass
        process.join(timeout=2.0)


def _receive_owned_group(
    ownership: Any,
    process: mp.Process,
    *,
    timeout: float = 0.0,
) -> int | None:
    try:
        if not ownership.poll(timeout):
            return None
        candidate = ownership.recv()
    except (EOFError, OSError):
        return None
    if (
        isinstance(candidate, int)
        and not isinstance(candidate, bool)
        and process.pid is not None
        and candidate == process.pid
    ):
        return candidate
    return None


def _close_connection(value: Any) -> None:
    try:
        value.close()
    except OSError:
        pass


def _drain_progress(
    progress: "mp.Queue[dict[str, Any]]",
    current: dict[str, Any],
) -> dict[str, Any]:
    while True:
        try:
            event = progress.get_nowait()
        except queue.Empty:
            return current
        except (OSError, ValueError):
            return current
        if event.get("phase") in {"compile", "case"}:
            current = {
                "phase": event["phase"],
                "case_id": event.get("case_id"),
                "stage": event.get("stage"),
            }


def _phase_timeout(phase: str) -> float:
    return (
        DEFAULT_C500_COMPILE_TIMEOUT_SEC
        if phase == "compile"
        else DEFAULT_C500_CASE_TIMEOUT_SEC
    )


def _wait_for_evaluation(
    process: mp.Process,
    progress: "mp.Queue[dict[str, Any]]",
    *,
    backend: str,
    explicit_timeout_sec: float | None,
) -> tuple[bool, dict[str, Any]]:
    """Wait with either a global override or real C500 phase deadlines."""

    started = time.monotonic()
    current = {
        "phase": "compile" if backend == "c500" else "mock",
        "case_id": None,
        "stage": "worker-start",
    }
    if explicit_timeout_sec is not None:
        mode = "global-override"
        deadline = started + explicit_timeout_sec
    elif backend == "mock":
        mode = "mock-global"
        deadline = started + DEFAULT_MOCK_TIMEOUT_SEC
    else:
        mode = "c500-phased"
        deadline = started + DEFAULT_C500_COMPILE_TIMEOUT_SEC

    while process.is_alive():
        previous = current
        current = _drain_progress(progress, current)
        if mode == "c500-phased" and current != previous:
            deadline = time.monotonic() + _phase_timeout(str(current["phase"]))

        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            return False, {
                "mode": mode,
                "phase": current["phase"],
                "case_id": current.get("case_id"),
                "stage": current.get("stage"),
                "compile_timeout_sec": DEFAULT_C500_COMPILE_TIMEOUT_SEC,
                "case_timeout_sec": DEFAULT_C500_CASE_TIMEOUT_SEC,
                "explicit_timeout_sec": explicit_timeout_sec,
            }
        process.join(timeout=min(0.1, remaining))

    current = _drain_progress(progress, current)
    return True, {
        "mode": mode,
        "last_phase": current["phase"],
        "last_case_id": current.get("case_id"),
        "last_stage": current.get("stage"),
        "compile_timeout_sec": DEFAULT_C500_COMPILE_TIMEOUT_SEC,
        "case_timeout_sec": DEFAULT_C500_CASE_TIMEOUT_SEC,
        "explicit_timeout_sec": explicit_timeout_sec,
    }


def _close_queue(value: "mp.Queue[Any]") -> None:
    value.close()
    value.join_thread()


def doctor_isolated(
    backend: str,
    *,
    timeout_sec: float | None = None,
) -> dict[str, Any]:
    if backend not in {"mock", "c500"}:
        raise ValueError(f"unsupported backend: {backend}")
    if timeout_sec is None:
        timeout_sec = (
            DEFAULT_MOCK_TIMEOUT_SEC
            if backend == "mock"
            else DEFAULT_C500_COMPILE_TIMEOUT_SEC
        )
    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive")

    context = mp.get_context("spawn")
    output: "mp.Queue[dict[str, Any]]" = context.Queue(maxsize=1)
    progress: "mp.Queue[dict[str, Any]]" = context.Queue(maxsize=1)
    ownership_read, ownership_write = context.Pipe(duplex=False)
    process = context.Process(
        target=_doctor_child_entry,
        args=(output, ownership_write, backend),
        name=f"kernel-research-doctor-{backend}",
    )
    owned_group_id: int | None = None
    started = False
    try:
        process.start()
        started = True
        _close_connection(ownership_write)
        completed, watchdog = _wait_for_evaluation(
            process,
            progress,
            backend=backend,
            explicit_timeout_sec=timeout_sec,
        )
        owned_group_id = _receive_owned_group(ownership_read, process)
        if not completed:
            return _result_dict(
                "TIMEOUT",
                error=f"backend doctor exceeded {timeout_sec:.3f}s",
                environment={
                    "timeout_phase": "compile-probe",
                    "compile_timeout_sec": timeout_sec,
                },
            )
        try:
            result = output.get(timeout=1.0)
        except queue.Empty:
            result = _result_dict(
                "CRASH",
                error=(
                    f"doctor worker exited with code {process.exitcode} "
                    "without a result"
                ),
            )
        environment = dict(result.get("environment") or {})
        environment["watchdog"] = watchdog
        result["environment"] = environment
        return result
    finally:
        if started or process.pid is not None:
            owned_group_id = owned_group_id or _receive_owned_group(
                ownership_read, process
            )
            _stop_process(process, owned_group_id=owned_group_id)
        _close_connection(ownership_write)
        _close_connection(ownership_read)
        _close_queue(output)
        _close_queue(progress)


def evaluate_isolated(
    candidate_path: str | Path,
    *,
    backend: str,
    suite: str = "smoke",
    timeout_sec: float | None = None,
    candidate_source: str | None = None,
    baseline_path: str | Path | None = None,
    baseline_source: str | None = None,
    evaluation_protocol_id: str | None = None,
    test_fault: str | None = None,
) -> dict[str, Any]:
    """Evaluate a candidate in a spawned process and return a result dictionary.

    ``test_fault`` is deliberately internal test injection. It is not exposed by
    the CLI and must never influence normal candidate scoring.
    """

    if backend not in {"mock", "c500"}:
        raise ValueError(f"unsupported backend: {backend}")
    if suite not in {"smoke", "quick", "full"}:
        raise ValueError(f"unsupported suite: {suite}")
    resolved_evaluation_protocol_id = _resolve_evaluation_protocol_id(
        evaluation_protocol_id
    )
    if test_fault not in {None, "crash", "timeout", "print"}:
        raise ValueError(f"unsupported test fault: {test_fault}")

    candidate = Path(candidate_path).resolve()
    validation = (
        validate_candidate(source=candidate_source)
        if candidate_source is not None
        else validate_candidate(candidate)
    )
    if not validation.is_valid:
        result = _result_dict(
            "CONTRACT_ERROR",
            candidate_hash=validation.sha256,
            error="; ".join(
                f"{candidate_error.code}: {candidate_error.message}"
                for candidate_error in validation.errors
            ),
        )
        result["evaluation_protocol_id"] = resolved_evaluation_protocol_id
        return result

    baseline_validation = None
    if baseline_path is not None or baseline_source is not None:
        if backend != "c500":
            raise ValueError("a baseline is only valid for the c500 backend")
        baseline_validation = (
            validate_candidate(source=baseline_source)
            if baseline_source is not None
            else validate_candidate(Path(baseline_path).resolve())
        )
        if not baseline_validation.is_valid:
            result = _result_dict(
                "CRASH",
                candidate_hash=validation.sha256,
                error="accepted baseline artifact failed static validation",
                environment={
                    "baseline_contract_errors": [
                        error.to_dict() for error in baseline_validation.errors
                    ]
                },
            )
            result["evaluation_protocol_id"] = resolved_evaluation_protocol_id
            return result

    if timeout_sec is not None and timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive")

    context = mp.get_context("spawn")
    output: "mp.Queue[dict[str, Any]]" = context.Queue(maxsize=1)
    progress: "mp.Queue[dict[str, Any]]" = context.Queue(maxsize=64)
    ownership_read, ownership_write = context.Pipe(duplex=False)
    owned_group_id: int | None = None
    watchdog: dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix="kernel-research-") as temporary_dir:
        isolated_candidate = Path(temporary_dir) / "kernel.py"
        isolated_candidate.write_text(validation.source, encoding="utf-8")
        isolated_baseline: Path | None = None
        if baseline_validation is not None:
            isolated_baseline = Path(temporary_dir) / "baseline.py"
            isolated_baseline.write_text(
                baseline_validation.source, encoding="utf-8"
            )
        process = context.Process(
            target=_child_entry,
            args=(
                output,
                progress,
                ownership_write,
                backend,
                str(isolated_candidate),
                str(isolated_baseline) if isolated_baseline is not None else None,
                suite,
                resolved_evaluation_protocol_id,
                test_fault,
            ),
            name=f"kernel-research-{backend}",
        )
        started = False
        try:
            process.start()
            started = True
            _close_connection(ownership_write)
            completed, watchdog = _wait_for_evaluation(
                process,
                progress,
                backend=backend,
                explicit_timeout_sec=timeout_sec,
            )
            owned_group_id = _receive_owned_group(ownership_read, process)
            if not completed:
                case_text = (
                    f" for case {watchdog['case_id']}"
                    if watchdog.get("case_id")
                    else ""
                )
                result = _result_dict(
                    "TIMEOUT",
                    candidate_hash=validation.sha256,
                    error=(
                        f"evaluation exceeded the {watchdog['phase']} deadline"
                        f"{case_text}"
                    ),
                    environment={
                        "timeout_mode": watchdog["mode"],
                        "timeout_phase": watchdog["phase"],
                        "timeout_case_id": watchdog.get("case_id"),
                        "timeout_stage": watchdog.get("stage"),
                        "compile_timeout_sec": watchdog["compile_timeout_sec"],
                        "case_timeout_sec": watchdog["case_timeout_sec"],
                        "explicit_timeout_sec": watchdog.get(
                            "explicit_timeout_sec"
                        ),
                    },
                )
            else:
                try:
                    # A multiprocessing queue may still be flushing its feeder
                    # thread briefly after the process has exited.
                    result = output.get(timeout=1.0)
                except queue.Empty:
                    result = _result_dict(
                        "CRASH",
                        candidate_hash=validation.sha256,
                        error=(
                            f"worker exited with code {process.exitcode} "
                            "without a result"
                        ),
                    )
        finally:
            if started or process.pid is not None:
                owned_group_id = owned_group_id or _receive_owned_group(
                    ownership_read, process
                )
                _stop_process(process, owned_group_id=owned_group_id)
            _close_connection(ownership_write)
            _close_connection(ownership_read)
            _close_queue(output)
            _close_queue(progress)

    result["candidate_hash"] = validation.sha256
    result["evaluation_protocol_id"] = resolved_evaluation_protocol_id
    environment = dict(result.get("environment") or {})
    environment["watchdog"] = watchdog
    result["environment"] = environment
    return result
