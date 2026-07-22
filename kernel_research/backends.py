"""Evaluation backends for the fixed kernel-research control plane.

The mock backend validates source and the framework's NumPy reference only. It
never imports candidate code and never fabricates timing data. The C500 backend
does import and execute a candidate, and therefore must be called by the
process-isolation layer in :mod:`kernel_research.executor` rather than directly
from a long-lived orchestrator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
import importlib.util
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

from .contract import validate_candidate


STATUS_MOCK_VALIDATED = "MOCK_VALIDATED"
STATUS_UNSUPPORTED_ENV = "UNSUPPORTED_ENV"
STATUS_CONTRACT_ERROR = "CONTRACT_ERROR"
STATUS_COMPILE_ERROR = "COMPILE_ERROR"
STATUS_PRECISION_FAILED = "PRECISION_FAILED"
STATUS_CRASH = "CRASH"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_SUCCESS = "SUCCESS"

SCHEMA_VERSION = 1
MATCHED_RATIO_THRESHOLD = 0.99
WARMUP_ITERATIONS = 10
MEASUREMENT_ROUNDS = 3
SAMPLES_PER_ROUND = 10

ProgressCallback = Callable[[str, str | None, str], None]


@dataclass(frozen=True)
class CaseResult:
    """Serializable outcome for one deterministic test case."""

    case_id: str
    status: str
    matched_ratio: float | None = None
    latency_samples_us: tuple[float, ...] = ()
    baseline_latency_samples_us: tuple[float, ...] = ()
    p20_us: float | None = None
    p50_us: float | None = None
    p80_us: float | None = None
    baseline_p20_us: float | None = None
    baseline_p50_us: float | None = None
    baseline_p80_us: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendResult:
    """JSON-ready result returned by both evaluation backends."""

    status: str
    eligible_for_promotion: bool
    candidate_hash: str
    cases: tuple[CaseResult, ...] = ()
    aggregate_score: float | None = None
    error: str | None = None
    environment: Mapping[str, Any] = field(default_factory=dict)
    benchmark_config: Mapping[str, Any] = field(default_factory=dict)
    backend: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "status": self.status,
            "eligible_for_promotion": self.eligible_for_promotion,
            "candidate_hash": self.candidate_hash,
            "cases": [case.to_dict() for case in self.cases],
            "aggregate_score": self.aggregate_score,
            "error": self.error,
            "environment": dict(self.environment),
            "benchmark_config": dict(self.benchmark_config),
        }


def _base_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "implementation": platform.python_implementation(),
    }


def _validation_error(validation: Any, *, backend: str) -> BackendResult:
    messages = [error.message for error in validation.errors]
    return BackendResult(
        status=STATUS_CONTRACT_ERROR,
        eligible_for_promotion=False,
        candidate_hash=validation.sha256,
        error="; ".join(messages),
        environment={
            **_base_environment(),
            "contract_errors": [error.to_dict() for error in validation.errors],
        },
        backend=backend,
    )


class MockBackend:
    """Deterministic control-plane validation that performs no candidate import."""

    name = "mock"

    def doctor(self) -> BackendResult:
        return BackendResult(
            status=STATUS_MOCK_VALIDATED,
            eligible_for_promotion=False,
            candidate_hash="",
            environment={
                **_base_environment(),
                "candidate_execution": False,
                "performance_measurement": False,
            },
            benchmark_config={
                "mode": "static-contract-and-reference-self-check",
                "latency_samples": 0,
            },
            backend=self.name,
        )

    def evaluate(
        self,
        candidate_path: str | Path,
        *,
        suite: str = "smoke",
    ) -> BackendResult:
        # validate_candidate reads and parses source. It intentionally never
        # imports it, even when the source contains valid top-level side effects.
        validation = validate_candidate(Path(candidate_path))
        if not validation.valid:
            return _validation_error(validation, backend=self.name)

        if suite not in {"smoke", "quick", "full"}:
            raise ValueError(f"unknown suite: {suite}")

        try:
            from .reference import reference_self_check

            self_check = reference_self_check()
        except Exception as exc:
            return BackendResult(
                status=STATUS_CRASH,
                eligible_for_promotion=False,
                candidate_hash=validation.sha256,
                error=f"reference self-check raised {type(exc).__name__}: {exc}",
                environment={**_base_environment(), "candidate_imported": False},
                backend=self.name,
            )

        check_ok = bool(self_check.get("ok", False))
        if not check_ok:
            return BackendResult(
                status=STATUS_PRECISION_FAILED,
                eligible_for_promotion=False,
                candidate_hash=validation.sha256,
                error="framework reference self-check failed",
                environment={
                    **_base_environment(),
                    "candidate_imported": False,
                    "reference_self_check": _json_safe(self_check),
                },
                backend=self.name,
            )

        # No candidate case is reported because the candidate was not executed.
        # The framework-only check remains explicit in environment metadata.
        return BackendResult(
            status=STATUS_MOCK_VALIDATED,
            eligible_for_promotion=False,
            candidate_hash=validation.sha256,
            cases=(),
            aggregate_score=None,
            environment={
                **_base_environment(),
                "candidate_imported": False,
                "candidate_executed": False,
                "performance_measurement": False,
                "reference_self_check": _json_safe(self_check),
                "requested_suite": suite,
            },
            benchmark_config={
                "mode": "static-contract-and-reference-self-check",
                "warmup_iterations": 0,
                "measurement_rounds": 0,
                "samples_per_round": 0,
                "l2_flush": False,
            },
            backend=self.name,
        )


class C500Backend:
    """Real single-candidate C500 correctness and warm-cache timing backend.

    This backend returns real raw samples but does not compare them with a
    current-best candidate or perform the required confirmation run. Therefore
    even ``SUCCESS`` is not, by itself, eligible for promotion. The scoring
    layer owns baseline interleaving, the 1% threshold, per-case regressions,
    and confirmation.
    """

    name = "c500"

    def doctor(self, *, compile_probe: bool = False) -> BackendResult:
        environment = {
            **_base_environment(),
            **_vendor_version_defaults(),
            "maca_path": os.environ.get("MACA_PATH", "/opt/maca"),
        }
        runtime, error = _load_c500_runtime()
        if error is not None:
            environment.update(error)
            return BackendResult(
                status=STATUS_UNSUPPORTED_ENV,
                eligible_for_promotion=False,
                candidate_hash="",
                error=str(error["reason"]),
                environment=environment,
                benchmark_config=_c500_benchmark_config(),
                backend=self.name,
            )

        assert runtime is not None
        torch, triton = runtime
        triton_version = str(getattr(triton, "__version__", "unknown"))
        triton_path = str(getattr(triton, "__file__", "unknown"))
        environment.update(
            {
                "torch_version": str(getattr(torch, "__version__", "unknown")),
                "triton_version": triton_version,
                "mctriton_version": triton_version,
                "torch_path": str(getattr(torch, "__file__", "unknown")),
                "triton_path": triton_path,
                "mctriton_path": triton_path,
            }
        )
        accelerator, accelerator_error = _find_accelerator(torch)
        if accelerator_error is not None:
            environment.update(accelerator_error)
            return BackendResult(
                status=STATUS_UNSUPPORTED_ENV,
                eligible_for_promotion=False,
                candidate_hash="",
                error=str(accelerator_error["reason"]),
                environment=environment,
                benchmark_config=_c500_benchmark_config(),
                backend=self.name,
            )

        environment.update(accelerator)
        environment.update(_vendor_version_fingerprint(torch))
        environment["compile_probe_requested"] = bool(compile_probe)
        environment["compile_probe_passed"] = False
        if compile_probe:
            try:
                from .c500_probe import run_probe

                run_probe(str(environment["device"]))
            except Exception as exc:
                return BackendResult(
                    status=STATUS_UNSUPPORTED_ENV,
                    eligible_for_promotion=False,
                    candidate_hash="",
                    error=f"mcTriton compile probe failed: {type(exc).__name__}: {exc}",
                    environment=environment,
                    benchmark_config=_c500_benchmark_config(),
                    backend=self.name,
                )
            environment["compile_probe_passed"] = True
        return BackendResult(
            status=STATUS_SUCCESS,
            eligible_for_promotion=False,
            candidate_hash="",
            environment=environment,
            benchmark_config=_c500_benchmark_config(),
            backend=self.name,
        )

    def evaluate(
        self,
        candidate_path: str | Path,
        *,
        suite: str = "smoke",
        baseline_path: str | Path | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> BackendResult:
        candidate_path = Path(candidate_path)
        validation = validate_candidate(candidate_path)
        if not validation.valid:
            return _validation_error(validation, backend=self.name)

        health = self.doctor()
        if health.status != STATUS_SUCCESS:
            return BackendResult(
                status=health.status,
                eligible_for_promotion=False,
                candidate_hash=validation.sha256,
                error=health.error,
                environment=health.environment,
                benchmark_config=health.benchmark_config,
                backend=self.name,
            )

        runtime, runtime_error = _load_c500_runtime()
        if runtime_error is not None or runtime is None:
            # A disappearing runtime is unusual but remains a closed, explicit
            # result instead of falling through to a candidate import.
            return BackendResult(
                status=STATUS_UNSUPPORTED_ENV,
                eligible_for_promotion=False,
                candidate_hash=validation.sha256,
                error="C500 runtime became unavailable after doctor",
                environment=health.environment,
                benchmark_config=health.benchmark_config,
                backend=self.name,
            )
        torch, _triton = runtime

        try:
            from .cases import generate_case, get_suite

            case_specs = get_suite(suite)
        except Exception as exc:
            return BackendResult(
                status=STATUS_CRASH,
                eligible_for_promotion=False,
                candidate_hash=validation.sha256,
                error=f"could not load suite {suite!r}: {type(exc).__name__}: {exc}",
                environment=health.environment,
                benchmark_config=health.benchmark_config,
                backend=self.name,
            )

        try:
            candidate_module = _import_candidate(candidate_path, validation.sha256)
            run_kernel = getattr(candidate_module, "run_kernel")
        except Exception as exc:
            return BackendResult(
                status=STATUS_COMPILE_ERROR,
                eligible_for_promotion=False,
                candidate_hash=validation.sha256,
                error=f"candidate import failed: {type(exc).__name__}: {exc}",
                environment=health.environment,
                benchmark_config=health.benchmark_config,
                backend=self.name,
            )

        baseline_validation = None
        baseline_run_kernel = None
        if baseline_path is not None:
            baseline_path = Path(baseline_path)
            baseline_validation = validate_candidate(baseline_path)
            if not baseline_validation.valid:
                return BackendResult(
                    status=STATUS_CRASH,
                    eligible_for_promotion=False,
                    candidate_hash=validation.sha256,
                    error="accepted baseline artifact failed static validation",
                    environment={
                        **dict(health.environment),
                        "baseline_contract_errors": [
                            error.to_dict() for error in baseline_validation.errors
                        ],
                    },
                    benchmark_config=health.benchmark_config,
                    backend=self.name,
                )
            try:
                baseline_module = _import_candidate(
                    baseline_path, baseline_validation.sha256
                )
                baseline_run_kernel = getattr(baseline_module, "run_kernel")
            except Exception as exc:
                return BackendResult(
                    status=STATUS_COMPILE_ERROR,
                    eligible_for_promotion=False,
                    candidate_hash=validation.sha256,
                    error=(
                        "accepted baseline import failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    environment=health.environment,
                    benchmark_config=health.benchmark_config,
                    backend=self.name,
                )

        device = str(health.environment.get("device", "cuda:0"))
        case_results: list[CaseResult] = []
        for index, case_spec in enumerate(case_specs):
            case_id = str(getattr(case_spec, "name", f"case-{index}"))
            _report_progress(progress_callback, "case", case_id, "prepare")
            try:
                numpy_dataset = generate_case(case_spec)
                tensors = _copy_dataset_to_device(torch, numpy_dataset, device)
                del numpy_dataset
                input_snapshots = _clone_readonly_inputs(tensors)
                expected = _torch_reference(torch, tensors)
            except MemoryError as exc:
                case_results.append(
                    CaseResult(
                        case_id=case_id,
                        status=STATUS_CRASH,
                        error=f"case allocation failed without downscaling: {exc}",
                    )
                )
                return _failed_c500_result(
                    validation.sha256, case_results, health, STATUS_CRASH
                )
            except Exception as exc:
                case_results.append(
                    CaseResult(
                        case_id=case_id,
                        status=STATUS_CRASH,
                        error=f"case preparation failed: {type(exc).__name__}: {exc}",
                    )
                )
                return _failed_c500_result(
                    validation.sha256, case_results, health, STATUS_CRASH
                )

            arguments = tensors["arguments"]
            baseline_arguments = None
            if baseline_run_kernel is not None:
                baseline_out = torch.empty_like(tensors["out"])
                baseline_arguments = (*arguments[:-1], baseline_out)
            try:
                # The first call includes JIT compilation and is intentionally
                # excluded from timing. The outer executor applies a fresh
                # compile deadline for every shape.
                _report_progress(
                    progress_callback, "compile", case_id, "first-invocation"
                )
                run_kernel(*arguments)
                if baseline_run_kernel is not None and baseline_arguments is not None:
                    baseline_run_kernel(*baseline_arguments)
                _synchronize(torch)
                _report_progress(
                    progress_callback, "case", case_id, "validate-and-benchmark"
                )
            except Exception as exc:
                case_results.append(
                    CaseResult(
                        case_id=case_id,
                        status=STATUS_COMPILE_ERROR,
                        error=f"first kernel invocation failed: {type(exc).__name__}: {exc}",
                    )
                )
                return _failed_c500_result(
                    validation.sha256, case_results, health, STATUS_COMPILE_ERROR
                )

            matched_ratio = _matched_ratio(torch, tensors["out"], expected)
            if matched_ratio < MATCHED_RATIO_THRESHOLD:
                case_results.append(
                    CaseResult(
                        case_id=case_id,
                        status=STATUS_PRECISION_FAILED,
                        matched_ratio=matched_ratio,
                        error=(
                            f"matched ratio {matched_ratio:.6f} is below "
                            f"{MATCHED_RATIO_THRESHOLD:.2f}"
                        ),
                    )
                )
                return _failed_c500_result(
                    validation.sha256,
                    case_results,
                    health,
                    STATUS_PRECISION_FAILED,
                )

            if baseline_arguments is not None:
                baseline_matched_ratio = _matched_ratio(
                    torch, baseline_arguments[-1], expected
                )
                if baseline_matched_ratio < MATCHED_RATIO_THRESHOLD:
                    case_results.append(
                        CaseResult(
                            case_id=case_id,
                            status=STATUS_CRASH,
                            matched_ratio=matched_ratio,
                            error=(
                                "accepted baseline no longer passes correctness: "
                                f"{baseline_matched_ratio:.6f}"
                            ),
                        )
                    )
                    return _failed_c500_result(
                        validation.sha256,
                        case_results,
                        health,
                        STATUS_CRASH,
                    )

            try:
                if baseline_run_kernel is None or baseline_arguments is None:
                    samples = _benchmark_candidate(torch, run_kernel, arguments)
                    baseline_samples: list[float] = []
                else:
                    samples, baseline_samples = _benchmark_interleaved(
                        torch,
                        run_kernel,
                        arguments,
                        baseline_run_kernel,
                        baseline_arguments,
                    )
            except Exception as exc:
                case_results.append(
                    CaseResult(
                        case_id=case_id,
                        status=STATUS_CRASH,
                        matched_ratio=matched_ratio,
                        error=f"timing failed: {type(exc).__name__}: {exc}",
                    )
                )
                return _failed_c500_result(
                    validation.sha256, case_results, health, STATUS_CRASH
                )

            fresh_matched_ratio = _fresh_output_check(
                torch, run_kernel, arguments, expected
            )
            if fresh_matched_ratio < MATCHED_RATIO_THRESHOLD:
                case_results.append(
                    CaseResult(
                        case_id=case_id,
                        status=STATUS_PRECISION_FAILED,
                        matched_ratio=fresh_matched_ratio,
                        error=(
                            "post-benchmark fresh-output check failed: "
                            f"{fresh_matched_ratio:.6f}"
                        ),
                    )
                )
                return _failed_c500_result(
                    validation.sha256,
                    case_results,
                    health,
                    STATUS_PRECISION_FAILED,
                )

            if baseline_run_kernel is not None and baseline_arguments is not None:
                baseline_fresh_ratio = _fresh_output_check(
                    torch, baseline_run_kernel, baseline_arguments, expected
                )
                if baseline_fresh_ratio < MATCHED_RATIO_THRESHOLD:
                    case_results.append(
                        CaseResult(
                            case_id=case_id,
                            status=STATUS_CRASH,
                            matched_ratio=fresh_matched_ratio,
                            error=(
                                "accepted baseline failed post-benchmark check: "
                                f"{baseline_fresh_ratio:.6f}"
                            ),
                        )
                    )
                    return _failed_c500_result(
                        validation.sha256,
                        case_results,
                        health,
                        STATUS_CRASH,
                    )

            readonly_ok, modified_input = _readonly_inputs_unchanged(
                torch, tensors, input_snapshots
            )
            if not readonly_ok:
                case_results.append(
                    CaseResult(
                        case_id=case_id,
                        status=STATUS_CRASH,
                        matched_ratio=fresh_matched_ratio,
                        error=f"read-only input was modified: {modified_input}",
                    )
                )
                return _failed_c500_result(
                    validation.sha256,
                    case_results,
                    health,
                    STATUS_CRASH,
                )

            case_results.append(
                CaseResult(
                    case_id=case_id,
                    status=STATUS_SUCCESS,
                    matched_ratio=fresh_matched_ratio,
                    latency_samples_us=tuple(samples),
                    baseline_latency_samples_us=tuple(baseline_samples),
                    p20_us=_percentile(samples, 20.0),
                    p50_us=_percentile(samples, 50.0),
                    p80_us=_percentile(samples, 80.0),
                    baseline_p20_us=(
                        _percentile(baseline_samples, 20.0)
                        if baseline_samples
                        else None
                    ),
                    baseline_p50_us=(
                        _percentile(baseline_samples, 50.0)
                        if baseline_samples
                        else None
                    ),
                    baseline_p80_us=(
                        _percentile(baseline_samples, 80.0)
                        if baseline_samples
                        else None
                    ),
                )
            )
            # Release every tensor from this case before constructing the next
            # full-size case. The allocator may retain reusable blocks, but no
            # live tensor from two cases is allowed to overlap.
            del input_snapshots, expected, arguments, baseline_arguments, tensors
            if baseline_run_kernel is not None:
                del baseline_out

        return BackendResult(
            status=STATUS_SUCCESS,
            # Promotion additionally requires interleaved baseline comparison
            # and a second confirmation evaluation, neither done here.
            eligible_for_promotion=False,
            candidate_hash=validation.sha256,
            cases=tuple(case_results),
            aggregate_score=None,
            environment={
                **dict(health.environment),
                "measurement_scope": (
                    "interleaved-current-best"
                    if baseline_validation is not None
                    else "initial-baseline"
                ),
                "baseline_candidate_hash": (
                    baseline_validation.sha256
                    if baseline_validation is not None
                    else None
                ),
                "hardware_validation": "executed",
            },
            benchmark_config={
                **dict(health.benchmark_config),
                "comparison": (
                    "alternating current-best and candidate"
                    if baseline_validation is not None
                    else "initial baseline; no comparison candidate"
                ),
            },
            backend=self.name,
        )


def _c500_benchmark_config() -> dict[str, Any]:
    return {
        "cache_policy": "warm-cache-steady-state",
        "l2_flush": False,
        "warmup_iterations": WARMUP_ITERATIONS,
        "measurement_rounds": MEASUREMENT_ROUNDS,
        "samples_per_round": SAMPLES_PER_ROUND,
        "total_samples_per_case": MEASUREMENT_ROUNDS * SAMPLES_PER_ROUND,
        "comparison": "current-best/candidate interleaving when a baseline is supplied",
    }


def _report_progress(
    callback: ProgressCallback | None,
    phase: str,
    case_id: str | None,
    stage: str,
) -> None:
    """Report watchdog phase changes without coupling the backend to IPC."""

    if callback is not None:
        callback(phase, case_id, stage)


def _load_c500_runtime() -> tuple[tuple[Any, Any] | None, dict[str, Any] | None]:
    loaded: list[str] = []
    try:
        torch = importlib.import_module("torch")
        loaded.append("torch")
        triton = importlib.import_module("triton")
        loaded.append("triton")
    except Exception as exc:
        missing = "triton" if loaded == ["torch"] else "torch"
        return None, {
            "reason": f"C500 runtime dependency unavailable: {missing}: {exc}",
            "missing_dependency": missing,
            "loaded_dependencies": loaded,
        }
    return (torch, triton), None


def _vendor_version_defaults() -> dict[str, Any]:
    return {
        "sdk_version": None,
        "sdk_version_source": None,
        "sdk_probe_error": "vendor runtime was not available for SDK probing",
        "driver_version": None,
        "driver_version_source": None,
        "driver_probe_error": "vendor runtime was not available for driver probing",
    }


def _sdk_version_fingerprint(torch: Any) -> dict[str, Any]:
    errors: list[str] = []
    for variable in ("MACA_SDK_VERSION", "MXMACA_VERSION", "MACA_VERSION"):
        value = os.environ.get(variable)
        if value:
            return {
                "sdk_version": value,
                "sdk_version_source": f"environment:{variable}",
                "sdk_probe_error": None,
            }

    torch_version = getattr(torch, "version", None)
    for attribute in ("maca", "mxmaca"):
        value = getattr(torch_version, attribute, None)
        if value:
            return {
                "sdk_version": str(value),
                "sdk_version_source": f"torch.version.{attribute}",
                "sdk_probe_error": None,
            }

    maca_path = Path(os.environ.get("MACA_PATH", "/opt/maca"))
    for filename in ("VERSION", "version.txt", "version"):
        candidate = maca_path / filename
        if not candidate.is_file():
            continue
        try:
            value = candidate.read_text(encoding="utf-8").strip()[:4096]
        except (OSError, UnicodeError) as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
            continue
        if value:
            return {
                "sdk_version": value,
                "sdk_version_source": str(candidate),
                "sdk_probe_error": None,
            }

    if not errors:
        errors.append(
            "no vendor SDK version attribute, environment value, or version file"
        )
    return {
        "sdk_version": None,
        "sdk_version_source": None,
        "sdk_probe_error": "; ".join(errors),
    }


def _runtime_driver_fingerprint(torch: Any) -> dict[str, Any]:
    errors: list[str] = []
    for api_name in ("cuda", "maca"):
        api = getattr(torch, api_name, None)
        if api is None:
            continue
        for attribute in ("driver_version", "get_driver_version"):
            probe = getattr(api, attribute, None)
            if not callable(probe):
                continue
            try:
                value = probe()
            except Exception as exc:
                errors.append(
                    f"torch.{api_name}.{attribute}: {type(exc).__name__}: {exc}"
                )
                continue
            if value is not None and str(value):
                return {
                    "driver_version": str(value),
                    "driver_version_source": f"torch.{api_name}.{attribute}",
                    "driver_probe_error": None,
                }
    return {
        "driver_version": None,
        "driver_version_source": None,
        "driver_probe_error": (
            "; ".join(errors)
            if errors
            else "vendor Torch runtime exposes no driver-version API"
        ),
    }


def _parse_driver_version(text: str) -> str | None:
    match = re.search(
        r"(?im)^\s*(?:kernel\s+)?driver(?:\s+version)?\s*[:=]\s*([^\s,]+)",
        text,
    )
    return None if match is None else match.group(1)


def _mx_smi_fingerprint() -> dict[str, Any]:
    executable = shutil.which("mx-smi")
    if executable is None:
        return {
            "mx_smi_path": None,
            "mx_smi_version": None,
            "mx_smi_driver_version": None,
            "mx_smi_probe_error": "mx-smi was not found on PATH",
        }

    results: dict[str, subprocess.CompletedProcess[str]] = {}
    errors: list[str] = []
    for label, arguments in (("version", ["--version"]), ("query", ["-q"])):
        try:
            completed = subprocess.run(
                [executable, *arguments],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"mx-smi {label}: {type(exc).__name__}: {exc}")
            continue
        results[label] = completed
        if completed.returncode != 0:
            errors.append(f"mx-smi {label} exited with {completed.returncode}")

    version_process = results.get("version")
    version_text = (
        ""
        if version_process is None
        else (version_process.stdout or version_process.stderr).strip()[:4096]
    )
    query_process = results.get("query")
    query_text = (
        ""
        if query_process is None
        else (query_process.stdout or query_process.stderr).strip()[:16384]
    )
    driver_version = _parse_driver_version("\n".join((version_text, query_text)))
    if driver_version is None:
        errors.append("driver version was not present in mx-smi output")
    return {
        "mx_smi_path": executable,
        "mx_smi_version": version_text or None,
        "mx_smi_driver_version": driver_version,
        "mx_smi_version_exit_code": (
            None if version_process is None else version_process.returncode
        ),
        "mx_smi_query_exit_code": (
            None if query_process is None else query_process.returncode
        ),
        "mx_smi_probe_error": "; ".join(errors) if errors else None,
    }


def _vendor_version_fingerprint(torch: Any) -> dict[str, Any]:
    sdk = _sdk_version_fingerprint(torch)
    runtime_driver = _runtime_driver_fingerprint(torch)
    mx_smi = _mx_smi_fingerprint()
    smi_driver = mx_smi.get("mx_smi_driver_version")
    driver = (
        {
            "driver_version": str(smi_driver),
            "driver_version_source": "mx-smi",
            "driver_probe_error": None,
        }
        if smi_driver
        else runtime_driver
    )
    if driver["driver_version"] is None and mx_smi.get("mx_smi_probe_error"):
        existing = driver.get("driver_probe_error")
        driver["driver_probe_error"] = "; ".join(
            value
            for value in (existing, str(mx_smi["mx_smi_probe_error"]))
            if value
        )
    return {**sdk, **driver, **mx_smi}


def _find_accelerator(torch: Any) -> tuple[dict[str, Any], dict[str, Any] | None]:
    for api_name in ("cuda", "maca"):
        api = getattr(torch, api_name, None)
        if api is None:
            continue
        try:
            available = bool(api.is_available())
        except Exception as exc:
            return {}, {"reason": f"torch.{api_name}.is_available failed: {exc}"}
        if not available:
            continue
        try:
            device_count = int(api.device_count())
            device_name = str(api.get_device_name(0))
        except Exception as exc:
            return {}, {"reason": f"could not inspect {api_name} device 0: {exc}"}
        if "c500" not in device_name.casefold():
            return {}, {
                "reason": (
                    "accelerator is available but is not a MetaX C500 target: "
                    f"{device_name}"
                ),
                "device_name": device_name,
                "accelerator_api": api_name,
            }
        return (
            {
                "accelerator_api": api_name,
                "device": f"{api_name}:0",
                "device_count": device_count,
                "device_name": device_name,
                "target_verified": True,
            },
            None,
        )
    return {}, {"reason": "no CUDA/MACA-compatible accelerator is available"}


def _import_candidate(candidate_path: Path, candidate_hash: str) -> ModuleType:
    module_name = f"kernel_candidate_{candidate_hash[:16]}"
    specification = importlib.util.spec_from_file_location(module_name, candidate_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"could not create import specification for {candidate_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _copy_dataset_to_device(
    torch: Any,
    dataset: Any,
    device: str,
) -> dict[str, Any]:
    def copy(name: str) -> Any:
        array = getattr(dataset, name)
        return torch.as_tensor(array).to(device=device)

    a = copy("a")
    b_col_major = copy("b_col_major")
    scale_a = copy("scale_a")
    scale_b = copy("scale_b")
    moe_weights = copy("moe_weights")
    token_ids = copy("token_ids")
    expert_ids = copy("expert_ids")
    em = int(a.shape[0])
    n = int(b_col_major.shape[1])
    out = torch.empty((em, n), dtype=torch.bfloat16, device=device)
    topk = int(dataset.topk)
    arguments = (
        a,
        b_col_major,
        scale_a,
        scale_b,
        moe_weights,
        token_ids,
        expert_ids,
        topk,
        out,
    )
    return {
        "a": a,
        "b_col_major": b_col_major,
        "scale_a": scale_a,
        "scale_b": scale_b,
        "moe_weights": moe_weights,
        "token_ids": token_ids,
        "expert_ids": expert_ids,
        "topk": topk,
        "out": out,
        "arguments": arguments,
    }


def _torch_reference(torch: Any, tensors: Mapping[str, Any]) -> Any:
    a = tensors["a"]
    b = tensors["b_col_major"]
    scale_a = tensors["scale_a"]
    scale_b = tensors["scale_b"]
    moe_weights = tensors["moe_weights"]
    expert_ids = tensors["expert_ids"]
    em = int(a.shape[0])
    n = int(b.shape[1])
    expected = torch.empty((em, n), dtype=torch.bfloat16, device=a.device)

    for tile in range(em // 128):
        row_start = tile * 128
        row_end = row_start + 128
        expert = int(expert_ids[tile].item())
        a_tile = a[row_start:row_end].contiguous()
        b_tile = b[expert].transpose(0, 1).contiguous()
        if hasattr(torch, "_int_mm"):
            accumulator = torch._int_mm(a_tile, b_tile)
        else:
            accumulator = torch.matmul(a_tile.to(torch.int32), b_tile.to(torch.int32))
        scaled = (
            accumulator.to(torch.float32)
            * scale_a[row_start:row_end, None]
            * scale_b[expert, None, :]
            * moe_weights[row_start:row_end, None]
        )
        expected[row_start:row_end].copy_(scaled.to(torch.bfloat16))
    return expected


def _matched_ratio(torch: Any, actual: Any, expected: Any) -> float:
    close = torch.isclose(
        actual.to(torch.float32),
        expected.to(torch.float32),
        rtol=2e-2,
        atol=5e-3,
    )
    return float(close.to(torch.float32).mean().item())


def _synchronize(torch: Any) -> None:
    for api_name in ("cuda", "maca"):
        api = getattr(torch, api_name, None)
        if api is not None and bool(api.is_available()):
            api.synchronize()
            return


def _benchmark_candidate(
    torch: Any,
    run_kernel: Any,
    arguments: Sequence[Any],
) -> list[float]:
    for _ in range(WARMUP_ITERATIONS):
        run_kernel(*arguments)
    _synchronize(torch)

    samples: list[float] = []
    for _round in range(MEASUREMENT_ROUNDS):
        for _sample in range(SAMPLES_PER_ROUND):
            _synchronize(torch)
            started = time.perf_counter_ns()
            run_kernel(*arguments)
            _synchronize(torch)
            elapsed_us = (time.perf_counter_ns() - started) / 1_000.0
            samples.append(elapsed_us)
    return samples


def _timed_invocation(
    torch: Any,
    run_kernel: Any,
    arguments: Sequence[Any],
) -> float:
    _synchronize(torch)
    started = time.perf_counter_ns()
    run_kernel(*arguments)
    _synchronize(torch)
    return (time.perf_counter_ns() - started) / 1_000.0


def _benchmark_interleaved(
    torch: Any,
    candidate: Any,
    candidate_arguments: Sequence[Any],
    baseline: Any,
    baseline_arguments: Sequence[Any],
) -> tuple[list[float], list[float]]:
    """Measure current-best and candidate in alternating AB/BA order."""

    for iteration in range(WARMUP_ITERATIONS):
        if iteration % 2 == 0:
            baseline(*baseline_arguments)
            candidate(*candidate_arguments)
        else:
            candidate(*candidate_arguments)
            baseline(*baseline_arguments)
    _synchronize(torch)

    candidate_samples: list[float] = []
    baseline_samples: list[float] = []
    for round_index in range(MEASUREMENT_ROUNDS):
        for sample_index in range(SAMPLES_PER_ROUND):
            if (round_index + sample_index) % 2 == 0:
                candidate_samples.append(
                    _timed_invocation(torch, candidate, candidate_arguments)
                )
                baseline_samples.append(
                    _timed_invocation(torch, baseline, baseline_arguments)
                )
            else:
                baseline_samples.append(
                    _timed_invocation(torch, baseline, baseline_arguments)
                )
                candidate_samples.append(
                    _timed_invocation(torch, candidate, candidate_arguments)
                )
    return candidate_samples, baseline_samples


def _clone_readonly_inputs(tensors: Mapping[str, Any]) -> dict[str, Any]:
    """Clone all contractually read-only tensors for post-run verification."""

    names = (
        "a",
        "b_col_major",
        "scale_a",
        "scale_b",
        "moe_weights",
        "token_ids",
        "expert_ids",
    )
    return {name: tensors[name].clone() for name in names}


def _readonly_inputs_unchanged(
    torch: Any,
    tensors: Mapping[str, Any],
    snapshots: Mapping[str, Any],
) -> tuple[bool, str | None]:
    for name, snapshot in snapshots.items():
        if not bool(torch.equal(tensors[name], snapshot)):
            return False, name
    return True, None


def _fresh_output_check(
    torch: Any,
    run_kernel: Any,
    arguments: Sequence[Any],
    expected: Any,
) -> float:
    """Ensure a post-timing invocation computes output from fresh storage."""

    output = arguments[-1]
    output.fill_(float("nan"))
    run_kernel(*arguments)
    _synchronize(torch)
    return _matched_ratio(torch, output, expected)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sample")
    ordered = sorted(float(value) for value in values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _failed_c500_result(
    candidate_hash: str,
    cases: Sequence[CaseResult],
    health: BackendResult,
    status: str,
) -> BackendResult:
    error = cases[-1].error if cases else "C500 evaluation failed"
    return BackendResult(
        status=status,
        eligible_for_promotion=False,
        candidate_hash=candidate_hash,
        cases=tuple(cases),
        aggregate_score=None,
        error=error,
        environment=health.environment,
        benchmark_config=health.benchmark_config,
        backend="c500",
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)
