"""Evaluator-side device-event probe for one shadow-scored candidate."""

from __future__ import annotations

import gc
import math
import re
import traceback
from pathlib import Path
from typing import Any, Iterable

from .contract import validate_candidate
from .device_timing import (
    benchmark_device_event_interleaved,
    device_event_protocol_snapshot,
)
from .platform.canonical import canonical_sha256
from .scoring_candidate_measurement import (
    SCORING_CANDIDATE_WORKER_REVISION,
    scoring_candidate_measurement_contract_snapshot,
)
from .scoring_measurement import validate_device_event_measurement


SCORING_CANDIDATE_PROBE_SCHEMA_VERSION = 1


def _release_case(torch_module: Any) -> None:
    gc.collect()
    for name in ("cuda", "maca"):
        api = getattr(torch_module, name, None)
        empty_cache = getattr(api, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()
            return


def _run_kernel(path: Path, source_hash: str) -> Any:
    from .backends import _import_candidate

    module = _import_candidate(path, source_hash)
    run_kernel = getattr(module, "run_kernel", None)
    if not callable(run_kernel):
        raise ValueError(f"{path} has no callable run_kernel")
    return run_kernel


def _qualified_ratio(value: object, minimum: float) -> bool:
    if isinstance(value, bool):
        return False
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(ratio) and minimum <= ratio <= 1.0


def run_scoring_candidate_probe(
    *,
    candidate_path: str | Path,
    incumbent_path: str | Path,
    expected_candidate_hash: str,
    expected_incumbent_hash: str,
    scoring_framework_git_commit: str,
    scoring_profile_digest: str,
    expected_measurement_contract_digest: str,
    torch_module: Any | None = None,
    case_specs: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Correctness-check and time candidate/incumbent under one exact clock."""

    for field, value, pattern in (
        ("expected_candidate_hash", expected_candidate_hash, r"[0-9a-f]{64}"),
        ("expected_incumbent_hash", expected_incumbent_hash, r"[0-9a-f]{64}"),
        (
            "scoring_framework_git_commit",
            scoring_framework_git_commit,
            r"[0-9a-f]{40}",
        ),
        ("scoring_profile_digest", scoring_profile_digest, r"sha256:[0-9a-f]{64}"),
        (
            "expected_measurement_contract_digest",
            expected_measurement_contract_digest,
            r"sha256:[0-9a-f]{64}",
        ),
    ):
        if not isinstance(value, str) or not re.fullmatch(pattern, value):
            raise ValueError(f"{field} has an invalid identity")
    contract = scoring_candidate_measurement_contract_snapshot()
    if contract["digest"] != expected_measurement_contract_digest:
        raise ValueError("candidate measurement contract digest mismatch")

    candidate = validate_candidate(Path(candidate_path))
    incumbent = validate_candidate(Path(incumbent_path))
    if candidate.sha256 != expected_candidate_hash:
        raise ValueError("candidate source hash mismatch")
    if incumbent.sha256 != expected_incumbent_hash:
        raise ValueError("incumbent source hash mismatch")

    phase = "pre-gpu-validation"
    gpu_state = "NOT_STARTED"
    completion_trusted = True

    def gpu_call(selected_phase: str, function: Any, *args: Any) -> Any:
        nonlocal phase, gpu_state, completion_trusted
        phase = selected_phase
        gpu_state = "STARTED"
        completion_trusted = False
        value = function(*args)
        gpu_state = "COMPLETED"
        completion_trusted = True
        return value

    try:
        if torch_module is None:
            import torch as torch_module  # type: ignore[no-redef]

        from .backends import (
            C500Backend,
            _clone_readonly_inputs,
            _copy_dataset_to_device,
            _fresh_output_check,
            _matched_ratio,
            _readonly_inputs_unchanged,
            _synchronize,
            _torch_reference,
        )
        from .cases import FULL_CASES, generate_case

        cases = tuple(FULL_CASES if case_specs is None else case_specs)
        if not cases:
            raise ValueError("candidate scoring probe requires full cases")
        health = C500Backend().doctor(compile_probe=False)
        if health.status != "SUCCESS":
            raise RuntimeError("C500 evaluator environment is not ready")
        environment = dict(health.environment)
        if canonical_sha256(environment) != contract[
            "qualified_probe_environment_snapshot_digest"
        ]:
            raise RuntimeError(
                "C500 evaluator environment differs from qualified scoring baseline"
            )
        device = str(health.environment["device"])
        candidate_run = _run_kernel(Path(candidate_path), candidate.sha256)
        incumbent_run = _run_kernel(Path(incumbent_path), incumbent.sha256)
        by_case: dict[str, dict[str, Any]] = {}

        # Objective phase: exactly mirror the qualified baseline's self-pair.
        for spec in cases:
            dataset = generate_case(spec)
            tensors = gpu_call(
                f"anchor:{spec.name}:copy", _copy_dataset_to_device,
                torch_module, dataset, device,
            )
            snapshots = gpu_call(
                f"anchor:{spec.name}:snapshot", _clone_readonly_inputs, tensors
            )
            expected = gpu_call(
                f"anchor:{spec.name}:reference", _torch_reference,
                torch_module, tensors,
            )
            arguments = tensors["arguments"]
            gpu_call(f"anchor:{spec.name}:candidate", candidate_run, *arguments)
            gpu_call(f"anchor:{spec.name}:synchronize", _synchronize, torch_module)
            matched = gpu_call(
                f"anchor:{spec.name}:correctness", _matched_ratio,
                torch_module, tensors["out"], expected,
            )
            if not _qualified_ratio(
                matched, float(contract["minimum_correctness_ratio"])
            ):
                raise RuntimeError(
                    f"candidate failed scoring correctness for {spec.name}"
                )
            measurement = gpu_call(
                f"anchor:{spec.name}:timing",
                benchmark_device_event_interleaved,
                torch_module, candidate_run, arguments, candidate_run, arguments,
            )
            measurement_value = measurement.to_dict()
            _, _, anchor_median_ms = validate_device_event_measurement(
                measurement_value, field="candidate_anchor_measurement"
            )
            fresh_ratio = gpu_call(
                f"anchor:{spec.name}:fresh-output", _fresh_output_check,
                torch_module, candidate_run, arguments, expected,
            )
            unchanged, modified = gpu_call(
                f"anchor:{spec.name}:readonly", _readonly_inputs_unchanged,
                torch_module, tensors, snapshots,
            )
            if not _qualified_ratio(
                fresh_ratio, float(contract["minimum_correctness_ratio"])
            ):
                raise RuntimeError(
                    f"candidate failed fresh-output check for {spec.name}"
                )
            if not unchanged:
                raise RuntimeError(f"candidate modified input {modified}")
            by_case[str(spec.name)] = {
                "case_id": str(spec.name),
                "candidate_matched_ratio": matched,
                "candidate_fresh_output_ratio": fresh_ratio,
                "candidate_anchor_median_ms": anchor_median_ms,
                "candidate_anchor_measurement": measurement_value,
            }
            del arguments, expected, snapshots, tensors, dataset
            _release_case(torch_module)

        # Safety phase: re-create each case so the two phase sets never coexist.
        for spec in cases:
            dataset = generate_case(spec)
            tensors = gpu_call(
                f"paired:{spec.name}:copy", _copy_dataset_to_device,
                torch_module, dataset, device,
            )
            snapshots = gpu_call(
                f"paired:{spec.name}:snapshot", _clone_readonly_inputs, tensors
            )
            expected = gpu_call(
                f"paired:{spec.name}:reference", _torch_reference,
                torch_module, tensors,
            )
            candidate_arguments = tensors["arguments"]
            incumbent_output = gpu_call(
                f"paired:{spec.name}:incumbent-output",
                torch_module.empty_like,
                tensors["out"],
            )
            incumbent_arguments = (*candidate_arguments[:-1], incumbent_output)
            gpu_call(
                f"paired:{spec.name}:candidate", candidate_run, *candidate_arguments
            )
            gpu_call(
                f"paired:{spec.name}:incumbent", incumbent_run, *incumbent_arguments
            )
            gpu_call(f"paired:{spec.name}:synchronize", _synchronize, torch_module)
            candidate_ratio = gpu_call(
                f"paired:{spec.name}:candidate-correctness", _matched_ratio,
                torch_module, tensors["out"], expected,
            )
            incumbent_ratio = gpu_call(
                f"paired:{spec.name}:incumbent-correctness", _matched_ratio,
                torch_module, incumbent_output, expected,
            )
            minimum = float(contract["minimum_correctness_ratio"])
            if not _qualified_ratio(candidate_ratio, minimum) or not _qualified_ratio(
                incumbent_ratio, minimum
            ):
                raise RuntimeError(
                    f"paired scoring correctness failed for {spec.name}"
                )
            measurement = gpu_call(
                f"paired:{spec.name}:timing",
                benchmark_device_event_interleaved,
                torch_module, candidate_run, candidate_arguments,
                incumbent_run, incumbent_arguments,
            )
            measurement_value = measurement.to_dict()
            candidate_median_ms, incumbent_median_ms, _ = (
                validate_device_event_measurement(
                    measurement_value, field="paired_safety_measurement"
                )
            )
            candidate_fresh = gpu_call(
                f"paired:{spec.name}:candidate-fresh", _fresh_output_check,
                torch_module, candidate_run, candidate_arguments, expected,
            )
            incumbent_fresh = gpu_call(
                f"paired:{spec.name}:incumbent-fresh", _fresh_output_check,
                torch_module, incumbent_run, incumbent_arguments, expected,
            )
            unchanged, modified = gpu_call(
                f"paired:{spec.name}:readonly", _readonly_inputs_unchanged,
                torch_module, tensors, snapshots,
            )
            if not _qualified_ratio(candidate_fresh, minimum) or not _qualified_ratio(
                incumbent_fresh, minimum
            ):
                raise RuntimeError(
                    f"paired fresh-output check failed for {spec.name}"
                )
            if not unchanged:
                raise RuntimeError(f"scoring probe modified input {modified}")
            by_case[str(spec.name)].update(
                {
                    "paired_candidate_matched_ratio": candidate_ratio,
                    "paired_incumbent_matched_ratio": incumbent_ratio,
                    "paired_candidate_fresh_output_ratio": candidate_fresh,
                    "paired_incumbent_fresh_output_ratio": incumbent_fresh,
                    "paired_candidate_median_ms": candidate_median_ms,
                    "paired_incumbent_median_ms": incumbent_median_ms,
                    "paired_safety_measurement": measurement_value,
                }
            )
            del (
                incumbent_arguments,
                incumbent_output,
                candidate_arguments,
                expected,
                snapshots,
                tensors,
                dataset,
            )
            _release_case(torch_module)

        phase = "complete"
        return {
            "schema_version": SCORING_CANDIDATE_PROBE_SCHEMA_VERSION,
            "command": "score-candidate-probe",
            "status": "QUALIFIED",
            "gpu_state": "COMPLETED",
            "completion_trusted": True,
            "phase": phase,
            "protocol_id": "xpuoj-th0-proxy-v1",
            "worker_revision": SCORING_CANDIDATE_WORKER_REVISION,
            "scoring_framework_git_commit": scoring_framework_git_commit,
            "scoring_profile_digest": scoring_profile_digest,
            "candidate_hash": candidate.sha256,
            "incumbent_hash": incumbent.sha256,
            "measurement_contract": contract,
            "timing_protocol": device_event_protocol_snapshot(),
            "environment": environment,
            "environment_snapshot_digest": canonical_sha256(environment),
            "cases": [by_case[str(spec.name)] for spec in cases],
        }
    except Exception as exc:
        status = (
            "UNKNOWN_OUTCOME"
            if gpu_state == "STARTED" and not completion_trusted
            else "UNQUALIFIED"
        )
        return {
            "schema_version": SCORING_CANDIDATE_PROBE_SCHEMA_VERSION,
            "command": "score-candidate-probe",
            "status": status,
            "gpu_state": gpu_state,
            "completion_trusted": completion_trusted,
            "phase": phase,
            "protocol_id": "xpuoj-th0-proxy-v1",
            "worker_revision": SCORING_CANDIDATE_WORKER_REVISION,
            "scoring_framework_git_commit": scoring_framework_git_commit,
            "scoring_profile_digest": scoring_profile_digest,
            "candidate_hash": candidate.sha256,
            "incumbent_hash": incumbent.sha256,
            "measurement_contract": contract,
            "timing_protocol": device_event_protocol_snapshot(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
            "cases": [],
        }


__all__ = [
    "SCORING_CANDIDATE_PROBE_SCHEMA_VERSION",
    "run_scoring_candidate_probe",
]
