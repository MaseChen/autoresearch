"""Evaluator-side probe for a compiled XPU-OJ scoring baseline."""

from __future__ import annotations

import re
import time
import traceback
from typing import Any, Iterable

from .compiled_reference import (
    SCORING_COMPILER_CONFIG,
    compile_scoring_reference,
    make_scoring_reference,
    scoring_reference_source_sha256,
)
from .device_timing import (
    benchmark_device_event_interleaved,
    device_event_protocol_snapshot,
)
from .platform.canonical import canonical_sha256


SCORING_BASELINE_PROBE_SCHEMA_VERSION = 1
MIN_MATCHED_RATIO = 0.99
MAX_COMPILED_TO_EAGER_RATIO = 1.01


def _selected_arguments(tensors: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tensors["a"],
        tensors["b_col_major"],
        tensors["scale_a"],
        tensors["scale_b"],
        tensors["moe_weights"],
        tensors["expert_ids"],
    )


def run_scoring_baseline_probe(
    *,
    scoring_framework_git_commit: str,
    torch_module: Any | None = None,
    case_specs: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Compile, validate, and measure the scoring reference once.

    The normal entrypoint imports Torch only inside the evaluator container.
    Test injection is deliberately keyword-only and is not exposed by the CLI.
    """

    if (
        not isinstance(scoring_framework_git_commit, str)
        or not re.fullmatch(r"[0-9a-f]{40}", scoring_framework_git_commit)
    ):
        raise ValueError(
            "scoring_framework_git_commit must be 40 lowercase hex digits"
        )
    try:
        if torch_module is None:
            import torch as torch_module  # type: ignore[no-redef]

        from .backends import (
            C500Backend,
            _clone_readonly_inputs,
            _copy_dataset_to_device,
            _matched_ratio,
            _readonly_inputs_unchanged,
            _synchronize,
            _torch_reference,
        )
        from .cases import FULL_CASES, generate_case

        cases = tuple(FULL_CASES if case_specs is None else case_specs)
        if not cases:
            raise ValueError("scoring baseline probe requires at least one case")
        health = C500Backend().doctor(compile_probe=False)
        if health.status != "SUCCESS":
            raise RuntimeError("C500 evaluator environment is not ready")
        eager_reference = make_scoring_reference(torch_module)
        compile_started = time.perf_counter()
        compiled_reference = compile_scoring_reference(torch_module)
        compiler_factory_seconds = time.perf_counter() - compile_started
        case_results: list[dict[str, Any]] = []
        for spec in cases:
            dataset = generate_case(spec)
            tensors = _copy_dataset_to_device(
                torch_module, dataset, str(health.environment["device"])
            )
            snapshots = _clone_readonly_inputs(tensors)
            expected = _torch_reference(torch_module, tensors)
            arguments = _selected_arguments(tensors)

            first_started = time.perf_counter()
            compiled_output = compiled_reference(*arguments)
            _synchronize(torch_module)
            first_invocation_seconds = time.perf_counter() - first_started
            eager_output = eager_reference(*arguments)
            _synchronize(torch_module)
            compiled_ratio = _matched_ratio(torch_module, compiled_output, expected)
            eager_ratio = _matched_ratio(torch_module, eager_output, expected)
            if compiled_ratio < MIN_MATCHED_RATIO or eager_ratio < MIN_MATCHED_RATIO:
                raise RuntimeError(
                    f"compiled scoring reference failed correctness for {spec.name}"
                )
            if str(compiled_output.device) != str(tensors["a"].device):
                raise RuntimeError("compiled scoring reference returned another device")

            measurement = benchmark_device_event_interleaved(
                torch_module,
                compiled_reference,
                arguments,
                eager_reference,
                arguments,
            )
            compiled_to_eager = (
                measurement.candidate_median_ms / measurement.incumbent_median_ms
            )
            if compiled_to_eager > MAX_COMPILED_TO_EAGER_RATIO:
                raise RuntimeError(
                    f"compiled scoring reference is slower than eager for {spec.name}"
                )
            unchanged, modified = _readonly_inputs_unchanged(
                torch_module, tensors, snapshots
            )
            if not unchanged:
                raise RuntimeError(f"scoring reference modified input {modified}")
            case_results.append(
                {
                    "case_id": str(spec.name),
                    "matched_ratio": compiled_ratio,
                    "eager_matched_ratio": eager_ratio,
                    "compiler_first_invocation_seconds": first_invocation_seconds,
                    "compiled_to_eager_ratio": compiled_to_eager,
                    "measurement": measurement.to_dict(),
                }
            )
            del (
                arguments,
                compiled_output,
                eager_output,
                expected,
                snapshots,
                tensors,
                dataset,
            )
        environment = dict(health.environment)
        payload: dict[str, Any] = {
            "schema_version": SCORING_BASELINE_PROBE_SCHEMA_VERSION,
            "command": "score-baseline-probe",
            "status": "QUALIFIED",
            "protocol_id": "xpuoj-th0-proxy-v1",
            "scoring_framework_git_commit": scoring_framework_git_commit,
            "reference_source_sha256": scoring_reference_source_sha256(),
            "compiler_config": dict(SCORING_COMPILER_CONFIG),
            "compiler_factory_seconds": compiler_factory_seconds,
            "compiler_proof": {
                "fullgraph_fail_closed": True,
                "dynamic_shapes": False,
                "output_device_match": True,
                "compile_time_included": False,
            },
            "timing_protocol": device_event_protocol_snapshot(),
            "environment": environment,
            "environment_snapshot_digest": canonical_sha256(environment),
            "cases": case_results,
        }
        return payload
    except Exception as exc:
        return {
            "schema_version": SCORING_BASELINE_PROBE_SCHEMA_VERSION,
            "command": "score-baseline-probe",
            "status": "UNQUALIFIED",
            "protocol_id": "xpuoj-th0-proxy-v1",
            "scoring_framework_git_commit": scoring_framework_git_commit,
            "reference_source_sha256": scoring_reference_source_sha256(),
            "compiler_config": dict(SCORING_COMPILER_CONFIG),
            "timing_protocol": device_event_protocol_snapshot(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
            "cases": [],
        }


__all__ = [
    "MAX_COMPILED_TO_EAGER_RATIO",
    "MIN_MATCHED_RATIO",
    "SCORING_BASELINE_PROBE_SCHEMA_VERSION",
    "run_scoring_baseline_probe",
]
