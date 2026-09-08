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
from .scoring_measurement import (
    MAX_COMPILED_TO_EAGER_RATIO,
    MIN_SCORING_REFERENCE_MATCHED_RATIO,
    scoring_baseline_measurement_contract_snapshot,
    validate_device_event_measurement,
)


SCORING_BASELINE_PROBE_SCHEMA_VERSION = 1
MIN_MATCHED_RATIO = MIN_SCORING_REFERENCE_MATCHED_RATIO


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
        case_contexts: list[dict[str, Any]] = []
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
            compiled_ratio = _matched_ratio(torch_module, compiled_output, expected)
            if compiled_ratio < MIN_MATCHED_RATIO:
                raise RuntimeError(
                    f"compiled scoring reference failed correctness for {spec.name}"
                )
            if str(compiled_output.device) != str(tensors["a"].device):
                raise RuntimeError("compiled scoring reference returned another device")

            anchor_measurement = benchmark_device_event_interleaved(
                torch_module,
                compiled_reference,
                arguments,
                compiled_reference,
                arguments,
            )
            anchor_measurement_value = anchor_measurement.to_dict()
            _, _, anchor_median_ms = validate_device_event_measurement(
                anchor_measurement_value,
                field="anchor_measurement",
            )
            unchanged, modified = _readonly_inputs_unchanged(
                torch_module, tensors, snapshots
            )
            if not unchanged:
                raise RuntimeError(f"scoring reference modified input {modified}")
            result = {
                "case_id": str(spec.name),
                "matched_ratio": compiled_ratio,
                "compiler_first_invocation_seconds": first_invocation_seconds,
                "anchor_median_ms": anchor_median_ms,
                "anchor_measurement": anchor_measurement_value,
            }
            case_results.append(result)
            case_contexts.append(
                {
                    "spec": spec,
                    "dataset": dataset,
                    "tensors": tensors,
                    "snapshots": snapshots,
                    "expected": expected,
                    "arguments": arguments,
                    "compiled_output": compiled_output,
                    "result": result,
                }
            )

        # Keep the absolute-anchor phase free from the substantially slower
        # eager workload.  The eager channel proves only that full-graph
        # compilation is a valid non-regressing baseline implementation.
        for context in case_contexts:
            spec = context["spec"]
            arguments = context["arguments"]
            eager_output = eager_reference(*arguments)
            _synchronize(torch_module)
            eager_ratio = _matched_ratio(
                torch_module, eager_output, context["expected"]
            )
            if eager_ratio < MIN_MATCHED_RATIO:
                raise RuntimeError(
                    f"eager scoring reference failed correctness for {spec.name}"
                )
            performance_measurement = benchmark_device_event_interleaved(
                torch_module,
                compiled_reference,
                arguments,
                eager_reference,
                arguments,
            )
            performance_measurement_value = performance_measurement.to_dict()
            performance_candidate_ms, performance_incumbent_ms, _ = (
                validate_device_event_measurement(
                    performance_measurement_value,
                    field="performance_measurement",
                )
            )
            compiled_to_eager = performance_candidate_ms / performance_incumbent_ms
            if compiled_to_eager > MAX_COMPILED_TO_EAGER_RATIO:
                raise RuntimeError(
                    f"compiled scoring reference is slower than eager for {spec.name}"
                )
            unchanged, modified = _readonly_inputs_unchanged(
                torch_module, context["tensors"], context["snapshots"]
            )
            if not unchanged:
                raise RuntimeError(f"scoring reference modified input {modified}")
            context["result"].update(
                {
                    "eager_matched_ratio": eager_ratio,
                    "compiled_to_eager_ratio": compiled_to_eager,
                    "performance_measurement": performance_measurement_value,
                }
            )
            del eager_output
        environment = dict(health.environment)
        payload: dict[str, Any] = {
            "schema_version": SCORING_BASELINE_PROBE_SCHEMA_VERSION,
            "command": "score-baseline-probe",
            "status": "QUALIFIED",
            "protocol_id": "xpuoj-th0-proxy-v1",
            "scoring_framework_git_commit": scoring_framework_git_commit,
            "reference_source_sha256": scoring_reference_source_sha256(),
            "compiler_config": dict(SCORING_COMPILER_CONFIG),
            "measurement_contract": (
                scoring_baseline_measurement_contract_snapshot()
            ),
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
            "measurement_contract": (
                scoring_baseline_measurement_contract_snapshot()
            ),
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
