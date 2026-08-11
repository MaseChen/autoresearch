"""Trusted V2 adapters around the existing Fused-MoE/C500 implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..backends import MEASUREMENT_ROUNDS, SAMPLES_PER_ROUND, WARMUP_ITERATIONS
from ..constants import (
    CURRENT_C500_EVALUATION_PROTOCOL_ID,
    LEGACY_C500_EVALUATION_PROTOCOL_ID,
)
from ..contract import validate_source
from ..research_policy import validate_research_candidate_bounded
from ..scoring import MAX_CASE_REGRESSION, MIN_AGGREGATE_SPEEDUP
from .components import (
    CaseRole,
    EvaluationProtocolDefinition,
    ReplicateKind,
    StageDefinition,
    TargetComponents,
    TrustedComponentRegistry,
)
from .profiles import (
    BUILTIN_PROFILE_REGISTRY,
    CURRENT_RESEARCH_NAMESPACE,
    LEGACY_RESEARCH_NAMESPACE,
    ProfileDefinition,
    ResearchNamespace,
)
from .proposal import BundleLimits, TRITON_PYTHON_BUNDLE_LIMITS


LEGACY_PROTOCOL_ID = LEGACY_C500_EVALUATION_PROTOCOL_ID
CURRENT_PROTOCOL_ID = CURRENT_C500_EVALUATION_PROTOCOL_ID

_OPERATOR_DEFINITION = BUILTIN_PROFILE_REGISTRY.resolve(
    LEGACY_RESEARCH_NAMESPACE.operator
)
_LANGUAGE_DEFINITION = BUILTIN_PROFILE_REGISTRY.resolve(
    LEGACY_RESEARCH_NAMESPACE.language
)
_EVALUATOR_DEFINITION = BUILTIN_PROFILE_REGISTRY.resolve(
    LEGACY_RESEARCH_NAMESPACE.evaluator
)
_LEGACY_PROTOCOL_DEFINITION = BUILTIN_PROFILE_REGISTRY.resolve(
    LEGACY_RESEARCH_NAMESPACE.evaluation_protocol
)
_CURRENT_PROTOCOL_DEFINITION = BUILTIN_PROFILE_REGISTRY.resolve(
    CURRENT_RESEARCH_NAMESPACE.evaluation_protocol
)
_PROMOTION_DEFINITION = BUILTIN_PROFILE_REGISTRY.resolve(
    LEGACY_RESEARCH_NAMESPACE.promotion_policy
)


def _language_bundle_limits() -> BundleLimits:
    config = _LANGUAGE_DEFINITION.config
    limits = BundleLimits(
        max_files=int(config["max_files"]),
        max_file_bytes=int(config["max_file_bytes"]),
        max_total_bytes=int(config["max_total_bytes"]),
        allowed_extensions=frozenset(config["allowed_extensions"]),
        allowed_media_types=frozenset(config["allowed_media_types"]),
        required_entrypoint=str(config["entrypoint"]),
    )
    if limits != TRITON_PYTHON_BUNDLE_LIMITS:
        raise RuntimeError(
            "Triton language profile disagrees with its bundle policy"
        )
    return TRITON_PYTHON_BUNDLE_LIMITS


_LANGUAGE_BUNDLE_LIMITS = _language_bundle_limits()


@dataclass(frozen=True, slots=True)
class FusedMoeW8A8TnPack:
    operator_id: str = _OPERATOR_DEFINITION.ref.id
    implementation_id: str = _OPERATOR_DEFINITION.implementation_id
    revision: str = _OPERATOR_DEFINITION.ref.revision

    def contract_summary(self) -> Mapping[str, Any]:
        return {
            "entrypoint": "run_kernel",
            "topk": 8,
            "expert_tile_rows": 128,
            "accumulator": "int32",
            "epilogue": "fp32",
            "output": "bf16",
        }

    def prompt_context(self) -> str:
        return (
            "Optimize the W8A8 TN Fused MoE kernel while preserving routed "
            "128-row expert tiles and an int64-safe B expert base."
        )

    def validate_operator_candidate(self, source: str) -> Any:
        return validate_source(source)


@dataclass(frozen=True, slots=True)
class TritonPythonAdapter:
    language_id: str = _LANGUAGE_DEFINITION.ref.id
    implementation_id: str = _LANGUAGE_DEFINITION.implementation_id
    revision: str = _LANGUAGE_DEFINITION.ref.revision
    entrypoint: str = str(_LANGUAGE_DEFINITION.config["entrypoint"])
    bundle_limits: BundleLimits = _LANGUAGE_BUNDLE_LIMITS

    def allowed_paths(self) -> Sequence[str]:
        return (self.entrypoint,)

    def validate_language_candidate(self, source: str) -> Any:
        # The legacy policy combines Python/Triton safety, MetaX compiler
        # constraints and the Fused-MoE int64 invariant.  Keeping this wrapper
        # preserves behavior while later adapters expose narrower policies.
        return validate_research_candidate_bounded(source)

    def compile_cache_material(self) -> Mapping[str, Any]:
        return {
            "language_id": self.language_id,
            "adapter_revision": self.revision,
            "entrypoint": self.entrypoint,
        }


@dataclass(frozen=True, slots=True)
class MetaXC500Backend:
    device_id: str = _EVALUATOR_DEFINITION.ref.id
    implementation_id: str = _EVALUATOR_DEFINITION.implementation_id
    revision: str = _EVALUATOR_DEFINITION.ref.revision

    def evaluator_backend(self) -> str:
        backend = _EVALUATOR_DEFINITION.config.get("backend")
        if not isinstance(backend, str) or not backend:
            raise RuntimeError("MetaX evaluator profile has no backend binding")
        return backend

    def resource_requirements(self) -> Mapping[str, Any]:
        return {
            "exclusive_accelerator": True,
            "device_nodes": (
                "/dev/mxcd",
                "/dev/dri/card2",
                "/dev/dri/renderD129",
            ),
        }

    def fatal_error_markers(self) -> Sequence[str]:
        return ("atu", "xnack", "illegal address", "exit 137")


@dataclass(frozen=True, slots=True)
class CurrentPromotionPolicy:
    policy_id: str = _PROMOTION_DEFINITION.ref.id
    implementation_id: str = _PROMOTION_DEFINITION.implementation_id
    revision: str = _PROMOTION_DEFINITION.ref.revision

    def thresholds(self) -> Mapping[str, float]:
        return {
            "min_aggregate_speedup": MIN_AGGREGATE_SPEEDUP,
            "max_case_regression": MAX_CASE_REGRESSION,
            "min_matched_ratio": 0.99,
        }


def _protocol(
    definition: ProfileDefinition,
) -> EvaluationProtocolDefinition:
    config = definition.config
    suites = {
        str(suite_id): tuple(str(case_id) for case_id in case_ids)
        for suite_id, case_ids in config["suite_case_ids"].items()
    }
    roles = {
        str(case_id): CaseRole(str(role))
        for case_id, role in config["case_roles"].items()
    }
    stages = tuple(
        StageDefinition(
            stage_id=str(stage["stage_id"]),
            suite_id=(
                None if stage["suite_id"] is None else str(stage["suite_id"])
            ),
            replicate_kind=ReplicateKind(str(stage["replicate_kind"])),
            requires_baseline=bool(stage["requires_baseline"]),
            may_request_confirmation=bool(stage["may_request_confirmation"]),
        )
        for stage in config["stages"]
    )
    measurements = (
        int(config["warmup_iterations"]),
        int(config["measurement_rounds"]),
        int(config["samples_per_round"]),
    )
    if measurements != (
        WARMUP_ITERATIONS,
        MEASUREMENT_ROUNDS,
        SAMPLES_PER_ROUND,
    ):
        raise RuntimeError(
            "evaluation protocol profile disagrees with evaluator constants"
        )
    return EvaluationProtocolDefinition(
        protocol_id=definition.ref.id,
        revision=definition.ref.revision,
        implementation_id=definition.implementation_id,
        stages=stages,
        suite_cases=suites,
        case_roles=roles,
        warmup_iterations=measurements[0],
        measurement_rounds=measurements[1],
        samples_per_round=measurements[2],
        interleaved_baseline=bool(config["interleaved_baseline"]),
    )


LEGACY_C500_PROTOCOL = _protocol(
    _LEGACY_PROTOCOL_DEFINITION,
)
CURRENT_C500_PROTOCOL = _protocol(
    _CURRENT_PROTOCOL_DEFINITION,
)
_OPERATOR = FusedMoeW8A8TnPack()
_LANGUAGE = TritonPythonAdapter()
_DEVICE = MetaXC500Backend()
_PROMOTION = CurrentPromotionPolicy()
_EXPECTED_PROMOTION_THRESHOLDS = {
    key: float(value)
    for key, value in _PROMOTION_DEFINITION.config.items()
    if key != "confirmation_required"
}
if dict(_PROMOTION.thresholds()) != _EXPECTED_PROMOTION_THRESHOLDS:
    raise RuntimeError(
        "promotion policy profile disagrees with its trusted component"
    )
LEGACY_TARGET = TargetComponents(
    operator=_OPERATOR,
    language=_LANGUAGE,
    device=_DEVICE,
    protocol=LEGACY_C500_PROTOCOL,
    promotion=_PROMOTION,
)
CURRENT_TARGET = TargetComponents(
    operator=_OPERATOR,
    language=_LANGUAGE,
    device=_DEVICE,
    protocol=CURRENT_C500_PROTOCOL,
    promotion=_PROMOTION,
)


def _build_component_registry() -> TrustedComponentRegistry:
    registry = TrustedComponentRegistry(
        profile_registry=BUILTIN_PROFILE_REGISTRY
    )
    for definition, component in (
        (_OPERATOR_DEFINITION, CURRENT_TARGET.operator),
        (_LANGUAGE_DEFINITION, CURRENT_TARGET.language),
        (_EVALUATOR_DEFINITION, CURRENT_TARGET.device),
        (_LEGACY_PROTOCOL_DEFINITION, LEGACY_TARGET.protocol),
        (_CURRENT_PROTOCOL_DEFINITION, CURRENT_TARGET.protocol),
        (_PROMOTION_DEFINITION, CURRENT_TARGET.promotion),
    ):
        registry.bind_profile(definition, component=component)
    registry.seal()
    return registry


_BUILTIN_COMPONENT_REGISTRY = _build_component_registry()


def builtin_component_registry() -> TrustedComponentRegistry:
    return _BUILTIN_COMPONENT_REGISTRY


def resolve_builtin_target(
    namespace: ResearchNamespace = CURRENT_RESEARCH_NAMESPACE,
) -> TargetComponents:
    """Resolve a complete target only from exact built-in profile refs."""

    return _BUILTIN_COMPONENT_REGISTRY.resolve_target(namespace)


__all__ = [
    "CURRENT_C500_PROTOCOL",
    "CURRENT_PROTOCOL_ID",
    "CURRENT_TARGET",
    "CurrentPromotionPolicy",
    "FusedMoeW8A8TnPack",
    "LEGACY_C500_PROTOCOL",
    "LEGACY_PROTOCOL_ID",
    "LEGACY_TARGET",
    "MetaXC500Backend",
    "TritonPythonAdapter",
    "builtin_component_registry",
    "resolve_builtin_target",
]
