"""RunSpecV2 resolution into an immutable, resumable workflow snapshot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import re
from types import MappingProxyType
from typing import Any

from .canonical import canonical_json_text, canonical_sha256
from .identity import (
    BaselineRef,
    ExecutionEnvironmentDigest,
    RUN_MODES,
)
from .profiles import (
    BUILTIN_PROFILE_REGISTRY,
    LEGACY_RESEARCH_NAMESPACE,
    ProfileDefinition,
    ProfileRef,
    ProfileRegistry,
    ResearchNamespace,
)


RUN_SPEC_SCHEMA_VERSION = 2
EXECUTION_ENVIRONMENT_BINDING_KEY = "execution_environment"
EXECUTION_ENVIRONMENT_BINDING_FIELDS = frozenset(
    {
        "evaluator_image",
        "toolchain",
        "framework",
        "operator_abi",
        "build_flags",
    }
)
MAX_RUN_CANDIDATES = 5
MAX_RUN_WALL_SECONDS = 6 * 60 * 60
MAX_RUN_CONSECUTIVE_FAILURES = 3
_SECRET_VALUE_KEYS = frozenset(
    {
        "api_key",
        "api_key_value",
        "access_token",
        "password",
        "secret",
        "secret_value",
        "credential_value",
        "auth_token",
        "bearer_token",
        "client_secret",
        "private_key",
    }
)
_STABLE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


def _plain(value: Any) -> Any:
    return json.loads(canonical_json_text(value))


def _frozen(value: Any) -> Any:
    normalized = _plain(value)

    def freeze(child: Any) -> Any:
        if isinstance(child, dict):
            return MappingProxyType({key: freeze(item) for key, item in child.items()})
        if isinstance(child, list):
            return tuple(freeze(item) for item in child)
        return child

    return freeze(normalized)


def _require_ref(value: ProfileRef, kind: str) -> None:
    if not isinstance(value, ProfileRef) or value.kind != kind:
        raise ValueError(f"{kind} must be a {kind} ProfileRef")


def _optional_identifier(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _STABLE_IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} must be a stable bounded identifier")
    return value


def _history_cutoff(value: object) -> str | int | None:
    if value is None:
        return None
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and 0 < len(value) <= 256:
        return value
    raise ValueError(
        "history_cutoff must be null, a non-negative integer or bounded token"
    )


def _scan_for_secret_values(value: Any, *, path: str = "runtime_bindings") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_string = str(key)
            if key_string.lower() in _SECRET_VALUE_KEYS:
                raise ValueError(
                    "secret content field is forbidden in snapshot: "
                    f"{path}.{key_string}"
                )
            _scan_for_secret_values(child, path=f"{path}.{key_string}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _scan_for_secret_values(child, path=f"{path}[{index}]")


def _execution_environment_from_runtime(
    runtime_bindings: Mapping[str, Any],
) -> ExecutionEnvironmentDigest | None:
    value = runtime_bindings.get(EXECUTION_ENVIRONMENT_BINDING_KEY)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            "runtime execution_environment must be a JSON object"
        )
    unknown = sorted(set(value) - EXECUTION_ENVIRONMENT_BINDING_FIELDS)
    missing = sorted(EXECUTION_ENVIRONMENT_BINDING_FIELDS - set(value))
    if unknown:
        raise ValueError(
            "runtime execution_environment contains unknown fields: "
            + ", ".join(unknown)
        )
    if missing:
        raise ValueError(
            "runtime execution_environment is missing fields: "
            + ", ".join(missing)
        )
    return ExecutionEnvironmentDigest.from_bindings(
        evaluator_image=value["evaluator_image"],
        toolchain=value["toolchain"],
        framework=value["framework"],
        operator_abi=value["operator_abi"],
        build_flags=value["build_flags"],
    )


def _validate_resolved_definitions(
    spec: "RunSpecV2", definitions: Mapping[str, Any]
) -> None:
    references = {
        "deployment": spec.deployment,
        "operator": spec.operator,
        "language": spec.language,
        "evaluator": spec.evaluator,
        "evaluation_protocol": spec.evaluation_protocol,
        "promotion_policy": spec.promotion_policy,
        "proposer": spec.proposer,
    }
    if spec.campaign is not None:
        references["campaign"] = spec.campaign
    if set(definitions) != set(references):
        raise ValueError(
            "resolved profile definitions do not match the RunSpec references"
        )
    definition_fields = {
        "kind",
        "id",
        "revision",
        "digest",
        "implementation_id",
        "config",
    }
    for name, reference in references.items():
        value = definitions[name]
        if not isinstance(value, Mapping) or set(value) != definition_fields:
            raise ValueError(
                f"resolved {name} profile fields do not match the contract"
            )
        parsed_ref = ProfileRef.from_value(
            {key: value[key] for key in ("kind", "id", "revision", "digest")}
        )
        if parsed_ref != reference:
            raise ValueError(
                f"resolved {name} profile does not match its exact reference"
            )
        # Recompute the profile digest from the resolved implementation/config,
        # rather than trusting the copied ProfileRef fields in the snapshot.
        ProfileDefinition(
            ref=parsed_ref,
            implementation_id=value["implementation_id"],
            config=value["config"],
        )


@dataclass(frozen=True, slots=True)
class RunBudget:
    max_candidates: int = MAX_RUN_CANDIDATES
    max_wall_seconds: int = MAX_RUN_WALL_SECONDS
    max_consecutive_failures: int = MAX_RUN_CONSECUTIVE_FAILURES
    stop_after_promotion: bool = True

    def __post_init__(self) -> None:
        for field, maximum in (
            ("max_candidates", MAX_RUN_CANDIDATES),
            ("max_wall_seconds", MAX_RUN_WALL_SECONDS),
            ("max_consecutive_failures", MAX_RUN_CONSECUTIVE_FAILURES),
        ):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= maximum
            ):
                raise ValueError(f"{field} must be between 1 and {maximum}")
        if self.stop_after_promotion is not True:
            raise ValueError("V2 bounded runs must stop after the first promotion")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_candidates": self.max_candidates,
            "max_wall_seconds": self.max_wall_seconds,
            "max_consecutive_failures": self.max_consecutive_failures,
            "stop_after_promotion": self.stop_after_promotion,
        }

    @classmethod
    def from_value(cls, value: object) -> "RunBudget":
        if not isinstance(value, dict):
            raise ValueError("run budget must be a JSON object")
        expected = {
            "max_candidates",
            "max_wall_seconds",
            "max_consecutive_failures",
            "stop_after_promotion",
        }
        if set(value) != expected:
            raise ValueError("run budget fields do not match the V2 contract")
        return cls(**{key: value[key] for key in expected})


@dataclass(frozen=True, slots=True)
class RunSpecV2:
    deployment: ProfileRef
    operator: ProfileRef
    language: ProfileRef
    evaluator: ProfileRef
    evaluation_protocol: ProfileRef
    promotion_policy: ProfileRef
    proposer: ProfileRef
    baseline: BaselineRef
    budget: RunBudget
    mode: str = "DISCOVERY"
    campaign: ProfileRef | None = None
    campaign_id: str | None = None
    cohort_id: str | None = None
    history_cutoff: str | int | None = None

    def __post_init__(self) -> None:
        for field, kind in (
            ("deployment", "deployment"),
            ("operator", "operator"),
            ("language", "language"),
            ("evaluator", "evaluator"),
            ("evaluation_protocol", "evaluation_protocol"),
            ("promotion_policy", "promotion_policy"),
            ("proposer", "proposer"),
        ):
            _require_ref(getattr(self, field), kind)
        if self.campaign is not None:
            _require_ref(self.campaign, "campaign")
        if not isinstance(self.baseline, BaselineRef):
            raise ValueError("baseline must be a BaselineRef")
        if not isinstance(self.budget, RunBudget):
            raise ValueError("budget must be a RunBudget")
        if self.mode not in RUN_MODES:
            raise ValueError("run mode must be BENCHMARK or DISCOVERY")
        object.__setattr__(
            self,
            "campaign_id",
            _optional_identifier(self.campaign_id, field="campaign_id"),
        )
        object.__setattr__(
            self,
            "cohort_id",
            _optional_identifier(self.cohort_id, field="cohort_id"),
        )
        object.__setattr__(
            self, "history_cutoff", _history_cutoff(self.history_cutoff)
        )
        if self.mode == "BENCHMARK" and (
            self.cohort_id is None or self.history_cutoff is None
        ):
            raise ValueError("benchmark RunSpec requires cohort_id and history_cutoff")
        if (self.campaign_id is None) != (self.campaign is None):
            raise ValueError(
                "campaign and campaign_id must either both be set or both be absent"
            )
        if self.campaign is not None and self.baseline.source != "campaign":
            raise ValueError("campaign runs require a Campaign-owned baseline")
        if self.namespace.namespace_id != self.baseline.namespace_id:
            raise ValueError("baseline belongs to a different research namespace")

    @property
    def namespace(self) -> ResearchNamespace:
        return ResearchNamespace.from_profiles(
            operator=self.operator,
            language=self.language,
            evaluator=self.evaluator,
            evaluation_protocol=self.evaluation_protocol,
            promotion_policy=self.promotion_policy,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_SPEC_SCHEMA_VERSION,
            "deployment": self.deployment.to_dict(),
            "operator": self.operator.to_dict(),
            "language": self.language.to_dict(),
            "evaluator": self.evaluator.to_dict(),
            "evaluation_protocol": self.evaluation_protocol.to_dict(),
            "promotion_policy": self.promotion_policy.to_dict(),
            "proposer": self.proposer.to_dict(),
            "campaign": None if self.campaign is None else self.campaign.to_dict(),
            "campaign_id": self.campaign_id,
            "baseline": self.baseline.to_dict(),
            "budget": self.budget.to_dict(),
            "mode": self.mode,
            "cohort_id": self.cohort_id,
            "history_cutoff": self.history_cutoff,
        }


@dataclass(frozen=True, slots=True)
class ResolvedRunSnapshot:
    spec: RunSpecV2
    definitions: Mapping[str, Mapping[str, Any]]
    runtime_bindings: Mapping[str, Any]
    execution_environment: ExecutionEnvironmentDigest

    def __post_init__(self) -> None:
        if not isinstance(self.spec, RunSpecV2):
            raise ValueError("spec must be RunSpecV2")
        if not isinstance(
            self.execution_environment, ExecutionEnvironmentDigest
        ):
            raise ValueError(
                "execution_environment must be an ExecutionEnvironmentDigest"
            )
        _scan_for_secret_values(self.runtime_bindings)
        plain_definitions = _plain(self.definitions)
        plain_runtime = _plain(self.runtime_bindings)
        _validate_resolved_definitions(self.spec, plain_definitions)
        runtime_environment = _execution_environment_from_runtime(plain_runtime)
        baseline_environment = self.spec.baseline.execution_environment
        if self.execution_environment.is_resolved:
            if runtime_environment is None:
                raise ValueError(
                    "resolved snapshot is missing runtime execution_environment"
                )
            if runtime_environment != self.execution_environment:
                raise ValueError(
                    "snapshot execution environment does not match runtime bindings"
                )
            baseline_environment.require_match(
                self.execution_environment, context="RunSpec baseline"
            )
        else:
            if runtime_environment is not None:
                raise ValueError(
                    "resolved runtime execution environment cannot be downgraded "
                    "to LEGACY_UNKNOWN"
                )
            if self.spec.namespace != LEGACY_RESEARCH_NAMESPACE:
                raise ValueError(
                    "only the legacy namespace may use LEGACY_UNKNOWN execution "
                    "evidence"
                )
            if baseline_environment != self.execution_environment:
                raise ValueError(
                    "legacy snapshot and baseline environment scopes differ"
                )
        object.__setattr__(self, "definitions", _frozen(plain_definitions))
        object.__setattr__(self, "runtime_bindings", _frozen(plain_runtime))

    @property
    def snapshot_material(self) -> dict[str, Any]:
        return {
            "schema_version": RUN_SPEC_SCHEMA_VERSION,
            "run_spec": self.spec.to_dict(),
            "namespace": self.spec.namespace.to_dict(),
            "resolved_profiles": _plain(self.definitions),
            "runtime_bindings": _plain(self.runtime_bindings),
            "execution_environment": self.execution_environment.to_dict(),
            "scientifically_comparable": self.execution_environment.is_resolved,
        }

    @property
    def snapshot_digest(self) -> str:
        return canonical_sha256(self.snapshot_material)

    def to_dict(self) -> dict[str, Any]:
        return {**self.snapshot_material, "snapshot_digest": self.snapshot_digest}


def resolve_run_spec(
    spec: RunSpecV2,
    *,
    registry: ProfileRegistry = BUILTIN_PROFILE_REGISTRY,
    runtime_bindings: Mapping[str, Any] | None = None,
) -> ResolvedRunSnapshot:
    if not isinstance(spec, RunSpecV2):
        raise TypeError("spec must be RunSpecV2")
    runtime = dict(runtime_bindings or {})
    _scan_for_secret_values(runtime)
    definitions: dict[str, Mapping[str, Any]] = {}
    references = {
        "deployment": spec.deployment,
        "operator": spec.operator,
        "language": spec.language,
        "evaluator": spec.evaluator,
        "evaluation_protocol": spec.evaluation_protocol,
        "promotion_policy": spec.promotion_policy,
        "proposer": spec.proposer,
    }
    if spec.campaign is not None:
        references["campaign"] = spec.campaign
    for name, reference in references.items():
        definition: ProfileDefinition = registry.resolve(reference)
        definitions[name] = definition.to_dict()
    runtime_environment = _execution_environment_from_runtime(runtime)
    execution_environment = (
        spec.baseline.execution_environment
        if runtime_environment is None
        else runtime_environment
    )
    return ResolvedRunSnapshot(
        spec=spec,
        definitions=definitions,
        runtime_bindings=runtime,
        execution_environment=execution_environment,
    )


__all__ = [
    "MAX_RUN_CANDIDATES",
    "MAX_RUN_CONSECUTIVE_FAILURES",
    "MAX_RUN_WALL_SECONDS",
    "EXECUTION_ENVIRONMENT_BINDING_FIELDS",
    "EXECUTION_ENVIRONMENT_BINDING_KEY",
    "RUN_SPEC_SCHEMA_VERSION",
    "ResolvedRunSnapshot",
    "RunBudget",
    "RunSpecV2",
    "resolve_run_spec",
]
