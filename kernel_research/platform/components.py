"""Trusted component contracts for the composable V2 research platform.

Profiles are immutable data identities.  Components are reviewed Python
implementations selected by those profiles.  Configuration never supplies an
import path or constructs an arbitrary workflow.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .profiles import (
    ProfileDefinition,
    ProfileRef,
    ProfileRegistry,
    ResearchNamespace,
)
from .proposal import BundleLimits


class CaseRole(str, Enum):
    SCORED = "scored"
    CORRECTNESS_ONLY = "correctness_only"
    HOLDOUT = "holdout"


class ReplicateKind(str, Enum):
    PRIMARY = "primary"
    CONFIRMATION = "confirmation"
    NOISE = "noise"
    RETEST = "retest"
    VALIDATION = "validation"


@dataclass(frozen=True, slots=True)
class StageDefinition:
    """One stage from a trusted, versioned evaluation workflow."""

    stage_id: str
    suite_id: str | None
    replicate_kind: ReplicateKind
    requires_baseline: bool = False
    may_request_confirmation: bool = False

    def __post_init__(self) -> None:
        if not self.stage_id or any(character.isspace() for character in self.stage_id):
            raise ValueError("stage_id must be a non-empty token")
        if self.suite_id is not None and (
            not self.suite_id or any(character.isspace() for character in self.suite_id)
        ):
            raise ValueError("suite_id must be null or a non-empty token")
        if self.may_request_confirmation and not self.requires_baseline:
            raise ValueError("a promotable stage must require a frozen baseline")


@dataclass(frozen=True, slots=True)
class EvaluationProtocolDefinition:
    """Immutable scientific protocol selected from trusted code."""

    protocol_id: str
    revision: str
    implementation_id: str
    stages: tuple[StageDefinition, ...]
    suite_cases: Mapping[str, tuple[str, ...]]
    case_roles: Mapping[str, CaseRole]
    warmup_iterations: int
    measurement_rounds: int
    samples_per_round: int
    interleaved_baseline: bool

    def __post_init__(self) -> None:
        if not self.protocol_id or not self.revision or not self.implementation_id:
            raise ValueError("protocol identity must not be empty")
        if not self.stages:
            raise ValueError("evaluation protocol must contain at least one stage")
        stage_ids = tuple(stage.stage_id for stage in self.stages)
        if len(set(stage_ids)) != len(stage_ids):
            raise ValueError("evaluation stage IDs must be unique")
        for value, name in (
            (self.warmup_iterations, "warmup_iterations"),
            (self.measurement_rounds, "measurement_rounds"),
            (self.samples_per_round, "samples_per_round"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        suites = {stage.suite_id for stage in self.stages if stage.suite_id is not None}
        missing_suites = suites - set(self.suite_cases)
        if missing_suites:
            raise ValueError(
                "workflow references undefined suites: "
                + ", ".join(sorted(missing_suites))
            )
        known_cases = {
            case_id for cases in self.suite_cases.values() for case_id in cases
        }
        missing_roles = known_cases - set(self.case_roles)
        if missing_roles:
            raise ValueError(
                "evaluation cases are missing roles: "
                + ", ".join(sorted(missing_roles))
            )
        object.__setattr__(
            self,
            "suite_cases",
            MappingProxyType(
                {name: tuple(cases) for name, cases in self.suite_cases.items()}
            ),
        )
        object.__setattr__(
            self,
            "case_roles",
            MappingProxyType(dict(self.case_roles)),
        )

    @property
    def samples_per_case(self) -> int:
        return self.measurement_rounds * self.samples_per_round

    def expected_cases(self, suite_id: str) -> tuple[str, ...]:
        try:
            return self.suite_cases[suite_id]
        except KeyError as exc:
            raise ValueError(f"unknown protocol suite: {suite_id}") from exc

    def scored_cases(self, suite_id: str) -> tuple[str, ...]:
        return tuple(
            case_id
            for case_id in self.expected_cases(suite_id)
            if self.case_roles[case_id] is CaseRole.SCORED
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "revision": self.revision,
            "implementation_id": self.implementation_id,
            "stages": [
                {
                    "stage_id": stage.stage_id,
                    "suite_id": stage.suite_id,
                    "replicate_kind": stage.replicate_kind.value,
                    "requires_baseline": stage.requires_baseline,
                    "may_request_confirmation": stage.may_request_confirmation,
                }
                for stage in self.stages
            ],
            "suite_cases": {
                name: list(cases) for name, cases in self.suite_cases.items()
            },
            "case_roles": {
                name: role.value for name, role in self.case_roles.items()
            },
            "warmup_iterations": self.warmup_iterations,
            "measurement_rounds": self.measurement_rounds,
            "samples_per_round": self.samples_per_round,
            "interleaved_baseline": self.interleaved_baseline,
        }


@runtime_checkable
class OperatorPack(Protocol):
    operator_id: str
    implementation_id: str
    revision: str

    def contract_summary(self) -> Mapping[str, Any]: ...

    def prompt_context(self) -> str: ...

    def validate_operator_candidate(self, source: str) -> Any: ...


@runtime_checkable
class LanguageAdapter(Protocol):
    language_id: str
    implementation_id: str
    revision: str
    entrypoint: str
    bundle_limits: BundleLimits

    def allowed_paths(self) -> Sequence[str]: ...

    def validate_language_candidate(self, source: str) -> Any: ...

    def compile_cache_material(self) -> Mapping[str, Any]: ...


@runtime_checkable
class DeviceBackend(Protocol):
    device_id: str
    implementation_id: str
    revision: str

    def evaluator_backend(self) -> str: ...

    def resource_requirements(self) -> Mapping[str, Any]: ...

    def fatal_error_markers(self) -> Sequence[str]: ...


@runtime_checkable
class PromotionPolicy(Protocol):
    policy_id: str
    implementation_id: str
    revision: str

    def thresholds(self) -> Mapping[str, float]: ...


@dataclass(frozen=True, slots=True)
class TargetComponents:
    operator: OperatorPack
    language: LanguageAdapter
    device: DeviceBackend
    protocol: EvaluationProtocolDefinition
    promotion: PromotionPolicy

    def validate_candidate(self, source: str) -> Any:
        """Apply both trusted language and operator policy axes.

        The current Triton adapter deliberately returns the historical
        ``ResearchPolicyResult`` so controller-visible behaviour remains
        byte-for-byte compatible.  The operator validation is still executed
        and its source identity is cross-checked; a future adapter therefore
        cannot silently bypass the operator ABI/semantic gate.
        """

        language_result = self.language.validate_language_candidate(source)
        operator_result = self.operator.validate_operator_candidate(source)
        for result, axis in (
            (language_result, "language"),
            (operator_result, "operator"),
        ):
            if getattr(result, "source", None) != source:
                raise RuntimeError(
                    f"trusted {axis} candidate policy changed source bytes"
                )
            if not isinstance(getattr(result, "sha256", None), str):
                raise RuntimeError(
                    f"trusted {axis} candidate policy omitted source identity"
                )
        if language_result.sha256 != operator_result.sha256:
            raise RuntimeError(
                "trusted language and operator candidate policies disagree"
            )
        language_errors = tuple(getattr(language_result, "errors", ()))
        operator_errors = tuple(getattr(operator_result, "errors", ()))
        additional_operator_errors = tuple(
            error for error in operator_errors if error not in language_errors
        )
        if additional_operator_errors:
            return _CombinedCandidatePolicyResult(
                base=language_result,
                errors=language_errors + additional_operator_errors,
            )
        return language_result


@dataclass(frozen=True, slots=True)
class _CombinedCandidatePolicyResult:
    """Adapter-neutral merged decision used only when axes add errors."""

    base: Any
    errors: tuple[Any, ...]

    @property
    def source(self) -> str:
        return str(self.base.source)

    @property
    def sha256(self) -> str:
        return str(self.base.sha256)

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def is_valid(self) -> bool:
        return self.valid

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.base.to_dict())
        payload["valid"] = self.valid
        payload["sha256"] = self.sha256
        payload["errors"] = [
            error.to_dict() if callable(getattr(error, "to_dict", None)) else error
            for error in self.errors
        ]
        return payload


@dataclass(frozen=True, slots=True)
class ProfileComponentBinding:
    """One exact reviewed profile revision bound to trusted Python code."""

    profile: ProfileRef
    implementation_id: str
    component: object


_PROFILE_COMPONENT_IDENTITIES = MappingProxyType(
    {
        "operator": "operator_id",
        "language": "language_id",
        "evaluator": "device_id",
        "evaluation_protocol": "protocol_id",
        "promotion_policy": "policy_id",
    }
)


class TrustedComponentRegistry:
    """Closed registry of reviewed component implementations.

    Registration happens in trusted module initialization.  Runtime config may
    select an exact key, but cannot register or import a new implementation.
    """

    def __init__(
        self, *, profile_registry: ProfileRegistry | None = None
    ) -> None:
        self._values: dict[tuple[str, str, str], object] = {}
        self._profile_values: dict[ProfileRef, ProfileComponentBinding] = {}
        self._profile_registry = profile_registry
        self._sealed = False

    def register(
        self,
        *,
        kind: str,
        component_id: str,
        revision: str,
        component: object,
    ) -> None:
        if self._sealed:
            raise RuntimeError("trusted component registry is sealed")
        for value, name in (
            (kind, "kind"),
            (component_id, "component_id"),
            (revision, "revision"),
        ):
            if not value or any(character.isspace() for character in value):
                raise ValueError(f"{name} must be a non-empty token")
        key = (kind, component_id, revision)
        if key in self._values:
            raise ValueError(f"component already registered: {key!r}")
        self._values[key] = component

    def bind_profile(
        self,
        definition: ProfileDefinition,
        *,
        component: object,
    ) -> None:
        """Bind reviewed code to an exact profile digest before sealing.

        The component's declared identity must equal the profile's
        ``implementation_id`` and revision.  Runtime configuration can then
        resolve only an exact built-in ``ProfileRef``; it can never supply an
        import path or substitute an implementation with a matching alias.
        """

        if self._sealed:
            raise RuntimeError("trusted component registry is sealed")
        if not isinstance(definition, ProfileDefinition):
            raise TypeError("definition must be a ProfileDefinition")
        if self._profile_registry is None:
            raise RuntimeError("profile binding requires a ProfileRegistry")
        definition = self._profile_registry.resolve(definition.ref)
        try:
            identity_field = _PROFILE_COMPONENT_IDENTITIES[
                definition.ref.kind
            ]
        except KeyError as exc:
            raise ValueError(
                f"profile kind {definition.ref.kind!r} has no component binding"
            ) from exc
        component_id = getattr(component, identity_field, None)
        implementation_id = getattr(component, "implementation_id", None)
        revision = getattr(component, "revision", None)
        if component_id != definition.ref.id:
            raise ValueError(
                f"component {identity_field} does not match profile id"
            )
        if implementation_id != definition.implementation_id:
            raise ValueError(
                "component implementation_id does not match profile definition"
            )
        if revision != definition.ref.revision:
            raise ValueError("component revision does not match profile revision")
        if definition.ref in self._profile_values:
            raise ValueError("profile revision already has a component binding")
        self.register(
            kind=definition.ref.kind,
            component_id=definition.implementation_id,
            revision=definition.ref.revision,
            component=component,
        )
        self._profile_values[definition.ref] = ProfileComponentBinding(
            profile=definition.ref,
            implementation_id=definition.implementation_id,
            component=component,
        )

    def seal(self) -> None:
        self._sealed = True

    def resolve(self, *, kind: str, component_id: str, revision: str) -> object:
        try:
            return self._values[(kind, component_id, revision)]
        except KeyError as exc:
            raise ValueError(
                f"unknown trusted {kind} component "
                f"{component_id!r}@{revision!r}"
            ) from exc

    def resolve_profile(
        self, ref: ProfileRef | Mapping[str, Any]
    ) -> object:
        """Resolve one exact built-in profile reference to reviewed code."""

        if self._profile_registry is None:
            raise RuntimeError("registry has no ProfileRegistry binding")
        if not isinstance(ref, ProfileRef):
            ref = ProfileRef.from_value(ref)
        # Resolve first so a known alias with a tampered digest fails as a
        # digest mismatch, rather than being reported as merely absent.
        definition = self._profile_registry.resolve(ref)
        try:
            return self._profile_values[definition.ref].component
        except KeyError as exc:
            raise ValueError(
                f"built-in profile {ref.kind}/{ref.id}@{ref.revision} "
                "has no trusted component binding"
            ) from exc

    def resolve_target(self, namespace: ResearchNamespace) -> TargetComponents:
        """Resolve all five target axes from one frozen research namespace."""

        if not isinstance(namespace, ResearchNamespace):
            raise TypeError("namespace must be a ResearchNamespace")
        for reference in (
            namespace.operator,
            namespace.language,
            namespace.evaluator,
            namespace.evaluation_protocol,
            namespace.promotion_policy,
        ):
            if self._profile_registry is None:
                raise RuntimeError("registry has no ProfileRegistry binding")
            definition = self._profile_registry.resolve(reference)
            activation_state = definition.config.get(
                "activation_state", "ACTIVE"
            )
            if activation_state != "ACTIVE":
                raise ValueError(
                    f"profile {reference.kind}/{reference.id}@"
                    f"{reference.revision} is not ACTIVE"
                )
        operator = self.resolve_profile(namespace.operator)
        language = self.resolve_profile(namespace.language)
        device = self.resolve_profile(namespace.evaluator)
        protocol = self.resolve_profile(namespace.evaluation_protocol)
        promotion = self.resolve_profile(namespace.promotion_policy)
        if not isinstance(operator, OperatorPack):
            raise RuntimeError("operator profile resolved to an invalid component")
        if not isinstance(language, LanguageAdapter):
            raise RuntimeError("language profile resolved to an invalid component")
        if not isinstance(device, DeviceBackend):
            raise RuntimeError("evaluator profile resolved to an invalid component")
        if not isinstance(protocol, EvaluationProtocolDefinition):
            raise RuntimeError("protocol profile resolved to an invalid component")
        if not isinstance(promotion, PromotionPolicy):
            raise RuntimeError("promotion profile resolved to an invalid component")
        return TargetComponents(
            operator=operator,
            language=language,
            device=device,
            protocol=protocol,
            promotion=promotion,
        )

    def list_keys(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(sorted(self._values))

    def bindings(self) -> tuple[ProfileComponentBinding, ...]:
        return tuple(
            self._profile_values[ref]
            for ref in sorted(
                self._profile_values,
                key=lambda item: (item.kind, item.id, item.revision),
            )
        )

    @property
    def sealed(self) -> bool:
        return self._sealed


__all__ = [
    "CaseRole",
    "DeviceBackend",
    "EvaluationProtocolDefinition",
    "LanguageAdapter",
    "OperatorPack",
    "PromotionPolicy",
    "ProfileComponentBinding",
    "ReplicateKind",
    "StageDefinition",
    "TargetComponents",
    "TrustedComponentRegistry",
]
