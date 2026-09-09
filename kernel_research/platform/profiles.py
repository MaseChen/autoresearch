"""Immutable profile references and the trusted built-in registry."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
import re
from types import MappingProxyType
from typing import Any

from ..constants import (
    CURRENT_C500_CASE_IDS,
    CURRENT_C500_EVALUATION_PROTOCOL_ID,
    XPUOJ_C500_CASE_IDS,
    XPUOJ_C500_EVALUATION_PROTOCOL_ID,
    CURRENT_C500_HOLDOUT_CASE_IDS,
    LEGACY_C500_CASE_IDS,
    LEGACY_C500_EVALUATION_PROTOCOL_ID,
)
from .canonical import (
    canonical_json_text,
    canonical_sha256,
    require_sha256_digest,
)


PROFILE_KINDS = frozenset(
    {
        "operator",
        "language",
        "evaluator",
        "evaluation_protocol",
        "promotion_policy",
        "proposer",
        "campaign",
        "deployment",
    }
)
_STABLE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")


def _strict_object(
    value: object, *, fields: frozenset[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    unknown = sorted(set(value) - fields)
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {', '.join(unknown)}")
    missing = sorted(fields - set(value))
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")
    return value


def _validate_kind(value: object) -> str:
    if not isinstance(value, str) or value not in PROFILE_KINDS:
        allowed = ", ".join(sorted(PROFILE_KINDS))
        raise ValueError(f"profile kind must be one of: {allowed}")
    return value


def _stable_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _STABLE_ID_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase stable identifier")
    return value


def _revision(value: object) -> str:
    if not isinstance(value, str) or not _REVISION_RE.fullmatch(value):
        raise ValueError("profile revision must be a stable non-empty revision")
    return value


def _frozen_json(value: Any) -> Any:
    """Copy a validated JSON value into immutable containers."""

    normalized = json.loads(canonical_json_text(value))

    def freeze(child: Any) -> Any:
        if isinstance(child, dict):
            return MappingProxyType(
                {key: freeze(item) for key, item in child.items()}
            )
        if isinstance(child, list):
            return tuple(freeze(item) for item in child)
        return child

    return freeze(normalized)


def _plain_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(child) for child in value]
    return value


@dataclass(frozen=True, order=True)
class ProfileRef:
    """An exact immutable profile revision selected by content digest."""

    kind: str
    id: str
    revision: str
    digest: str

    def __post_init__(self) -> None:
        _validate_kind(self.kind)
        _stable_id(self.id, field="profile id")
        _revision(self.revision)
        require_sha256_digest(self.digest, field="profile digest")

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        profile_id: str,
        revision: str,
        profile: Mapping[str, Any],
    ) -> "ProfileRef":
        kind = _validate_kind(kind)
        profile_id = _stable_id(profile_id, field="profile id")
        revision = _revision(revision)
        digest = canonical_sha256(
            {
                "kind": kind,
                "id": profile_id,
                "revision": revision,
                "profile": profile,
            }
        )
        return cls(kind, profile_id, revision, digest)

    @classmethod
    def from_value(cls, value: object) -> "ProfileRef":
        obj = _strict_object(
            value,
            fields=frozenset({"kind", "id", "revision", "digest"}),
            name="profile reference",
        )
        return cls(
            kind=obj["kind"],
            id=obj["id"],
            revision=obj["revision"],
            digest=obj["digest"],
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "id": self.id,
            "revision": self.revision,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class ProfileDefinition:
    """One reviewed built-in implementation and its bounded JSON options."""

    ref: ProfileRef
    implementation_id: str
    config: Mapping[str, Any]

    def __post_init__(self) -> None:
        _stable_id(self.implementation_id, field="implementation_id")
        frozen = _frozen_json(self.config)
        if not isinstance(frozen, Mapping):
            raise ValueError("profile config must be a JSON object")
        object.__setattr__(self, "config", frozen)
        expected = ProfileRef.create(
            kind=self.ref.kind,
            profile_id=self.ref.id,
            revision=self.ref.revision,
            profile={
                "schema_version": 1,
                "implementation_id": self.implementation_id,
                "config": _plain_json(frozen),
            },
        )
        if expected != self.ref:
            raise ValueError("profile reference digest does not match definition")

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        profile_id: str,
        revision: str,
        implementation_id: str,
        config: Mapping[str, Any],
    ) -> "ProfileDefinition":
        frozen = _frozen_json(config)
        if not isinstance(frozen, Mapping):
            raise ValueError("profile config must be a JSON object")
        ref = ProfileRef.create(
            kind=kind,
            profile_id=profile_id,
            revision=revision,
            profile={
                "schema_version": 1,
                "implementation_id": implementation_id,
                "config": _plain_json(frozen),
            },
        )
        return cls(ref, implementation_id, frozen)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.ref.to_dict(),
            "implementation_id": self.implementation_id,
            "config": _plain_json(self.config),
        }


class ProfileRegistry:
    """Immutable exact-match registry; it never imports configured code."""

    def __init__(self, definitions: Iterable[ProfileDefinition]) -> None:
        by_key: dict[tuple[str, str, str], ProfileDefinition] = {}
        by_digest: dict[str, ProfileDefinition] = {}
        for definition in definitions:
            if not isinstance(definition, ProfileDefinition):
                raise TypeError("profile registry accepts ProfileDefinition values")
            key = (
                definition.ref.kind,
                definition.ref.id,
                definition.ref.revision,
            )
            if key in by_key:
                raise ValueError("profile registry contains a duplicate revision")
            if definition.ref.digest in by_digest:
                raise ValueError("profile registry contains a duplicate digest")
            by_key[key] = definition
            by_digest[definition.ref.digest] = definition
        self._by_key = MappingProxyType(by_key)
        self._by_digest = MappingProxyType(by_digest)

    def resolve(self, ref: ProfileRef | Mapping[str, Any]) -> ProfileDefinition:
        if not isinstance(ref, ProfileRef):
            ref = ProfileRef.from_value(ref)
        key = (ref.kind, ref.id, ref.revision)
        try:
            definition = self._by_key[key]
        except KeyError as exc:
            raise KeyError(
                f"unknown built-in profile {ref.kind}/{ref.id}@{ref.revision}"
            ) from exc
        if definition.ref.digest != ref.digest:
            raise ValueError("profile digest does not match the built-in revision")
        return definition

    def get(self, *, kind: str, profile_id: str, revision: str) -> ProfileDefinition:
        key = (
            _validate_kind(kind),
            _stable_id(profile_id, field="profile id"),
            _revision(revision),
        )
        try:
            return self._by_key[key]
        except KeyError as exc:
            raise KeyError(
                f"unknown built-in profile {kind}/{profile_id}@{revision}"
            ) from exc

    def references(self) -> tuple[ProfileRef, ...]:
        return tuple(
            definition.ref
            for _, definition in sorted(self._by_key.items())
        )

    def definitions(self) -> tuple[ProfileDefinition, ...]:
        return tuple(
            definition for _, definition in sorted(self._by_key.items())
        )


@dataclass(frozen=True)
class ResearchNamespace:
    """The five scientific profiles that isolate evidence and baselines."""

    operator: ProfileRef
    language: ProfileRef
    evaluator: ProfileRef
    evaluation_protocol: ProfileRef
    promotion_policy: ProfileRef

    def __post_init__(self) -> None:
        for field, expected_kind in (
            ("operator", "operator"),
            ("language", "language"),
            ("evaluator", "evaluator"),
            ("evaluation_protocol", "evaluation_protocol"),
            ("promotion_policy", "promotion_policy"),
        ):
            value = getattr(self, field)
            if not isinstance(value, ProfileRef) or value.kind != expected_kind:
                raise ValueError(f"{field} must be a {expected_kind} ProfileRef")

    @classmethod
    def from_profiles(
        cls,
        *,
        operator: ProfileRef,
        language: ProfileRef,
        evaluator: ProfileRef,
        evaluation_protocol: ProfileRef,
        promotion_policy: ProfileRef,
    ) -> "ResearchNamespace":
        return cls(
            operator=operator,
            language=language,
            evaluator=evaluator,
            evaluation_protocol=evaluation_protocol,
            promotion_policy=promotion_policy,
        )

    @property
    def namespace_id(self) -> str:
        return canonical_sha256(self._identity_dict())

    def _identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "operator": self.operator.to_dict(),
            "language": self.language.to_dict(),
            "evaluator": self.evaluator.to_dict(),
            "evaluation_protocol": self.evaluation_protocol.to_dict(),
            "promotion_policy": self.promotion_policy.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity_dict(), "namespace_id": self.namespace_id}

    @classmethod
    def from_value(cls, value: object) -> "ResearchNamespace":
        fields = frozenset(
            {
                "schema_version",
                "namespace_id",
                "operator",
                "language",
                "evaluator",
                "evaluation_protocol",
                "promotion_policy",
            }
        )
        obj = _strict_object(value, fields=fields, name="research namespace")
        if type(obj["schema_version"]) is not int or obj["schema_version"] != 1:
            raise ValueError("research namespace schema_version must be 1")
        namespace = cls(
            operator=ProfileRef.from_value(obj["operator"]),
            language=ProfileRef.from_value(obj["language"]),
            evaluator=ProfileRef.from_value(obj["evaluator"]),
            evaluation_protocol=ProfileRef.from_value(
                obj["evaluation_protocol"]
            ),
            promotion_policy=ProfileRef.from_value(obj["promotion_policy"]),
        )
        if obj["namespace_id"] != namespace.namespace_id:
            raise ValueError("namespace_id does not match research profiles")
        return namespace


def _definition(
    kind: str,
    profile_id: str,
    revision: str,
    implementation_id: str,
    config: Mapping[str, Any],
) -> ProfileDefinition:
    return ProfileDefinition.create(
        kind=kind,
        profile_id=profile_id,
        revision=revision,
        implementation_id=implementation_id,
        config=config,
    )


def _protocol_config(
    *, case_ids: Mapping[str, tuple[str, ...]], suite_revision: str
) -> dict[str, Any]:
    correctness_only = tuple(
        case_id
        for case_id in case_ids["quick"]
        if case_id not in LEGACY_C500_CASE_IDS["quick"]
    )
    holdout = frozenset(CURRENT_C500_HOLDOUT_CASE_IDS).intersection(
        correctness_only
    )
    stages = (
        ("PROPOSE", None, "validation", False, False),
        ("POLICY", None, "validation", False, False),
        ("SMOKE", "smoke", "validation", False, False),
        ("QUICK", "quick", "validation", False, False),
        ("FULL_PRIMARY", "full", "primary", True, True),
        ("CONFIRMATION", "full", "confirmation", True, False),
    )
    return {
        "stages": [
            {
                "stage_id": stage_id,
                "suite_id": suite_id,
                "replicate_kind": replicate_kind,
                "requires_baseline": requires_baseline,
                "may_request_confirmation": may_request_confirmation,
            }
            for (
                stage_id,
                suite_id,
                replicate_kind,
                requires_baseline,
                may_request_confirmation,
            ) in stages
        ],
        "suite_revision": suite_revision,
        "suite_case_ids": {
            suite: list(cases) for suite, cases in case_ids.items()
        },
        "case_roles": {
            case_id: (
                "holdout"
                if case_id in holdout
                else (
                    "correctness_only"
                    if case_id in correctness_only
                    else "scored"
                )
            )
            for cases in case_ids.values()
            for case_id in cases
        },
        "warmup_iterations": 10,
        "measurement_rounds": 3,
        "samples_per_round": 10,
        "interleaved_baseline": True,
    }


def _xpuoj_protocol_config() -> dict[str, Any]:
    """Frozen four-case protocol for external XPU-OJ baseline comparisons."""

    stages = (
        ("PROPOSE", None, "validation", False, False),
        ("POLICY", None, "validation", False, False),
        ("FULL_PRIMARY", "full", "primary", True, True),
        ("CONFIRMATION", "full", "confirmation", True, False),
    )
    return {
        "stages": [
            {
                "stage_id": stage_id,
                "suite_id": suite_id,
                "replicate_kind": replicate_kind,
                "requires_baseline": requires_baseline,
                "may_request_confirmation": may_request_confirmation,
            }
            for (
                stage_id,
                suite_id,
                replicate_kind,
                requires_baseline,
                may_request_confirmation,
            ) in stages
        ],
        "suite_revision": "xpuoj-four-case-v1",
        "suite_case_ids": {"full": list(XPUOJ_C500_CASE_IDS["full"])},
        "case_roles": {
            case_id: "scored" for case_id in XPUOJ_C500_CASE_IDS["full"]
        },
        "warmup_iterations": 10,
        "measurement_rounds": 3,
        "samples_per_round": 10,
        "interleaved_baseline": True,
        "benchmark_only": True,
        "external_baseline_authority": "xpuoj-accepted-submission-v1",
    }


def _proposer_config(*, harness: str, model: str) -> dict[str, str]:
    return {
        "harness": harness,
        "harness_revision": "v1",
        "model": model,
        "model_revision": "v1",
        "prompt_protocol": "proposal-v1",
        "prompt_protocol_revision": "v1",
    }


_BUILTIN_DEFINITIONS = (
    _definition(
        "operator",
        "fused-moe-w8a8-tn",
        "v1",
        "fused-moe-w8a8-tn",
        {"abi": "run-kernel-v1", "case_catalog": "fused-moe-v1"},
    ),
    _definition(
        "language",
        "triton-python",
        "v1",
        "triton-python",
        {
            "entrypoint": "kernel.py",
            "allowed_extensions": [".py"],
            "allowed_media_types": ["text/x-python"],
            "max_files": 1,
            "max_file_bytes": 262144,
            "max_total_bytes": 262144,
            "toolchain_fingerprint": "legacy-unknown",
        },
    ),
    _definition(
        "language",
        "tilelang-python",
        "v0-probe",
        "tilelang-python",
        {
            "activation_state": "INACTIVE",
            "agent_enabled": False,
            "promotion_eligible": False,
            "entrypoint": "kernel.py",
            "allowed_extensions": [".py"],
            "allowed_media_types": ["text/x-python"],
            "max_files": 1,
            "max_file_bytes": 262144,
            "max_total_bytes": 262144,
            "abi": "run-kernel-v1",
            "probe_recipe_revision": "tilelang-c500-probe-v1",
            "toolchain_fingerprint": "UNAVAILABLE",
        },
    ),
    _definition(
        "language",
        "maca-cuda",
        "v0-contract",
        "maca-cuda",
        {
            "activation_state": "INACTIVE",
            "agent_enabled": False,
            "promotion_eligible": False,
            "entrypoint": "kernel.cu",
            "allowed_extensions": [".cu"],
            "allowed_media_types": ["text/x-cuda"],
            "max_files": 1,
            "max_file_bytes": 524288,
            "max_total_bytes": 524288,
            "abi": "autoresearch-maca-cuda-abi-v1",
            "compile_recipe": "maca-cuda-compile-v1",
            "execute_recipe": "maca-cuda-execute-v1",
            "compile_execute_separation": True,
            "compile_environment": "maca-cuda-compile-sandbox-v1",
            "execute_environment": "maca-cuda-execute-sandbox-v1",
            "network_access": False,
            "toolchain_fingerprint": "UNAVAILABLE",
        },
    ),
    _definition(
        "operator",
        "ragged-prefill",
        "v0-inactive",
        "ragged-prefill",
        {
            "activation_state": "INACTIVE",
            "implementation_status": "oracle-ready",
            "agent_enabled": False,
            "promotion_eligible": False,
            "required_manual_full_evidence": 2,
            "abi": "ragged-prefill-v1",
            "oracle": "ragged-prefill-numpy-v1",
        },
    ),
    _definition(
        "operator",
        "paged-decode",
        "v0-inactive",
        "paged-decode",
        {
            "activation_state": "INACTIVE",
            "implementation_status": "descriptor-only",
            "agent_enabled": False,
            "promotion_eligible": False,
            "required_manual_full_evidence": 2,
            "abi": "unpublished",
            "oracle": "unpublished",
        },
    ),
    _definition(
        "operator",
        "paged-prefill",
        "v0-inactive",
        "paged-prefill",
        {
            "activation_state": "INACTIVE",
            "implementation_status": "descriptor-only",
            "agent_enabled": False,
            "promotion_eligible": False,
            "required_manual_full_evidence": 2,
            "abi": "unpublished",
            "oracle": "unpublished",
        },
    ),
    _definition(
        "operator",
        "kv-cache-decode",
        "v0-inactive",
        "kv-cache-decode",
        {
            "activation_state": "INACTIVE",
            "implementation_status": "descriptor-only",
            "agent_enabled": False,
            "promotion_eligible": False,
            "required_manual_full_evidence": 2,
            "abi": "unpublished",
            "oracle": "unpublished",
        },
    ),
    _definition(
        "operator",
        "mla",
        "v0-inactive",
        "mla",
        {
            "activation_state": "INACTIVE",
            "implementation_status": "descriptor-only",
            "agent_enabled": False,
            "promotion_eligible": False,
            "required_manual_full_evidence": 2,
            "abi": "unpublished",
            "oracle": "unpublished",
        },
    ),
    _definition(
        "evaluator",
        "metax-c500",
        "legacy-v1",
        "metax-c500",
        {
            "backend": "c500",
            "device": "MetaX C500",
            "image_fingerprint": "legacy-deployment-binding",
        },
    ),
    _definition(
        "evaluation_protocol",
        LEGACY_C500_EVALUATION_PROTOCOL_ID,
        "v1",
        "c500-evaluation-protocol",
        _protocol_config(
            case_ids=LEGACY_C500_CASE_IDS,
            suite_revision="legacy-v1",
        ),
    ),
    _definition(
        "evaluation_protocol",
        CURRENT_C500_EVALUATION_PROTOCOL_ID,
        "v2-shadow-holdout",
        "c500-evaluation-protocol",
        _protocol_config(
            case_ids=CURRENT_C500_CASE_IDS,
            suite_revision="shadow-holdout-v2",
        ),
    ),
    _definition(
        "evaluation_protocol",
        XPUOJ_C500_EVALUATION_PROTOCOL_ID,
        "v1",
        "c500-xpuoj-evaluation-protocol",
        _xpuoj_protocol_config(),
    ),
    _definition(
        "promotion_policy",
        "current-c500",
        "v1",
        "current-c500",
        {
            "confirmation_required": True,
            "min_aggregate_speedup": 1.01,
            "max_case_regression": 0.03,
            "min_matched_ratio": 0.99,
        },
    ),
    _definition(
        "proposer",
        "opencode-deepseek-v4-pro",
        "v1",
        "opencode-deepseek-v4-pro",
        _proposer_config(
            harness="opencode", model="deepseek/deepseek-v4-pro"
        ),
    ),
    _definition(
        "proposer",
        "opencode-deepseek-v4-flash",
        "v1",
        "opencode-deepseek-v4-flash",
        _proposer_config(
            harness="opencode", model="deepseek/deepseek-v4-flash"
        ),
    ),
    _definition(
        "proposer",
        "direct-api-deepseek-v4-pro",
        "v1",
        "direct-api-deepseek-v4-pro",
        _proposer_config(
            harness="direct-api", model="deepseek/deepseek-v4-pro"
        ),
    ),
    _definition(
        "proposer",
        "direct-api-deepseek-v4-flash",
        "v1",
        "direct-api-deepseek-v4-flash",
        _proposer_config(
            harness="direct-api", model="deepseek/deepseek-v4-flash"
        ),
    ),
    _definition(
        "proposer",
        "pi-deepseek-v4-pro",
        "v1",
        "pi-deepseek-v4-pro",
        _proposer_config(
            harness="pi", model="deepseek/deepseek-v4-pro"
        ),
    ),
    _definition(
        "proposer",
        "pi-deepseek-v4-flash",
        "v1",
        "pi-deepseek-v4-flash",
        _proposer_config(
            harness="pi", model="deepseek/deepseek-v4-flash"
        ),
    ),
    _definition(
        "campaign",
        "bounded-discovery",
        "v1",
        "bounded-discovery",
        {"lineage_mode": "pause-on-promotion"},
    ),
    _definition(
        "deployment",
        "legacy-local-c500",
        "v1",
        "legacy-local-c500",
        {"resource_pool": ["gpu1"], "remote_scheduling": False},
    ),
)

BUILTIN_PROFILE_REGISTRY = ProfileRegistry(_BUILTIN_DEFINITIONS)


def builtin_profile_registry() -> ProfileRegistry:
    """Return the immutable reviewed registry used by the trusted host."""

    return BUILTIN_PROFILE_REGISTRY


def _builtin_ref(kind: str, profile_id: str, revision: str) -> ProfileRef:
    return BUILTIN_PROFILE_REGISTRY.get(
        kind=kind, profile_id=profile_id, revision=revision
    ).ref


LEGACY_RESEARCH_NAMESPACE = ResearchNamespace.from_profiles(
    operator=_builtin_ref("operator", "fused-moe-w8a8-tn", "v1"),
    language=_builtin_ref("language", "triton-python", "v1"),
    evaluator=_builtin_ref("evaluator", "metax-c500", "legacy-v1"),
    evaluation_protocol=_builtin_ref(
        "evaluation_protocol", LEGACY_C500_EVALUATION_PROTOCOL_ID, "v1"
    ),
    promotion_policy=_builtin_ref("promotion_policy", "current-c500", "v1"),
)

CURRENT_RESEARCH_NAMESPACE = ResearchNamespace.from_profiles(
    operator=_builtin_ref("operator", "fused-moe-w8a8-tn", "v1"),
    language=_builtin_ref("language", "triton-python", "v1"),
    evaluator=_builtin_ref("evaluator", "metax-c500", "legacy-v1"),
    evaluation_protocol=_builtin_ref(
        "evaluation_protocol",
        CURRENT_C500_EVALUATION_PROTOCOL_ID,
        "v2-shadow-holdout",
    ),
    promotion_policy=_builtin_ref("promotion_policy", "current-c500", "v1"),
)

XPUOJ_BENCHMARK_NAMESPACE = ResearchNamespace.from_profiles(
    operator=_builtin_ref("operator", "fused-moe-w8a8-tn", "v1"),
    language=_builtin_ref("language", "triton-python", "v1"),
    evaluator=_builtin_ref("evaluator", "metax-c500", "legacy-v1"),
    evaluation_protocol=_builtin_ref(
        "evaluation_protocol", XPUOJ_C500_EVALUATION_PROTOCOL_ID, "v1"
    ),
    promotion_policy=_builtin_ref("promotion_policy", "current-c500", "v1"),
)


__all__ = [
    "BUILTIN_PROFILE_REGISTRY",
    "CURRENT_RESEARCH_NAMESPACE",
    "LEGACY_RESEARCH_NAMESPACE",
    "XPUOJ_BENCHMARK_NAMESPACE",
    "PROFILE_KINDS",
    "ProfileDefinition",
    "ProfileRef",
    "ProfileRegistry",
    "ResearchNamespace",
    "builtin_profile_registry",
]
