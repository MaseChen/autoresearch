"""Stable public contracts for the modular Autoresearch V2 host."""

from importlib import import_module

from .artifacts import (
    ARTIFACT_TAGS,
    BUNDLE_SHA256_V1,
    SOURCE_SHA256_V1,
    ArtifactId,
    tag_legacy_source_hash,
)
from .canonical import (
    canonical_json_bytes,
    canonical_json_text,
    canonical_sha256,
    require_sha256_digest,
    sha256_hex,
)
from .components import (
    CaseRole,
    DeviceBackend,
    EvaluationProtocolDefinition,
    LanguageAdapter,
    OperatorPack,
    ProfileComponentBinding,
    PromotionPolicy,
    ReplicateKind,
    StageDefinition,
    TargetComponents,
    TrustedComponentRegistry,
)
from .identity import (
    BaselineRef,
    ExecutionEnvironmentDigest,
    ExperimentIdentity,
)
from .profiles import (
    BUILTIN_PROFILE_REGISTRY,
    CURRENT_RESEARCH_NAMESPACE,
    LEGACY_RESEARCH_NAMESPACE,
    PROFILE_KINDS,
    ProfileDefinition,
    ProfileRef,
    ProfileRegistry,
    ResearchNamespace,
    builtin_profile_registry,
)
from .proposal import (
    SOURCE_BUNDLE_V1,
    SYSTEM_BUNDLE_LIMITS,
    TRITON_PYTHON_BUNDLE_LIMITS,
    BundleLimits,
    CandidateBundle,
    CandidateFile,
    ProposalV2,
)
from .proposers import (
    BUILTIN_PROPOSER_REGISTRY,
    CredentialRef,
    ProposerRequestContract,
    ResolvedProposerProfile,
    TrustedProposerRegistry,
    builtin_proposer_registry,
)
from .run_spec import (
    EXECUTION_ENVIRONMENT_BINDING_FIELDS,
    EXECUTION_ENVIRONMENT_BINDING_KEY,
    ResolvedRunSnapshot,
    RunBudget,
    RunSpecV2,
    resolve_run_spec,
)


_LAZY_LEGACY_EXPORTS = frozenset(
    {
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
    }
)


def __getattr__(name: str) -> object:
    """Load NumPy-backed legacy adapters only when explicitly requested."""

    if name not in _LAZY_LEGACY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".legacy", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LAZY_LEGACY_EXPORTS)


__all__ = [
    "ARTIFACT_TAGS",
    "BUILTIN_PROFILE_REGISTRY",
    "BUNDLE_SHA256_V1",
    "BUILTIN_PROPOSER_REGISTRY",
    "CURRENT_RESEARCH_NAMESPACE",
    "LEGACY_RESEARCH_NAMESPACE",
    "PROFILE_KINDS",
    "SOURCE_BUNDLE_V1",
    "SOURCE_SHA256_V1",
    "SYSTEM_BUNDLE_LIMITS",
    "TRITON_PYTHON_BUNDLE_LIMITS",
    "ArtifactId",
    "BaselineRef",
    "BundleLimits",
    "CaseRole",
    "CandidateBundle",
    "CandidateFile",
    "CredentialRef",
    "DeviceBackend",
    "EXECUTION_ENVIRONMENT_BINDING_FIELDS",
    "EXECUTION_ENVIRONMENT_BINDING_KEY",
    "ExecutionEnvironmentDigest",
    "ExperimentIdentity",
    "EvaluationProtocolDefinition",
    "LanguageAdapter",
    "OperatorPack",
    "ProfileComponentBinding",
    "ProfileDefinition",
    "ProfileRef",
    "ProfileRegistry",
    "ProposalV2",
    "ProposerRequestContract",
    "PromotionPolicy",
    "ReplicateKind",
    "ResearchNamespace",
    "ResolvedProposerProfile",
    "ResolvedRunSnapshot",
    "RunBudget",
    "RunSpecV2",
    "StageDefinition",
    "TargetComponents",
    "TrustedComponentRegistry",
    "TrustedProposerRegistry",
    "builtin_proposer_registry",
    "builtin_profile_registry",
    "canonical_json_bytes",
    "canonical_json_text",
    "canonical_sha256",
    "require_sha256_digest",
    "resolve_run_spec",
    "sha256_hex",
    "tag_legacy_source_hash",
    *sorted(_LAZY_LEGACY_EXPORTS),
]
