"""Inactive, trusted contracts for candidate R5 target extensions.

Nothing in this module is connected to the active evaluator or promotion
policy.  The language probes require an injected trusted runner, the native
CUDA adapter only produces compile/execution contracts, and inactive operator
descriptors can only become eligible for a later *manual* profile revision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ast
from dataclasses import dataclass
from enum import Enum
import math
import re
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable
import uuid

from .artifacts import ArtifactId, BUNDLE_SHA256_V1
from .canonical import canonical_sha256, require_sha256_digest
from .profiles import BUILTIN_PROFILE_REGISTRY, ProfileDefinition, ProfileRef
from .proposal import BundleLimits, CandidateBundle


INACTIVE = "INACTIVE"
REQUIRED_MANUAL_FULL_EVIDENCE = 2
_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]{0,191}$")
_PROFILE_DESCRIPTOR_FIELDS = frozenset(
    {
        "activation_state",
        "implementation_status",
        "agent_enabled",
        "promotion_eligible",
        "required_manual_full_evidence",
        "abi",
        "oracle",
    }
)
_TILELANG_PROFILE_FIELDS = frozenset(
    {
        "activation_state",
        "agent_enabled",
        "promotion_eligible",
        "entrypoint",
        "allowed_extensions",
        "allowed_media_types",
        "max_files",
        "max_file_bytes",
        "max_total_bytes",
        "abi",
        "probe_recipe_revision",
        "toolchain_fingerprint",
    }
)
_MACA_CUDA_PROFILE_FIELDS = frozenset(
    {
        "activation_state",
        "agent_enabled",
        "promotion_eligible",
        "entrypoint",
        "allowed_extensions",
        "allowed_media_types",
        "max_files",
        "max_file_bytes",
        "max_total_bytes",
        "abi",
        "compile_recipe",
        "execute_recipe",
        "compile_execute_separation",
        "compile_environment",
        "execute_environment",
        "network_access",
        "toolchain_fingerprint",
    }
)


def _token(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _TOKEN_RE.fullmatch(value):
        raise ValueError(f"{field} must be a stable lowercase identifier")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"{field} contains a non-canonical segment")
    return value


def _canonical_uuid(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"{field} must be a canonical UUID") from exc
    if str(parsed) != value:
        raise ValueError(f"{field} must be a canonical lowercase UUID")
    return value


def _bounded_text(value: object, *, field: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{field} exceeds its byte limit")
    return value


def _definition(kind: str, profile_id: str, revision: str) -> ProfileDefinition:
    return BUILTIN_PROFILE_REGISTRY.get(
        kind=kind,
        profile_id=profile_id,
        revision=revision,
    )


_TILELANG_DEFINITION = _definition(
    "language", "tilelang-python", "v0-probe"
)
_MACA_CUDA_DEFINITION = _definition(
    "language", "maca-cuda", "v0-contract"
)
_RAGGED_PREFILL_DEFINITION = _definition(
    "operator", "ragged-prefill", "v0-inactive"
)


TILELANG_PYTHON_BUNDLE_LIMITS = BundleLimits(
    max_files=1,
    max_file_bytes=256 * 1024,
    max_total_bytes=256 * 1024,
    allowed_extensions=frozenset({".py"}),
    allowed_media_types=frozenset({"text/x-python"}),
    required_entrypoint="kernel.py",
)
MACA_CUDA_BUNDLE_LIMITS = BundleLimits(
    max_files=1,
    max_file_bytes=512 * 1024,
    max_total_bytes=512 * 1024,
    allowed_extensions=frozenset({".cu"}),
    allowed_media_types=frozenset({"text/x-cuda"}),
    required_entrypoint="kernel.cu",
)


def _limits_from_profile(
    definition: ProfileDefinition,
    expected: BundleLimits,
    *,
    expected_fields: frozenset[str],
) -> BundleLimits:
    config = definition.config
    if set(config) != expected_fields:
        raise RuntimeError(
            f"{definition.ref.id} profile fields are not recognized"
        )
    if (
        config["activation_state"] != INACTIVE
        or config["agent_enabled"] is not False
        or config["promotion_eligible"] is not False
    ):
        raise RuntimeError(
            f"{definition.ref.id} extension profile must remain inactive"
        )
    actual = BundleLimits(
        max_files=int(config["max_files"]),
        max_file_bytes=int(config["max_file_bytes"]),
        max_total_bytes=int(config["max_total_bytes"]),
        allowed_extensions=frozenset(config["allowed_extensions"]),
        allowed_media_types=frozenset(config["allowed_media_types"]),
        required_entrypoint=str(config["entrypoint"]),
    )
    if actual != expected:
        raise RuntimeError(
            f"{definition.ref.id} profile disagrees with its bundle policy"
        )
    return expected


_TILELANG_LIMITS = _limits_from_profile(
    _TILELANG_DEFINITION,
    TILELANG_PYTHON_BUNDLE_LIMITS,
    expected_fields=_TILELANG_PROFILE_FIELDS,
)
_MACA_CUDA_LIMITS = _limits_from_profile(
    _MACA_CUDA_DEFINITION,
    MACA_CUDA_BUNDLE_LIMITS,
    expected_fields=_MACA_CUDA_PROFILE_FIELDS,
)


class ProbeStatus(str, Enum):
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"
    PASSED = "PASSED"


@dataclass(frozen=True, slots=True)
class FixedProbeRecipe:
    """A host-owned probe recipe with no runtime argv or network fields."""

    recipe_id: str
    phase: str
    capability_id: str
    argv: tuple[str, ...]
    timeout_sec: int
    max_output_bytes: int
    network_access: bool = False
    argv_policy: str = "FIXED"
    device_binding: str = "metax-c500-exclusive"
    image_binding: str = "pinned-image-digest-required"

    def __post_init__(self) -> None:
        _token(self.recipe_id, field="recipe_id")
        _token(self.capability_id, field="capability_id")
        if self.phase not in {"DOCTOR", "MICRO_COMPILE_EXECUTE"}:
            raise ValueError("unknown probe phase")
        if (
            not isinstance(self.argv, tuple)
            or not self.argv
            or any(
                not isinstance(item, str)
                or not item
                or "\x00" in item
                or len(item.encode("utf-8")) > 512
                for item in self.argv
            )
        ):
            raise ValueError("probe argv must be a fixed non-empty tuple")
        if self.argv[0] != "/usr/bin/python3" or "-I" not in self.argv:
            raise ValueError("probe recipe must use isolated pinned-image Python")
        if type(self.timeout_sec) is not int or not 1 <= self.timeout_sec <= 120:
            raise ValueError("probe timeout is outside the trusted bound")
        if (
            type(self.max_output_bytes) is not int
            or not 1 <= self.max_output_bytes <= 256 * 1024
        ):
            raise ValueError("probe output limit is outside the trusted bound")
        if self.network_access is not False:
            raise ValueError("probe recipes must disable network access")
        if self.argv_policy != "FIXED":
            raise ValueError("probe recipes may not accept dynamic argv")
        if self.device_binding != "metax-c500-exclusive":
            raise ValueError("probe recipe must bind one exclusive C500")
        if self.image_binding != "pinned-image-digest-required":
            raise ValueError("probe recipe must require a pinned image digest")

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "recipe_id": self.recipe_id,
            "phase": self.phase,
            "capability_id": self.capability_id,
            "argv": list(self.argv),
            "timeout_sec": self.timeout_sec,
            "max_output_bytes": self.max_output_bytes,
            "network_access": self.network_access,
            "argv_policy": self.argv_policy,
            "device_binding": self.device_binding,
            "image_binding": self.image_binding,
        }


TILELANG_C500_DOCTOR_RECIPE = FixedProbeRecipe(
    recipe_id="tilelang-c500-doctor-v1",
    phase="DOCTOR",
    capability_id="tilelang-c500-doctor-v1",
    argv=(
        "/usr/bin/python3",
        "-I",
        "/opt/autoresearch/probes/tilelang_c500_v1.py",
        "doctor",
        "--json",
    ),
    timeout_sec=30,
    max_output_bytes=64 * 1024,
)
TILELANG_C500_MICRO_RECIPE = FixedProbeRecipe(
    recipe_id="tilelang-c500-micro-compile-execute-v1",
    phase="MICRO_COMPILE_EXECUTE",
    capability_id="tilelang-c500-micro-compile-execute-v1",
    argv=(
        "/usr/bin/python3",
        "-I",
        "/opt/autoresearch/probes/tilelang_c500_v1.py",
        "micro-compile-execute",
        "--case",
        "vector-add-16",
        "--json",
    ),
    timeout_sec=60,
    max_output_bytes=64 * 1024,
)
if (
    _TILELANG_DEFINITION.config["abi"] != "run-kernel-v1"
    or _TILELANG_DEFINITION.config["probe_recipe_revision"]
    != "tilelang-c500-probe-v1"
    or _TILELANG_DEFINITION.config["toolchain_fingerprint"] != "UNAVAILABLE"
):
    raise RuntimeError("TileLang profile disagrees with the inactive probe contract")


@dataclass(frozen=True, slots=True)
class ProbeObservation:
    recipe_digest: str
    status: ProbeStatus
    reason: str
    evidence_digest: str | None = None
    image_digest: str | None = None
    toolchain_fingerprint: str | None = None
    device_id: str | None = None

    def __post_init__(self) -> None:
        require_sha256_digest(self.recipe_digest, field="recipe_digest")
        if not isinstance(self.status, ProbeStatus):
            try:
                object.__setattr__(self, "status", ProbeStatus(self.status))
            except (TypeError, ValueError) as exc:
                raise ValueError("unknown probe status") from exc
        _bounded_text(self.reason, field="probe reason", maximum=1024)
        if self.status is ProbeStatus.PASSED:
            if self.evidence_digest is None or self.image_digest is None:
                raise ValueError(
                    "passed probe requires evidence and pinned image digests"
                )
            require_sha256_digest(
                self.evidence_digest, field="probe evidence_digest"
            )
            require_sha256_digest(self.image_digest, field="probe image_digest")
            _bounded_text(
                self.toolchain_fingerprint,
                field="toolchain_fingerprint",
            )
            if self.device_id != "metax-c500":
                raise ValueError("passed probe must identify the MetaX C500")
        else:
            for value, field in (
                (self.evidence_digest, "probe evidence_digest"),
                (self.image_digest, "probe image_digest"),
            ):
                if value is not None:
                    require_sha256_digest(value, field=field)
            if self.toolchain_fingerprint is not None:
                _bounded_text(
                    self.toolchain_fingerprint,
                    field="toolchain_fingerprint",
                )
            if self.device_id is not None:
                _token(self.device_id, field="probe device_id")

    @classmethod
    def unavailable(
        cls, recipe: FixedProbeRecipe, reason: str
    ) -> "ProbeObservation":
        return cls(recipe.digest, ProbeStatus.UNAVAILABLE, reason)

    @classmethod
    def failed(
        cls, recipe: FixedProbeRecipe, reason: str
    ) -> "ProbeObservation":
        return cls(recipe.digest, ProbeStatus.FAILED, reason)

    @classmethod
    def passed(
        cls,
        recipe: FixedProbeRecipe,
        *,
        evidence_digest: str,
        image_digest: str,
        toolchain_fingerprint: str,
    ) -> "ProbeObservation":
        return cls(
            recipe.digest,
            ProbeStatus.PASSED,
            "trusted fixed probe completed",
            evidence_digest=evidence_digest,
            image_digest=image_digest,
            toolchain_fingerprint=toolchain_fingerprint,
            device_id="metax-c500",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "recipe_digest": self.recipe_digest,
            "status": self.status.value,
            "reason": self.reason,
            "evidence_digest": self.evidence_digest,
            "image_digest": self.image_digest,
            "toolchain_fingerprint": self.toolchain_fingerprint,
            "device_id": self.device_id,
        }


@runtime_checkable
class FixedProbeRunner(Protocol):
    """Trusted-host capability; implementations live outside this module."""

    def supports(self, capability_id: str) -> bool: ...

    def run_fixed(self, recipe: FixedProbeRecipe) -> ProbeObservation: ...


@dataclass(frozen=True, slots=True)
class UnavailableProbeRunner:
    """Safe default: it performs no subprocess, device or network action."""

    def supports(self, capability_id: str) -> bool:
        _token(capability_id, field="capability_id")
        return False

    def run_fixed(self, recipe: FixedProbeRecipe) -> ProbeObservation:
        raise RuntimeError("unavailable runner cannot execute a probe")


def _run_fixed_probe(
    recipe: FixedProbeRecipe,
    runner: FixedProbeRunner | None,
) -> ProbeObservation:
    selected: FixedProbeRunner = (
        UnavailableProbeRunner() if runner is None else runner
    )
    if not isinstance(selected, FixedProbeRunner):
        raise TypeError("runner must implement FixedProbeRunner")
    try:
        supported = selected.supports(recipe.capability_id)
    except Exception as exc:  # a trusted runner failure is bounded provenance
        return ProbeObservation.failed(
            recipe,
            f"trusted runner capability check raised {type(exc).__name__}",
        )
    if supported is not True:
        return ProbeObservation.unavailable(
            recipe, "trusted runner capability is unavailable"
        )
    try:
        result = selected.run_fixed(recipe)
    except Exception as exc:
        return ProbeObservation.failed(
            recipe,
            f"trusted runner raised {type(exc).__name__}",
        )
    if not isinstance(result, ProbeObservation):
        return ProbeObservation.failed(
            recipe, "trusted runner returned an invalid observation"
        )
    if result.recipe_digest != recipe.digest:
        return ProbeObservation.failed(
            recipe, "trusted runner returned a mismatched recipe identity"
        )
    return result


@dataclass(frozen=True, slots=True)
class TileLangPythonAdapter:
    language_id: str = _TILELANG_DEFINITION.ref.id
    implementation_id: str = _TILELANG_DEFINITION.implementation_id
    revision: str = _TILELANG_DEFINITION.ref.revision
    entrypoint: str = str(_TILELANG_DEFINITION.config["entrypoint"])
    bundle_limits: BundleLimits = _TILELANG_LIMITS
    activation_state: str = INACTIVE

    def __post_init__(self) -> None:
        if (
            self.language_id != _TILELANG_DEFINITION.ref.id
            or self.implementation_id != _TILELANG_DEFINITION.implementation_id
            or self.revision != _TILELANG_DEFINITION.ref.revision
            or self.entrypoint != _TILELANG_DEFINITION.config["entrypoint"]
            or self.bundle_limits != _TILELANG_LIMITS
            or self.activation_state != INACTIVE
        ):
            raise ValueError("TileLang adapter identity is not trusted")

    def allowed_paths(self) -> Sequence[str]:
        return (self.entrypoint,)

    def validate_language_candidate(self, source: str) -> Mapping[str, Any]:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("TileLang candidate source must not be empty")
        try:
            tree = ast.parse(source, filename=self.entrypoint)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("TileLang candidate must be valid Python syntax") from exc
        entrypoints = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if "run_kernel" not in entrypoints:
            raise ValueError("TileLang candidate must define run_kernel")
        return MappingProxyType(
            {
                "status": "STATIC_ONLY",
                "activation_state": self.activation_state,
                "entrypoint": "run_kernel",
            }
        )

    def compile_cache_material(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "language_id": self.language_id,
                "adapter_revision": self.revision,
                "toolchain_fingerprint": str(
                    _TILELANG_DEFINITION.config["toolchain_fingerprint"]
                ),
                "probe_recipe_digest": TILELANG_C500_MICRO_RECIPE.digest,
                "activation_state": self.activation_state,
            }
        )

    def doctor(
        self, runner: FixedProbeRunner | None = None
    ) -> ProbeObservation:
        return _run_fixed_probe(TILELANG_C500_DOCTOR_RECIPE, runner)

    def micro_compile_execute_probe(
        self, runner: FixedProbeRunner | None = None
    ) -> ProbeObservation:
        return _run_fixed_probe(TILELANG_C500_MICRO_RECIPE, runner)


@dataclass(frozen=True, slots=True)
class MacaCudaAbi:
    abi_id: str = "autoresearch-maca-cuda-abi-v1"
    injected_header: str = "autoresearch_maca_kernel_abi_v1.h"
    entry_symbol: str = "autoresearch_kernel_v1"
    signature: str = (
        'extern "C" int autoresearch_kernel_v1('
        "const AutoresearchKernelArgsV1*, "
        "AutoresearchKernelResultV1*, void*)"
    )

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "abi_id": self.abi_id,
            "injected_header": self.injected_header,
            "entry_symbol": self.entry_symbol,
            "signature": self.signature,
        }


MACA_CUDA_ABI = MacaCudaAbi()
MACA_CUDA_BUILD_FLAGS = (
    "-std=c++17",
    "-O3",
    "-fPIC",
    "-shared",
)
MACA_CUDA_COMPILE_RECIPE_ID = "maca-cuda-compile-v1"
MACA_CUDA_EXECUTE_RECIPE_ID = "maca-cuda-execute-v1"
MACA_CUDA_COMPILE_ENVIRONMENT = "maca-cuda-compile-sandbox-v1"
MACA_CUDA_EXECUTE_ENVIRONMENT = "maca-cuda-execute-sandbox-v1"
if (
    _MACA_CUDA_DEFINITION.config["abi"] != MACA_CUDA_ABI.abi_id
    or _MACA_CUDA_DEFINITION.config["compile_recipe"]
    != MACA_CUDA_COMPILE_RECIPE_ID
    or _MACA_CUDA_DEFINITION.config["execute_recipe"]
    != MACA_CUDA_EXECUTE_RECIPE_ID
    or _MACA_CUDA_DEFINITION.config["compile_execute_separation"] is not True
    or _MACA_CUDA_DEFINITION.config["compile_environment"]
    != MACA_CUDA_COMPILE_ENVIRONMENT
    or _MACA_CUDA_DEFINITION.config["execute_environment"]
    != MACA_CUDA_EXECUTE_ENVIRONMENT
    or _MACA_CUDA_DEFINITION.config["network_access"] is not False
    or _MACA_CUDA_DEFINITION.config["toolchain_fingerprint"] != "UNAVAILABLE"
):
    raise RuntimeError("MACA CUDA profile disagrees with its inactive contract")


@dataclass(frozen=True, slots=True)
class MacaCompileContract:
    schema_version: int
    compile_uid: str
    language_profile: ProfileRef
    candidate_artifact_id: ArtifactId
    abi_digest: str
    toolchain_digest: str
    image_digest: str
    recipe_id: str
    build_flags: tuple[str, ...]
    environment_id: str
    network_access: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError("compile contract schema_version must be 1")
        _canonical_uuid(self.compile_uid, field="compile_uid")
        if self.language_profile != _MACA_CUDA_DEFINITION.ref:
            raise ValueError("compile contract has the wrong language profile")
        if (
            not isinstance(self.candidate_artifact_id, ArtifactId)
            or self.candidate_artifact_id.tag != BUNDLE_SHA256_V1
        ):
            raise ValueError("compile contract requires a bundle artifact")
        if self.abi_digest != MACA_CUDA_ABI.digest:
            raise ValueError("compile contract has the wrong ABI digest")
        require_sha256_digest(
            self.toolchain_digest, field="compile toolchain_digest"
        )
        require_sha256_digest(self.image_digest, field="compile image_digest")
        if self.recipe_id != MACA_CUDA_COMPILE_RECIPE_ID:
            raise ValueError("compile contract has the wrong fixed recipe")
        if self.build_flags != MACA_CUDA_BUILD_FLAGS:
            raise ValueError("compile contract build flags are not trusted")
        if self.environment_id != MACA_CUDA_COMPILE_ENVIRONMENT:
            raise ValueError("compile contract has the wrong isolation environment")
        if self.network_access is not False:
            raise ValueError("compile contract must disable network access")

    @property
    def condition_digest(self) -> str:
        return canonical_sha256(self.identity_material)

    @property
    def identity_material(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "language_profile": self.language_profile.to_dict(),
            "candidate_artifact_id": str(self.candidate_artifact_id),
            "abi_digest": self.abi_digest,
            "toolchain_digest": self.toolchain_digest,
            "image_digest": self.image_digest,
            "recipe_id": self.recipe_id,
            "build_flags": list(self.build_flags),
            "environment_id": self.environment_id,
            "network_access": self.network_access,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_material,
            "compile_uid": self.compile_uid,
            "condition_digest": self.condition_digest,
        }


@dataclass(frozen=True, slots=True)
class CompiledMacaBinaryRef:
    language_profile: ProfileRef
    source_artifact_id: ArtifactId
    compile_condition_digest: str
    abi_digest: str
    binary_digest: str

    def __post_init__(self) -> None:
        if self.language_profile != _MACA_CUDA_DEFINITION.ref:
            raise ValueError("compiled binary has the wrong language profile")
        if (
            not isinstance(self.source_artifact_id, ArtifactId)
            or self.source_artifact_id.tag != BUNDLE_SHA256_V1
        ):
            raise ValueError("compiled binary must reference a source bundle")
        require_sha256_digest(
            self.compile_condition_digest,
            field="compile_condition_digest",
        )
        if self.abi_digest != MACA_CUDA_ABI.digest:
            raise ValueError("compiled binary has the wrong ABI digest")
        require_sha256_digest(self.binary_digest, field="binary_digest")

    @classmethod
    def from_compile_contract(
        cls,
        contract: MacaCompileContract,
        *,
        binary_digest: str,
    ) -> "CompiledMacaBinaryRef":
        if not isinstance(contract, MacaCompileContract):
            raise TypeError("contract must be a MacaCompileContract")
        return cls(
            language_profile=contract.language_profile,
            source_artifact_id=contract.candidate_artifact_id,
            compile_condition_digest=contract.condition_digest,
            abi_digest=contract.abi_digest,
            binary_digest=binary_digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "language_profile": self.language_profile.to_dict(),
            "source_artifact_id": str(self.source_artifact_id),
            "compile_condition_digest": self.compile_condition_digest,
            "abi_digest": self.abi_digest,
            "binary_digest": self.binary_digest,
        }


@dataclass(frozen=True, slots=True)
class MacaExecutionContract:
    schema_version: int
    execution_uid: str
    language_profile: ProfileRef
    compiled_binary: CompiledMacaBinaryRef
    abi_digest: str
    input_digest: str
    image_digest: str
    case_id: str
    recipe_id: str
    environment_id: str
    network_access: bool = False
    device_binding: str = "metax-c500-exclusive"

    def __post_init__(self) -> None:
        if self.schema_version != 1 or type(self.schema_version) is not int:
            raise ValueError("execution contract schema_version must be 1")
        _canonical_uuid(self.execution_uid, field="execution_uid")
        if self.language_profile != _MACA_CUDA_DEFINITION.ref:
            raise ValueError("execution contract has the wrong language profile")
        if not isinstance(self.compiled_binary, CompiledMacaBinaryRef):
            raise ValueError("execution requires a compiled binary reference")
        if self.compiled_binary.language_profile != self.language_profile:
            raise ValueError("compiled binary language profile is inconsistent")
        if self.abi_digest != MACA_CUDA_ABI.digest:
            raise ValueError("execution contract has the wrong ABI digest")
        require_sha256_digest(self.input_digest, field="input_digest")
        require_sha256_digest(self.image_digest, field="execution image_digest")
        _token(self.case_id, field="case_id")
        if self.recipe_id != MACA_CUDA_EXECUTE_RECIPE_ID:
            raise ValueError("execution contract has the wrong fixed recipe")
        if self.environment_id != MACA_CUDA_EXECUTE_ENVIRONMENT:
            raise ValueError("execution contract has the wrong isolation environment")
        if self.network_access is not False:
            raise ValueError("execution contract must disable network access")
        if self.device_binding != "metax-c500-exclusive":
            raise ValueError("execution contract must bind one exclusive C500")

    @property
    def condition_digest(self) -> str:
        return canonical_sha256(self.identity_material)

    @property
    def identity_material(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "language_profile": self.language_profile.to_dict(),
            "compiled_binary": self.compiled_binary.to_dict(),
            "abi_digest": self.abi_digest,
            "input_digest": self.input_digest,
            "image_digest": self.image_digest,
            "case_id": self.case_id,
            "recipe_id": self.recipe_id,
            "environment_id": self.environment_id,
            "network_access": self.network_access,
            "device_binding": self.device_binding,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_material,
            "execution_uid": self.execution_uid,
            "condition_digest": self.condition_digest,
        }


@dataclass(frozen=True, slots=True)
class MACACudaAdapter:
    language_id: str = _MACA_CUDA_DEFINITION.ref.id
    implementation_id: str = _MACA_CUDA_DEFINITION.implementation_id
    revision: str = _MACA_CUDA_DEFINITION.ref.revision
    entrypoint: str = str(_MACA_CUDA_DEFINITION.config["entrypoint"])
    bundle_limits: BundleLimits = _MACA_CUDA_LIMITS
    abi: MacaCudaAbi = MACA_CUDA_ABI
    activation_state: str = INACTIVE

    def __post_init__(self) -> None:
        if (
            self.language_id != _MACA_CUDA_DEFINITION.ref.id
            or self.implementation_id != _MACA_CUDA_DEFINITION.implementation_id
            or self.revision != _MACA_CUDA_DEFINITION.ref.revision
            or self.entrypoint != _MACA_CUDA_DEFINITION.config["entrypoint"]
            or self.bundle_limits != _MACA_CUDA_LIMITS
            or self.abi != MACA_CUDA_ABI
            or self.activation_state != INACTIVE
        ):
            raise ValueError("MACA CUDA adapter identity is not trusted")

    def allowed_paths(self) -> Sequence[str]:
        return (self.entrypoint,)

    def validate_language_candidate(self, source: str) -> Mapping[str, Any]:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("MACA CUDA candidate source must not be empty")
        include = f'#include "{self.abi.injected_header}"'
        if include not in source:
            raise ValueError("MACA CUDA candidate must include the fixed ABI header")
        optional_name = r"(?:[A-Za-z_][A-Za-z0-9_]*)?"
        signature = re.compile(
            r'extern\s+"C"\s+int\s+'
            + re.escape(self.abi.entry_symbol)
            + r"\s*\(\s*const\s+AutoresearchKernelArgsV1\s*\*\s*"
            + optional_name
            + r"\s*,\s*AutoresearchKernelResultV1\s*\*\s*"
            + optional_name
            + r"\s*,\s*void\s*\*\s*"
            + optional_name
            + r"\s*\)"
        )
        if signature.search(source) is None:
            raise ValueError("MACA CUDA candidate must match the fixed ABI")
        return MappingProxyType(
            {
                "status": "STATIC_ONLY",
                "activation_state": self.activation_state,
                "abi_digest": self.abi.digest,
            }
        )

    def compile_cache_material(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "language_id": self.language_id,
                "adapter_revision": self.revision,
                "abi_digest": self.abi.digest,
                "compile_recipe": MACA_CUDA_COMPILE_RECIPE_ID,
                "build_flags": MACA_CUDA_BUILD_FLAGS,
                "compile_environment": MACA_CUDA_COMPILE_ENVIRONMENT,
                "execute_environment": MACA_CUDA_EXECUTE_ENVIRONMENT,
                "network_access": False,
                "activation_state": self.activation_state,
            }
        )

    def make_compile_contract(
        self,
        bundle: CandidateBundle,
        *,
        toolchain_digest: str,
        image_digest: str,
        compile_uid: str | None = None,
    ) -> MacaCompileContract:
        if not isinstance(bundle, CandidateBundle):
            raise TypeError("bundle must be a CandidateBundle")
        bundle.validate(self.bundle_limits)
        source = next(
            file.content
            for file in bundle.files
            if file.path == self.entrypoint
        )
        self.validate_language_candidate(source)
        return MacaCompileContract(
            schema_version=1,
            compile_uid=(str(uuid.uuid4()) if compile_uid is None else compile_uid),
            language_profile=_MACA_CUDA_DEFINITION.ref,
            candidate_artifact_id=bundle.artifact_id,
            abi_digest=self.abi.digest,
            toolchain_digest=toolchain_digest,
            image_digest=image_digest,
            recipe_id=MACA_CUDA_COMPILE_RECIPE_ID,
            build_flags=MACA_CUDA_BUILD_FLAGS,
            environment_id=MACA_CUDA_COMPILE_ENVIRONMENT,
        )

    def make_execution_contract(
        self,
        compiled_binary: CompiledMacaBinaryRef,
        *,
        input_digest: str,
        image_digest: str,
        case_id: str,
        execution_uid: str | None = None,
    ) -> MacaExecutionContract:
        return MacaExecutionContract(
            schema_version=1,
            execution_uid=(
                str(uuid.uuid4()) if execution_uid is None else execution_uid
            ),
            language_profile=_MACA_CUDA_DEFINITION.ref,
            compiled_binary=compiled_binary,
            abi_digest=self.abi.digest,
            input_digest=input_digest,
            image_digest=image_digest,
            case_id=case_id,
            recipe_id=MACA_CUDA_EXECUTE_RECIPE_ID,
            environment_id=MACA_CUDA_EXECUTE_ENVIRONMENT,
        )


RAGGED_PREFILL_INVARIANTS = (
    "q, k and v are rank-3 [token, head, feature] tensors",
    "q_offsets and kv_offsets are monotonic batch boundaries",
    "query heads are divisible by key/value heads",
    "causal sequences have kv_length >= q_length",
    "attention softmax is accumulated in float64 in the CPU oracle",
)
RAGGED_ORACLE_MAX_TOKENS = 512
RAGGED_ORACLE_MAX_HEADS = 64
RAGGED_ORACLE_MAX_FEATURES = 256
RAGGED_ORACLE_MAX_INTERMEDIATES = 8 * 1024 * 1024


class OracleUnavailableError(RuntimeError):
    pass


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise OracleUnavailableError(
            "NumPy is unavailable for the CPU-only Ragged Prefill oracle"
        ) from exc
    return np


def _offsets(np: Any, value: Any, *, total: int, field: str) -> Any:
    result = np.asarray(value)
    if result.ndim != 1 or result.size < 2:
        raise ValueError(f"{field} must be a rank-1 batch boundary array")
    if not np.issubdtype(result.dtype, np.integer):
        raise ValueError(f"{field} must contain integers")
    result = result.astype(np.int64, copy=False)
    if result[0] != 0 or result[-1] != total:
        raise ValueError(f"{field} must start at zero and end at token count")
    if np.any(result[1:] < result[:-1]):
        raise ValueError(f"{field} must be monotonic")
    return result


def ragged_prefill_oracle(
    q: Any,
    k: Any,
    v: Any,
    q_offsets: Any,
    kv_offsets: Any,
    *,
    causal: bool = True,
    scale: float | None = None,
) -> Any:
    """Small, CPU-only reference for ragged prefill attention.

    This intentionally favors clarity over performance and is not an evaluator
    or hardware result.  NumPy is imported only when the oracle is invoked.
    """

    np = _numpy()
    q_array = np.asarray(q)
    k_array = np.asarray(k)
    v_array = np.asarray(v)
    if any(array.ndim != 3 for array in (q_array, k_array, v_array)):
        raise ValueError("q, k and v must all be rank-3 tensors")
    if k_array.shape[:2] != v_array.shape[:2]:
        raise ValueError("k and v token/head dimensions must match")
    if q_array.shape[2] != k_array.shape[2]:
        raise ValueError("q and k head dimensions must match")
    q_heads = q_array.shape[1]
    kv_heads = k_array.shape[1]
    if q_array.shape[2] <= 0 or v_array.shape[2] <= 0:
        raise ValueError("attention feature dimensions must be positive")
    if q_heads <= 0 or kv_heads <= 0 or q_heads % kv_heads != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    if not isinstance(causal, bool):
        raise ValueError("causal must be a boolean")
    q_boundaries = _offsets(
        np, q_offsets, total=q_array.shape[0], field="q_offsets"
    )
    kv_boundaries = _offsets(
        np, kv_offsets, total=k_array.shape[0], field="kv_offsets"
    )
    if q_boundaries.size != kv_boundaries.size:
        raise ValueError("q_offsets and kv_offsets must describe the same batch")
    if not all(
        (
            np.issubdtype(array.dtype, np.floating)
            or np.issubdtype(array.dtype, np.integer)
        )
        and np.all(np.isfinite(array))
        for array in (q_array, k_array, v_array)
    ):
        raise ValueError("q, k and v must contain finite numeric values")
    if (
        q_array.shape[0] > RAGGED_ORACLE_MAX_TOKENS
        or k_array.shape[0] > RAGGED_ORACLE_MAX_TOKENS
        or q_heads > RAGGED_ORACLE_MAX_HEADS
        or kv_heads > RAGGED_ORACLE_MAX_HEADS
        or q_array.shape[2] > RAGGED_ORACLE_MAX_FEATURES
        or v_array.shape[2] > RAGGED_ORACLE_MAX_FEATURES
        or q_array.size > RAGGED_ORACLE_MAX_INTERMEDIATES
        or k_array.size > RAGGED_ORACLE_MAX_INTERMEDIATES
        or v_array.size > RAGGED_ORACLE_MAX_INTERMEDIATES
    ):
        raise ValueError("Ragged Prefill oracle input exceeds the small-scale limit")
    if isinstance(scale, (bool, np.bool_)):
        raise ValueError("scale must be a positive finite number")
    try:
        resolved_scale = (
            1.0 / math.sqrt(q_array.shape[2])
            if scale is None
            else float(scale)
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scale must be a positive finite number") from exc
    if not math.isfinite(resolved_scale) or resolved_scale <= 0:
        raise ValueError("scale must be a positive finite number")

    q_float = q_array.astype(np.float64, copy=False)
    k_float = k_array.astype(np.float64, copy=False)
    v_float = v_array.astype(np.float64, copy=False)
    output = np.empty(
        (q_array.shape[0], q_heads, v_array.shape[2]), dtype=np.float64
    )
    heads_per_kv = q_heads // kv_heads
    score_elements = 0
    for batch_index in range(q_boundaries.size - 1):
        q_start, q_end = (
            int(q_boundaries[batch_index]),
            int(q_boundaries[batch_index + 1]),
        )
        kv_start, kv_end = (
            int(kv_boundaries[batch_index]),
            int(kv_boundaries[batch_index + 1]),
        )
        q_length = q_end - q_start
        kv_length = kv_end - kv_start
        if q_length and not kv_length:
            raise ValueError("a non-empty query sequence requires key/value tokens")
        if causal and kv_length < q_length:
            raise ValueError("causal ragged prefill requires kv_length >= q_length")
        score_elements += q_length * kv_length * q_heads
        if score_elements > RAGGED_ORACLE_MAX_INTERMEDIATES:
            raise ValueError(
                "Ragged Prefill oracle scores exceed the small-scale limit"
            )
        prefix_length = kv_length - q_length
        for query_head in range(q_heads):
            kv_head = query_head // heads_per_kv
            scores = (
                q_float[q_start:q_end, query_head]
                @ k_float[kv_start:kv_end, kv_head].T
            ) * float(resolved_scale)
            if causal and q_length:
                query_positions = np.arange(q_length)[:, None]
                key_positions = np.arange(kv_length)[None, :]
                scores = np.where(
                    key_positions <= prefix_length + query_positions,
                    scores,
                    -np.inf,
                )
            if q_length:
                scores -= np.max(scores, axis=1, keepdims=True)
                weights = np.exp(scores)
                weights /= np.sum(weights, axis=1, keepdims=True)
                output[q_start:q_end, query_head] = (
                    weights @ v_float[kv_start:kv_end, kv_head]
                )
    return output


@dataclass(frozen=True, slots=True)
class RaggedPrefillPack:
    operator_id: str = _RAGGED_PREFILL_DEFINITION.ref.id
    implementation_id: str = _RAGGED_PREFILL_DEFINITION.implementation_id
    revision: str = _RAGGED_PREFILL_DEFINITION.ref.revision
    activation_state: str = INACTIVE

    def __post_init__(self) -> None:
        if (
            self.operator_id != _RAGGED_PREFILL_DEFINITION.ref.id
            or self.implementation_id
            != _RAGGED_PREFILL_DEFINITION.implementation_id
            or self.revision != _RAGGED_PREFILL_DEFINITION.ref.revision
            or self.activation_state != INACTIVE
        ):
            raise ValueError("Ragged Prefill pack identity is not trusted")

    def contract_summary(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "abi": "ragged-prefill-v1",
                "entrypoint": "run_kernel",
                "layout": "token-head-feature",
                "supports_causal_prefix": True,
                "oracle": "ragged-prefill-numpy-v1",
                "activation_state": self.activation_state,
            }
        )

    def prompt_context(self) -> str:
        return (
            "Optimize ragged prefill attention over q_offsets and kv_offsets. "
            "Preserve GQA head mapping, causal prefix masking, stable softmax, "
            "and every empty-sequence boundary; this profile remains inactive."
        )

    def invariants(self) -> tuple[str, ...]:
        return RAGGED_PREFILL_INVARIANTS

    def validate_operator_candidate(self, source: str) -> Mapping[str, Any]:
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Ragged Prefill candidate source must not be empty")
        return MappingProxyType(
            {
                "status": "CONTRACT_ONLY",
                "activation_state": self.activation_state,
                "invariants": self.invariants(),
            }
        )

    def oracle(
        self,
        q: Any,
        k: Any,
        v: Any,
        q_offsets: Any,
        kv_offsets: Any,
        *,
        causal: bool = True,
        scale: float | None = None,
    ) -> Any:
        return ragged_prefill_oracle(
            q,
            k,
            v,
            q_offsets,
            kv_offsets,
            causal=causal,
            scale=scale,
        )


@dataclass(frozen=True, slots=True)
class ManualFullEvidence:
    operator_profile: ProfileRef
    experiment_uid: str
    evidence_digest: str
    reviewer: str
    stage_id: str = "FULL_PRIMARY"
    execution_mode: str = "MANUAL"
    verdict: str = "PASSED"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.operator_profile, ProfileRef)
            or self.operator_profile.kind != "operator"
        ):
            raise ValueError("manual evidence requires an operator ProfileRef")
        _canonical_uuid(self.experiment_uid, field="experiment_uid")
        require_sha256_digest(self.evidence_digest, field="evidence_digest")
        _bounded_text(self.reviewer, field="reviewer")
        if self.stage_id != "FULL_PRIMARY":
            raise ValueError("activation evidence must be FULL_PRIMARY")
        if self.execution_mode != "MANUAL":
            raise ValueError("activation evidence must be manual")
        if self.verdict != "PASSED":
            raise ValueError("activation evidence must have passed")


@dataclass(frozen=True, slots=True)
class ActivationReview:
    operator_profile: ProfileRef
    activation_state: str
    accepted_evidence: int
    required_evidence: int
    eligible_for_manual_profile_revision: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.operator_profile, ProfileRef)
            or self.operator_profile.kind != "operator"
        ):
            raise ValueError("activation review requires an operator profile")
        if self.activation_state != INACTIVE:
            raise ValueError("this module cannot produce an active state")
        if (
            type(self.accepted_evidence) is not int
            or self.accepted_evidence < 0
            or self.required_evidence != REQUIRED_MANUAL_FULL_EVIDENCE
        ):
            raise ValueError("activation review evidence counts are invalid")
        expected = self.accepted_evidence >= self.required_evidence
        if self.eligible_for_manual_profile_revision is not expected:
            raise ValueError("activation review eligibility is inconsistent")
        if not isinstance(self.reasons, tuple) or not self.reasons:
            raise ValueError("activation review must explain its decision")


@dataclass(frozen=True, slots=True)
class InactiveOperatorDescriptor:
    profile: ProfileRef
    implementation_status: str
    abi_id: str
    oracle_id: str
    required_manual_full_evidence: int = REQUIRED_MANUAL_FULL_EVIDENCE
    activation_state: str = INACTIVE
    agent_enabled: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ProfileRef) or self.profile.kind != "operator":
            raise ValueError("inactive descriptor requires an operator profile")
        if self.implementation_status not in {"oracle-ready", "descriptor-only"}:
            raise ValueError("unknown inactive implementation status")
        if self.required_manual_full_evidence != 2:
            raise ValueError("inactive operators require two manual full results")
        if (
            self.activation_state != INACTIVE
            or self.agent_enabled is not False
            or self.promotion_eligible is not False
        ):
            raise ValueError("inactive descriptors cannot enable agent or promotion")
        _token(self.abi_id, field="abi_id")
        _token(self.oracle_id, field="oracle_id")

    def review_manual_evidence(
        self, evidence: Sequence[ManualFullEvidence]
    ) -> ActivationReview:
        if isinstance(evidence, (str, bytes)):
            raise TypeError("evidence must be a sequence of manual full records")
        seen_uids: set[str] = set()
        seen_digests: set[str] = set()
        accepted = 0
        reasons: list[str] = []
        for index, item in enumerate(evidence):
            if not isinstance(item, ManualFullEvidence):
                reasons.append(f"evidence[{index}] has an invalid contract")
                continue
            if item.operator_profile != self.profile:
                reasons.append(f"evidence[{index}] targets another profile")
                continue
            if (
                item.experiment_uid in seen_uids
                or item.evidence_digest in seen_digests
            ):
                reasons.append(f"evidence[{index}] is not independent")
                continue
            seen_uids.add(item.experiment_uid)
            seen_digests.add(item.evidence_digest)
            accepted += 1
        eligible = accepted >= self.required_manual_full_evidence
        if not eligible:
            reasons.append(
                "two independent manual FULL_PRIMARY records are required"
            )
        else:
            reasons.append(
                "evidence permits human review only; publish a new profile "
                "revision to activate"
            )
        return ActivationReview(
            operator_profile=self.profile,
            activation_state=INACTIVE,
            accepted_evidence=accepted,
            required_evidence=self.required_manual_full_evidence,
            eligible_for_manual_profile_revision=eligible,
            reasons=tuple(reasons),
        )


INACTIVE_OPERATOR_PROFILE_IDS = (
    "ragged-prefill",
    "paged-decode",
    "paged-prefill",
    "kv-cache-decode",
    "mla",
)
DESCRIPTOR_ONLY_OPERATOR_IDS = (
    "paged-decode",
    "paged-prefill",
    "kv-cache-decode",
    "mla",
)


def resolve_inactive_operator_descriptor(
    ref: ProfileRef | Mapping[str, Any],
) -> InactiveOperatorDescriptor:
    if not isinstance(ref, ProfileRef):
        ref = ProfileRef.from_value(ref)
    definition = BUILTIN_PROFILE_REGISTRY.resolve(ref)
    if (
        definition.ref.kind != "operator"
        or definition.ref.id not in INACTIVE_OPERATOR_PROFILE_IDS
    ):
        raise ValueError("profile is not a trusted inactive operator")
    config = definition.config
    if set(config) != _PROFILE_DESCRIPTOR_FIELDS:
        raise ValueError("inactive operator profile fields are not recognized")
    if (
        config["activation_state"] != INACTIVE
        or config["agent_enabled"] is not False
        or config["promotion_eligible"] is not False
        or config["required_manual_full_evidence"]
        != REQUIRED_MANUAL_FULL_EVIDENCE
    ):
        raise ValueError("inactive operator profile violates the activation gate")
    return InactiveOperatorDescriptor(
        profile=definition.ref,
        implementation_status=str(config["implementation_status"]),
        abi_id=str(config["abi"]),
        oracle_id=str(config["oracle"]),
    )


INACTIVE_OPERATOR_DESCRIPTORS = MappingProxyType(
    {
        profile_id: resolve_inactive_operator_descriptor(
            _definition("operator", profile_id, "v0-inactive").ref
        )
        for profile_id in INACTIVE_OPERATOR_PROFILE_IDS
    }
)
RAGGED_PREFILL_DESCRIPTOR = INACTIVE_OPERATOR_DESCRIPTORS["ragged-prefill"]
DESCRIPTOR_ONLY_OPERATORS = tuple(
    INACTIVE_OPERATOR_DESCRIPTORS[profile_id]
    for profile_id in DESCRIPTOR_ONLY_OPERATOR_IDS
)


__all__ = [
    "DESCRIPTOR_ONLY_OPERATORS",
    "DESCRIPTOR_ONLY_OPERATOR_IDS",
    "INACTIVE",
    "INACTIVE_OPERATOR_DESCRIPTORS",
    "INACTIVE_OPERATOR_PROFILE_IDS",
    "MACACudaAdapter",
    "MACA_CUDA_ABI",
    "MACA_CUDA_BUILD_FLAGS",
    "MACA_CUDA_BUNDLE_LIMITS",
    "MACA_CUDA_COMPILE_RECIPE_ID",
    "MACA_CUDA_COMPILE_ENVIRONMENT",
    "MACA_CUDA_EXECUTE_RECIPE_ID",
    "MACA_CUDA_EXECUTE_ENVIRONMENT",
    "RAGGED_PREFILL_DESCRIPTOR",
    "RAGGED_PREFILL_INVARIANTS",
    "RAGGED_ORACLE_MAX_FEATURES",
    "RAGGED_ORACLE_MAX_HEADS",
    "RAGGED_ORACLE_MAX_INTERMEDIATES",
    "RAGGED_ORACLE_MAX_TOKENS",
    "REQUIRED_MANUAL_FULL_EVIDENCE",
    "TILELANG_C500_DOCTOR_RECIPE",
    "TILELANG_C500_MICRO_RECIPE",
    "TILELANG_PYTHON_BUNDLE_LIMITS",
    "ActivationReview",
    "CompiledMacaBinaryRef",
    "FixedProbeRecipe",
    "FixedProbeRunner",
    "InactiveOperatorDescriptor",
    "MacaCompileContract",
    "MacaCudaAbi",
    "MacaExecutionContract",
    "ManualFullEvidence",
    "OracleUnavailableError",
    "ProbeObservation",
    "ProbeStatus",
    "RaggedPrefillPack",
    "TileLangPythonAdapter",
    "UnavailableProbeRunner",
    "ragged_prefill_oracle",
    "resolve_inactive_operator_descriptor",
]
