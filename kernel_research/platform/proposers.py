"""Trusted, dependency-light proposer composition and request contracts.

This module resolves ``ModelProfile x HarnessAdapter x PromptProtocol`` from
an exact built-in proposer ``ProfileRef``.  It only constructs immutable
request/provenance values; no adapter performs I/O or reads a credential.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import re
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable
import uuid

from ..constants import MAX_PROPOSER_TIMEOUT_SEC
from .artifacts import ArtifactId
from .canonical import (
    canonical_json_text,
    canonical_sha256,
    require_sha256_digest,
    sha256_hex,
)
from .profiles import (
    BUILTIN_PROFILE_REGISTRY,
    ProfileDefinition,
    ProfileRef,
    ProfileRegistry,
)


MAX_PROPOSER_PROMPT_BYTES = 2 * 1024 * 1024
BUILTIN_PROPOSER_PROFILE_IDS = (
    "direct-api-deepseek-v4-flash",
    "direct-api-deepseek-v4-pro",
    "opencode-deepseek-v4-flash",
    "opencode-deepseek-v4-pro",
    "pi-deepseek-v4-flash",
    "pi-deepseek-v4-pro",
)
_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9._/-]{0,191}$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_CREDENTIAL_REF_RE = re.compile(
    r"^credential-ref:v1:[a-z0-9][a-z0-9._/-]{0,191}$"
)
_PROPOSER_CONFIG_FIELDS = frozenset(
    {
        "harness",
        "harness_revision",
        "model",
        "model_revision",
        "prompt_protocol",
        "prompt_protocol_revision",
    }
)
_SECRET_FIELDS = frozenset(
    {
        "access_key",
        "access_key_id",
        "api_key",
        "api_key_value",
        "access_token",
        "authorization",
        "bearer_token",
        "client_secret",
        "credential",
        "credential_value",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "secret_access_key",
        "secret_value",
        "token",
    }
)


def _token(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _TOKEN_RE.fullmatch(value):
        raise ValueError(f"{field} must be a stable lowercase identifier")
    if "//" in value or "/../" in f"/{value}/":
        raise ValueError(f"{field} contains a non-canonical segment")
    return value


def _revision(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _REVISION_RE.fullmatch(value):
        raise ValueError(f"{field} must be a stable revision")
    return value


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _plain(value: Any) -> Any:
    return json.loads(canonical_json_text(value))


def _frozen(value: Any) -> Any:
    normalized = _plain(value)

    def freeze(child: Any) -> Any:
        if isinstance(child, dict):
            return MappingProxyType(
                {key: freeze(item) for key, item in child.items()}
            )
        if isinstance(child, list):
            return tuple(freeze(item) for item in child)
        return child

    return freeze(normalized)


def _reject_secret_fields(value: Any, *, path: str = "payload") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_string = str(key)
            if key_string.lower() in _SECRET_FIELDS:
                raise ValueError(
                    f"secret content field is forbidden in {path}.{key_string}"
                )
            _reject_secret_fields(child, path=f"{path}.{key_string}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_secret_fields(child, path=f"{path}[{index}]")


def _canonical_uuid(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{field} must be a canonical UUID") from exc
    if str(parsed) != value:
        raise ValueError(f"{field} must be a canonical lowercase UUID")
    return value


@dataclass(frozen=True, slots=True)
class CredentialRef:
    """An opaque lookup name; it can never carry credential content."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or not _CREDENTIAL_REF_RE.fullmatch(
            self.value
        ):
            raise ValueError(
                "credential reference must use credential-ref:v1:<stable-id>"
            )
        suffix = self.value.removeprefix("credential-ref:v1:")
        if any(segment in {"", ".", ".."} for segment in suffix.split("/")):
            raise ValueError("credential reference contains a non-canonical segment")

    @classmethod
    def parse(cls, value: object) -> "CredentialRef":
        return value if isinstance(value, cls) else cls(value)  # type: ignore[arg-type]

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ModelProfile:
    model_id: str
    revision: str
    provider_id: str
    provider_model_id: str
    display_name: str
    context_tokens: int
    output_tokens: int
    request_output_token_cap: int
    reasoning_effort: str

    def __post_init__(self) -> None:
        _token(self.model_id, field="model_id")
        _revision(self.revision, field="model revision")
        _token(self.provider_id, field="provider_id")
        _token(self.provider_model_id, field="provider_model_id")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("display_name must be a non-empty string")
        context = _positive_int(self.context_tokens, field="context_tokens")
        output = _positive_int(self.output_tokens, field="output_tokens")
        cap = _positive_int(
            self.request_output_token_cap,
            field="request_output_token_cap",
        )
        if output > context:
            raise ValueError("output_tokens may not exceed context_tokens")
        if cap > output:
            raise ValueError("request_output_token_cap may not exceed output_tokens")
        if self.reasoning_effort not in {"high", "max"}:
            raise ValueError("reasoning_effort must be high or max")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "provider_id": self.provider_id,
            "provider_model_id": self.provider_model_id,
            "display_name": self.display_name,
            "context_tokens": self.context_tokens,
            "output_tokens": self.output_tokens,
            "request_output_token_cap": self.request_output_token_cap,
            "reasoning_effort": self.reasoning_effort,
        }


@dataclass(frozen=True, slots=True)
class PromptProtocol:
    protocol_id: str
    revision: str
    proposal_schema_version: int
    response_format: str = "strict-json-object"

    def __post_init__(self) -> None:
        _token(self.protocol_id, field="prompt protocol_id")
        _revision(self.revision, field="prompt protocol revision")
        if type(self.proposal_schema_version) is not int or (
            self.proposal_schema_version not in {1, 2}
        ):
            raise ValueError("proposal_schema_version must be 1 or 2")
        if self.response_format != "strict-json-object":
            raise ValueError("response_format must be strict-json-object")

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "revision": self.revision,
            "proposal_schema_version": self.proposal_schema_version,
            "response_format": self.response_format,
        }


@dataclass(frozen=True, slots=True)
class HarnessInvocation:
    harness_id: str
    revision: str
    transport: str
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        _token(self.harness_id, field="harness_id")
        _revision(self.revision, field="harness revision")
        if self.transport not in {"direct-api", "opencode", "pi"}:
            raise ValueError("unknown proposer transport")
        if not isinstance(self.payload, Mapping):
            raise ValueError("harness payload must be a mapping")
        _reject_secret_fields(self.payload)
        object.__setattr__(self, "payload", _frozen(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "harness_id": self.harness_id,
            "revision": self.revision,
            "transport": self.transport,
            "payload": _plain(self.payload),
        }


@runtime_checkable
class HarnessAdapter(Protocol):
    harness_id: str
    revision: str

    def build_invocation(
        self,
        *,
        model: ModelProfile,
        prompt_protocol: PromptProtocol,
        prompt: str,
        max_output_tokens: int,
    ) -> HarnessInvocation: ...


def _response_contract(protocol: PromptProtocol) -> dict[str, Any]:
    return {
        "protocol_id": protocol.protocol_id,
        "revision": protocol.revision,
        "proposal_schema_version": protocol.proposal_schema_version,
        "format": protocol.response_format,
    }


@dataclass(frozen=True, slots=True)
class DirectAPIHarnessAdapter:
    harness_id: str = "direct-api"
    revision: str = "v1"

    def build_invocation(
        self,
        *,
        model: ModelProfile,
        prompt_protocol: PromptProtocol,
        prompt: str,
        max_output_tokens: int,
    ) -> HarnessInvocation:
        return HarnessInvocation(
            self.harness_id,
            self.revision,
            "direct-api",
            {
                "provider_id": model.provider_id,
                "provider_model_id": model.provider_model_id,
                "messages": ({"role": "user", "content": prompt},),
                "max_output_tokens": max_output_tokens,
                "reasoning_effort": model.reasoning_effort,
                "response_contract": _response_contract(prompt_protocol),
            },
        )


@dataclass(frozen=True, slots=True)
class OpenCodeHarnessAdapter:
    harness_id: str = "opencode"
    revision: str = "v1"

    def build_invocation(
        self,
        *,
        model: ModelProfile,
        prompt_protocol: PromptProtocol,
        prompt: str,
        max_output_tokens: int,
    ) -> HarnessInvocation:
        return HarnessInvocation(
            self.harness_id,
            self.revision,
            "opencode",
            {
                "model": model.model_id,
                "input": prompt,
                "format": "json-events",
                "max_output_tokens": max_output_tokens,
                "reasoning_effort": model.reasoning_effort,
                "response_contract": _response_contract(prompt_protocol),
            },
        )


@dataclass(frozen=True, slots=True)
class PiHarnessAdapter:
    harness_id: str = "pi"
    revision: str = "v1"

    def build_invocation(
        self,
        *,
        model: ModelProfile,
        prompt_protocol: PromptProtocol,
        prompt: str,
        max_output_tokens: int,
    ) -> HarnessInvocation:
        return HarnessInvocation(
            self.harness_id,
            self.revision,
            "pi",
            {
                "model": model.model_id,
                "input": prompt,
                "session_mode": "non-interactive",
                "max_output_tokens": max_output_tokens,
                "reasoning_effort": model.reasoning_effort,
                "response_contract": _response_contract(prompt_protocol),
            },
        )


@dataclass(frozen=True, slots=True)
class ResolvedProposerProfile:
    profile: ProfileRef
    implementation_id: str
    model: ModelProfile
    harness: HarnessAdapter
    prompt_protocol: PromptProtocol

    def __post_init__(self) -> None:
        if not isinstance(self.profile, ProfileRef) or self.profile.kind != "proposer":
            raise ValueError("profile must be a proposer ProfileRef")
        _token(self.implementation_id, field="implementation_id")
        if not isinstance(self.model, ModelProfile):
            raise ValueError("model must be a ModelProfile")
        if not isinstance(self.harness, HarnessAdapter):
            raise ValueError("harness must implement HarnessAdapter")
        if not isinstance(self.prompt_protocol, PromptProtocol):
            raise ValueError("prompt_protocol must be a PromptProtocol")

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "implementation_id": self.implementation_id,
            "model": self.model.to_dict(),
            "harness": {
                "harness_id": self.harness.harness_id,
                "revision": self.harness.revision,
            },
            "prompt_protocol": self.prompt_protocol.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ProposerRequestContract:
    schema_version: int
    request_id: str
    proposer: ResolvedProposerProfile
    proposal_context_id: str
    parent_artifact_id: ArtifactId
    prompt: str
    credential_ref: CredentialRef
    timeout_sec: float
    max_output_tokens: int
    invocation: HarnessInvocation

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("proposer request schema_version must be 1")
        _canonical_uuid(self.request_id, field="request_id")
        if not isinstance(self.proposer, ResolvedProposerProfile):
            raise ValueError("proposer must be a ResolvedProposerProfile")
        require_sha256_digest(
            self.proposal_context_id, field="proposal_context_id"
        )
        if not isinstance(self.parent_artifact_id, ArtifactId):
            raise ValueError("parent_artifact_id must be an ArtifactId")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        try:
            prompt_bytes = self.prompt.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("prompt contains invalid Unicode") from exc
        if len(prompt_bytes) > MAX_PROPOSER_PROMPT_BYTES:
            raise ValueError("prompt exceeds the trusted host limit in bytes")
        if not isinstance(self.credential_ref, CredentialRef):
            raise ValueError("credential_ref must be a CredentialRef")
        if isinstance(self.timeout_sec, bool) or not isinstance(
            self.timeout_sec, (int, float)
        ):
            raise ValueError("timeout_sec must be a positive number")
        if not 0 < float(self.timeout_sec) <= MAX_PROPOSER_TIMEOUT_SEC:
            raise ValueError("timeout_sec exceeds the trusted host limit")
        cap = _positive_int(
            self.max_output_tokens, field="max_output_tokens"
        )
        if cap > self.proposer.model.request_output_token_cap:
            raise ValueError("max_output_tokens exceeds the model profile cap")
        if not isinstance(self.invocation, HarnessInvocation):
            raise ValueError("invocation must be a HarnessInvocation")
        expected = self.proposer.harness.build_invocation(
            model=self.proposer.model,
            prompt_protocol=self.proposer.prompt_protocol,
            prompt=self.prompt,
            max_output_tokens=cap,
        )
        if expected != self.invocation:
            raise ValueError("invocation does not match the resolved proposer")

    @property
    def prompt_digest(self) -> str:
        return f"sha256:{sha256_hex(self.prompt.encode('utf-8'))}"

    @property
    def invocation_digest(self) -> str:
        return canonical_sha256(self.invocation.to_dict())

    @property
    def identity_material(self) -> dict[str, Any]:
        # request_id and credential_ref are deliberately excluded.  A retry
        # or credential rotation does not create a new proposer condition.
        return {
            "schema_version": 1,
            "proposer": self.proposer.to_dict(),
            "proposal_context_id": self.proposal_context_id,
            "parent_artifact_id": str(self.parent_artifact_id),
            "prompt_digest": self.prompt_digest,
            "timeout_sec": float(self.timeout_sec),
            "max_output_tokens": self.max_output_tokens,
            "invocation_digest": self.invocation_digest,
        }

    @property
    def condition_digest(self) -> str:
        return canonical_sha256(self.identity_material)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete inert request contract, including prompt text."""

        return {
            "schema_version": 1,
            "request_id": self.request_id,
            "proposer": self.proposer.to_dict(),
            "proposal_context_id": self.proposal_context_id,
            "parent_artifact_id": str(self.parent_artifact_id),
            "prompt": self.prompt,
            "credential_ref": str(self.credential_ref),
            "timeout_sec": float(self.timeout_sec),
            "max_output_tokens": self.max_output_tokens,
            "invocation": self.invocation.to_dict(),
            "condition_digest": self.condition_digest,
        }

    def provenance_dict(self) -> dict[str, Any]:
        """Return bounded audit metadata without raw prompt or secret content."""

        return {
            "schema_version": 1,
            "request_id": self.request_id,
            "proposer": self.proposer.to_dict(),
            "proposal_context_id": self.proposal_context_id,
            "parent_artifact_id": str(self.parent_artifact_id),
            "prompt_digest": self.prompt_digest,
            "credential_ref": str(self.credential_ref),
            "timeout_sec": float(self.timeout_sec),
            "max_output_tokens": self.max_output_tokens,
            "transport": self.invocation.transport,
            "invocation_digest": self.invocation_digest,
            "condition_digest": self.condition_digest,
        }


class TrustedProposerRegistry:
    """Sealed exact-match registry for models, harnesses and prompts."""

    def __init__(
        self, *, profile_registry: ProfileRegistry = BUILTIN_PROFILE_REGISTRY
    ) -> None:
        self._profile_registry = profile_registry
        self._models: dict[tuple[str, str], ModelProfile] = {}
        self._harnesses: dict[tuple[str, str], HarnessAdapter] = {}
        self._prompts: dict[tuple[str, str], PromptProtocol] = {}
        self._profiles: dict[ProfileRef, ResolvedProposerProfile] = {}
        self._sealed = False

    def _ensure_open(self) -> None:
        if self._sealed:
            raise RuntimeError("trusted proposer registry is sealed")

    def register_model(self, model: ModelProfile) -> None:
        self._ensure_open()
        if not isinstance(model, ModelProfile):
            raise TypeError("model must be a ModelProfile")
        key = (model.model_id, model.revision)
        if key in self._models:
            raise ValueError("model profile is already registered")
        self._models[key] = model

    def register_harness(self, harness: HarnessAdapter) -> None:
        self._ensure_open()
        if not isinstance(harness, HarnessAdapter):
            raise TypeError("harness must implement HarnessAdapter")
        _token(harness.harness_id, field="harness_id")
        _revision(harness.revision, field="harness revision")
        key = (harness.harness_id, harness.revision)
        if key in self._harnesses:
            raise ValueError("harness adapter is already registered")
        self._harnesses[key] = harness

    def register_prompt(self, protocol: PromptProtocol) -> None:
        self._ensure_open()
        if not isinstance(protocol, PromptProtocol):
            raise TypeError("protocol must be a PromptProtocol")
        key = (protocol.protocol_id, protocol.revision)
        if key in self._prompts:
            raise ValueError("prompt protocol is already registered")
        self._prompts[key] = protocol

    def bind_profile(self, definition: ProfileDefinition) -> None:
        """Bind one exact reviewed outer profile to trusted components."""

        self._ensure_open()
        if not isinstance(definition, ProfileDefinition):
            raise TypeError("definition must be a ProfileDefinition")
        definition = self._profile_registry.resolve(definition.ref)
        if definition.ref.kind != "proposer":
            raise ValueError("profile definition must have proposer kind")
        config = definition.config
        if set(config) != _PROPOSER_CONFIG_FIELDS:
            raise ValueError("proposer profile config fields are not recognized")
        values: dict[str, str] = {}
        for field in _PROPOSER_CONFIG_FIELDS:
            value = config[field]
            if not isinstance(value, str):
                raise ValueError(f"proposer profile {field} must be a string")
            values[field] = value
        resolved = ResolvedProposerProfile(
            profile=definition.ref,
            implementation_id=definition.implementation_id,
            model=self.resolve_model(
                values["model"], values["model_revision"]
            ),
            harness=self.resolve_harness(
                values["harness"], values["harness_revision"]
            ),
            prompt_protocol=self.resolve_prompt(
                values["prompt_protocol"],
                values["prompt_protocol_revision"],
            ),
        )
        if definition.ref in self._profiles:
            raise ValueError("proposer profile is already bound")
        self._profiles[definition.ref] = resolved

    def seal(self) -> None:
        self._sealed = True

    @property
    def sealed(self) -> bool:
        return self._sealed

    def resolve_model(self, model_id: str, revision: str) -> ModelProfile:
        model_id = _token(model_id, field="model_id")
        revision = _revision(revision, field="model revision")
        try:
            return self._models[(model_id, revision)]
        except KeyError as exc:
            raise ValueError(
                f"unknown trusted model {model_id!r}@{revision!r}"
            ) from exc

    def resolve_harness(self, harness_id: str, revision: str) -> HarnessAdapter:
        harness_id = _token(harness_id, field="harness_id")
        revision = _revision(revision, field="harness revision")
        try:
            return self._harnesses[(harness_id, revision)]
        except KeyError as exc:
            raise ValueError(
                f"unknown trusted harness {harness_id!r}@{revision!r}"
            ) from exc

    def resolve_prompt(self, protocol_id: str, revision: str) -> PromptProtocol:
        protocol_id = _token(protocol_id, field="prompt protocol_id")
        revision = _revision(revision, field="prompt protocol revision")
        try:
            return self._prompts[(protocol_id, revision)]
        except KeyError as exc:
            raise ValueError(
                f"unknown trusted prompt {protocol_id!r}@{revision!r}"
            ) from exc

    def resolve_profile(
        self, ref: ProfileRef | Mapping[str, Any]
    ) -> ResolvedProposerProfile:
        if not isinstance(ref, ProfileRef):
            ref = ProfileRef.from_value(ref)
        definition = self._profile_registry.resolve(ref)
        if definition.ref.kind != "proposer":
            raise ValueError("profile reference must have proposer kind")
        try:
            return self._profiles[definition.ref]
        except KeyError as exc:
            raise ValueError("proposer profile has no trusted binding") from exc

    def create_request(
        self,
        proposer: ProfileRef | Mapping[str, Any],
        *,
        proposal_context_id: str,
        parent_artifact_id: ArtifactId | str,
        prompt: str,
        credential_ref: CredentialRef | str,
        request_id: str | None = None,
        timeout_sec: float = MAX_PROPOSER_TIMEOUT_SEC,
        max_output_tokens: int | None = None,
    ) -> ProposerRequestContract:
        resolved = self.resolve_profile(proposer)
        cap = (
            resolved.model.request_output_token_cap
            if max_output_tokens is None
            else max_output_tokens
        )
        invocation = resolved.harness.build_invocation(
            model=resolved.model,
            prompt_protocol=resolved.prompt_protocol,
            prompt=prompt,
            max_output_tokens=cap,
        )
        return ProposerRequestContract(
            schema_version=1,
            request_id=str(uuid.uuid4()) if request_id is None else request_id,
            proposer=resolved,
            proposal_context_id=proposal_context_id,
            parent_artifact_id=ArtifactId.parse(parent_artifact_id),
            prompt=prompt,
            credential_ref=CredentialRef.parse(credential_ref),
            timeout_sec=timeout_sec,
            max_output_tokens=cap,
            invocation=invocation,
        )


def _build_builtin_registry() -> TrustedProposerRegistry:
    registry = TrustedProposerRegistry()
    for model in (
        ModelProfile(
            model_id="deepseek/deepseek-v4-pro",
            revision="v1",
            provider_id="deepseek",
            provider_model_id="deepseek-v4-pro",
            display_name="DeepSeek V4 Pro",
            context_tokens=1_000_000,
            output_tokens=384_000,
            request_output_token_cap=384_000,
            reasoning_effort="max",
        ),
        ModelProfile(
            model_id="deepseek/deepseek-v4-flash",
            revision="v1",
            provider_id="deepseek",
            provider_model_id="deepseek-v4-flash",
            display_name="DeepSeek V4 Flash",
            context_tokens=1_000_000,
            output_tokens=384_000,
            request_output_token_cap=384_000,
            reasoning_effort="max",
        ),
    ):
        registry.register_model(model)
    for harness in (
        DirectAPIHarnessAdapter(),
        OpenCodeHarnessAdapter(),
        PiHarnessAdapter(),
    ):
        registry.register_harness(harness)
    for protocol in (
        PromptProtocol("proposal-v1", "v1", 1),
        PromptProtocol("proposal-v2", "v1", 2),
    ):
        registry.register_prompt(protocol)
    for profile_id in BUILTIN_PROPOSER_PROFILE_IDS:
        registry.bind_profile(
            BUILTIN_PROFILE_REGISTRY.get(
                kind="proposer", profile_id=profile_id, revision="v1"
            )
        )
    registry.seal()
    return registry


BUILTIN_PROPOSER_REGISTRY = _build_builtin_registry()


def builtin_proposer_registry() -> TrustedProposerRegistry:
    return BUILTIN_PROPOSER_REGISTRY


__all__ = [
    "BUILTIN_PROPOSER_REGISTRY",
    "BUILTIN_PROPOSER_PROFILE_IDS",
    "MAX_PROPOSER_PROMPT_BYTES",
    "CredentialRef",
    "DirectAPIHarnessAdapter",
    "HarnessAdapter",
    "HarnessInvocation",
    "ModelProfile",
    "OpenCodeHarnessAdapter",
    "PiHarnessAdapter",
    "PromptProtocol",
    "ProposerRequestContract",
    "ResolvedProposerProfile",
    "TrustedProposerRegistry",
    "builtin_proposer_registry",
]
