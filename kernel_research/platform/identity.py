"""Baseline and per-evaluation identities for the V2 platform."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import re
from typing import Any
import uuid

from .artifacts import ArtifactId
from .canonical import canonical_json_text, canonical_sha256, require_sha256_digest
from .profiles import ProfileRef, ResearchNamespace


BASELINE_SOURCES = frozenset({"deployment", "campaign"})
RUN_MODES = frozenset({"BENCHMARK", "DISCOVERY"})
EXECUTION_ENVIRONMENT_STATUSES = frozenset({"RESOLVED", "LEGACY_UNKNOWN"})
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_PINNED_IMAGE_RE = re.compile(
    r"^(?:[^\s@]+@)?sha256:([0-9a-f]{64})$"
)


def _name(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise ValueError(f"{field} must be a stable non-empty identifier")
    return value


def _uuid(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("experiment_uid must be a canonical UUID")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError("experiment_uid must be a canonical UUID") from exc
    if str(parsed) != value:
        raise ValueError("experiment_uid must be a canonical lowercase UUID")
    return value


def _history_cutoff(value: object) -> str | int | None:
    if value is None:
        return None
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, str) and 0 < len(value) <= 256:
        return value
    raise ValueError("history_cutoff must be null, a non-negative integer or string")


def _plain_json(value: Any) -> Any:
    """Return a detached strict-JSON value using the canonical encoder."""

    # Round-tripping also rejects NaN, non-string mapping keys and objects whose
    # representation is not part of the platform's stable JSON contract.
    return json.loads(canonical_json_text(value))


def _binding_digest(value: Any, *, field: str) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field} binding must be a non-empty stable JSON value")
    if isinstance(value, str):
        if not value or len(value) > 4096 or "\x00" in value:
            raise ValueError(f"{field} binding must be a non-empty stable JSON value")
        if value.startswith("sha256:"):
            return require_sha256_digest(value, field=f"{field} binding")
    elif not isinstance(value, Mapping):
        raise ValueError(f"{field} binding must be a string or JSON object")
    normalized = _plain_json(value)
    if isinstance(normalized, dict) and not normalized:
        raise ValueError(f"{field} binding must not be empty")
    return canonical_sha256(
        {"schema_version": 1, "field": field, "binding": normalized}
    )


@dataclass(frozen=True, slots=True)
class ExecutionEnvironmentDigest:
    """Digest of the environment that can change scientific comparability.

    A resolved value binds all five execution dimensions. ``LEGACY_UNKNOWN`` is
    deliberately not comparable, even with another value carrying the same
    digest.  Its digest is only a stable audit scope for migrated records.
    """

    status: str
    digest: str
    evaluator_image_digest: str | None = None
    toolchain_digest: str | None = None
    framework_digest: str | None = None
    operator_abi_digest: str | None = None
    build_flags_digest: str | None = None

    def __post_init__(self) -> None:
        if self.status not in EXECUTION_ENVIRONMENT_STATUSES:
            allowed = ", ".join(sorted(EXECUTION_ENVIRONMENT_STATUSES))
            raise ValueError(
                f"execution environment status must be one of: {allowed}"
            )
        require_sha256_digest(self.digest, field="execution environment digest")
        components = {
            "evaluator_image_digest": self.evaluator_image_digest,
            "toolchain_digest": self.toolchain_digest,
            "framework_digest": self.framework_digest,
            "operator_abi_digest": self.operator_abi_digest,
            "build_flags_digest": self.build_flags_digest,
        }
        if self.status == "LEGACY_UNKNOWN":
            if any(value is not None for value in components.values()):
                raise ValueError(
                    "LEGACY_UNKNOWN execution environment cannot claim components"
                )
            return
        if any(value is None for value in components.values()):
            raise ValueError(
                "RESOLVED execution environment requires every component digest"
            )
        normalized = {
            name: require_sha256_digest(value, field=name)
            for name, value in components.items()
        }
        expected = canonical_sha256(
            {
                "schema_version": 1,
                "status": "RESOLVED",
                "components": normalized,
            }
        )
        if self.digest != expected:
            raise ValueError(
                "execution environment digest does not match its components"
            )

    @property
    def is_resolved(self) -> bool:
        return self.status == "RESOLVED"

    @classmethod
    def resolved(
        cls,
        *,
        evaluator_image_digest: str,
        toolchain_digest: str,
        framework_digest: str,
        operator_abi_digest: str,
        build_flags_digest: str,
    ) -> "ExecutionEnvironmentDigest":
        components = {
            "evaluator_image_digest": require_sha256_digest(
                evaluator_image_digest, field="evaluator_image_digest"
            ),
            "toolchain_digest": require_sha256_digest(
                toolchain_digest, field="toolchain_digest"
            ),
            "framework_digest": require_sha256_digest(
                framework_digest, field="framework_digest"
            ),
            "operator_abi_digest": require_sha256_digest(
                operator_abi_digest, field="operator_abi_digest"
            ),
            "build_flags_digest": require_sha256_digest(
                build_flags_digest, field="build_flags_digest"
            ),
        }
        return cls(
            status="RESOLVED",
            digest=canonical_sha256(
                {
                    "schema_version": 1,
                    "status": "RESOLVED",
                    "components": components,
                }
            ),
            **components,
        )

    @classmethod
    def from_bindings(
        cls,
        *,
        evaluator_image: str,
        toolchain: str | Mapping[str, Any],
        framework: str | Mapping[str, Any],
        operator_abi: str | Mapping[str, Any],
        build_flags: Sequence[str],
    ) -> "ExecutionEnvironmentDigest":
        if not isinstance(evaluator_image, str):
            raise ValueError("evaluator_image must be a pinned image digest")
        image_match = _PINNED_IMAGE_RE.fullmatch(evaluator_image)
        if image_match is None:
            raise ValueError(
                "evaluator_image must be pinned by an exact sha256 digest"
            )
        if isinstance(build_flags, (str, bytes)) or not isinstance(
            build_flags, Sequence
        ):
            raise ValueError("build_flags must be an ordered sequence of strings")
        flags = tuple(build_flags)
        if len(flags) > 128 or any(
            not isinstance(flag, str)
            or len(flag) > 1024
            or "\x00" in flag
            for flag in flags
        ):
            raise ValueError("build_flags contain an invalid or oversized flag")
        return cls.resolved(
            evaluator_image_digest="sha256:" + image_match.group(1),
            toolchain_digest=_binding_digest(toolchain, field="toolchain"),
            framework_digest=_binding_digest(framework, field="framework"),
            operator_abi_digest=_binding_digest(
                operator_abi, field="operator_abi"
            ),
            build_flags_digest=canonical_sha256(
                {
                    "schema_version": 1,
                    "field": "build_flags",
                    "binding": list(flags),
                }
            ),
        )

    @classmethod
    def legacy_unknown(cls, *, scope: Any) -> "ExecutionEnvironmentDigest":
        return cls(
            status="LEGACY_UNKNOWN",
            digest=canonical_sha256(
                {
                    "schema_version": 1,
                    "status": "LEGACY_UNKNOWN",
                    "scope": _plain_json(scope),
                }
            ),
        )

    def require_match(
        self, other: "ExecutionEnvironmentDigest", *, context: str
    ) -> None:
        if not isinstance(other, ExecutionEnvironmentDigest):
            raise ValueError(f"{context} is missing an execution environment")
        if not self.is_resolved or not other.is_resolved:
            raise ValueError(
                f"{context} uses LEGACY_UNKNOWN execution evidence and is not "
                "scientifically comparable"
            )
        if self != other:
            raise ValueError(f"{context} execution environment mismatch")

    def to_dict(self) -> dict[str, Any]:
        components = None
        if self.is_resolved:
            components = {
                "evaluator_image_digest": self.evaluator_image_digest,
                "toolchain_digest": self.toolchain_digest,
                "framework_digest": self.framework_digest,
                "operator_abi_digest": self.operator_abi_digest,
                "build_flags_digest": self.build_flags_digest,
            }
        return {
            "schema_version": 1,
            "status": self.status,
            "digest": self.digest,
            "components": components,
        }

    @classmethod
    def from_value(cls, value: object) -> "ExecutionEnvironmentDigest":
        fields = {"schema_version", "status", "digest", "components"}
        if not isinstance(value, dict) or set(value) != fields:
            raise ValueError(
                "execution environment fields do not match the V1 contract"
            )
        if type(value["schema_version"]) is not int or value["schema_version"] != 1:
            raise ValueError("execution environment schema_version must be 1")
        components = value["components"]
        if value["status"] == "LEGACY_UNKNOWN":
            if components is not None:
                raise ValueError(
                    "LEGACY_UNKNOWN execution environment cannot claim components"
                )
            return cls(status=value["status"], digest=value["digest"])
        component_fields = {
            "evaluator_image_digest",
            "toolchain_digest",
            "framework_digest",
            "operator_abi_digest",
            "build_flags_digest",
        }
        if not isinstance(components, dict) or set(components) != component_fields:
            raise ValueError(
                "resolved execution environment components are incomplete"
            )
        return cls(
            status=value["status"],
            digest=value["digest"],
            **components,
        )


@dataclass(frozen=True)
class BaselineRef:
    """One baseline artifact frozen in one namespace and owner domain."""

    namespace_id: str
    artifact_id: ArtifactId
    source: str
    revision: str
    execution_environment: ExecutionEnvironmentDigest | None = None

    def __post_init__(self) -> None:
        require_sha256_digest(self.namespace_id, field="baseline namespace_id")
        if not isinstance(self.artifact_id, ArtifactId):
            raise ValueError("baseline artifact_id must be an ArtifactId")
        if self.source not in BASELINE_SOURCES:
            allowed = ", ".join(sorted(BASELINE_SOURCES))
            raise ValueError(f"baseline source must be one of: {allowed}")
        _name(self.revision, field="baseline revision")
        if self.execution_environment is None:
            object.__setattr__(
                self,
                "execution_environment",
                ExecutionEnvironmentDigest.legacy_unknown(
                    scope={
                        "namespace_id": self.namespace_id,
                        "artifact_id": str(self.artifact_id),
                        "source": self.source,
                        "revision": self.revision,
                    }
                ),
            )
        elif not isinstance(
            self.execution_environment, ExecutionEnvironmentDigest
        ):
            raise ValueError(
                "baseline execution_environment must be an "
                "ExecutionEnvironmentDigest"
            )

    @classmethod
    def create(
        cls,
        *,
        namespace: ResearchNamespace | str,
        artifact_id: ArtifactId | str,
        source: str,
        revision: str,
        execution_environment: ExecutionEnvironmentDigest | None = None,
    ) -> "BaselineRef":
        namespace_id = (
            namespace.namespace_id
            if isinstance(namespace, ResearchNamespace)
            else require_sha256_digest(namespace, field="baseline namespace_id")
        )
        parsed_artifact = ArtifactId.parse(artifact_id)
        environment = execution_environment
        if environment is None:
            environment = ExecutionEnvironmentDigest.legacy_unknown(
                scope={
                    "namespace_id": namespace_id,
                    "artifact_id": str(parsed_artifact),
                    "source": source,
                    "revision": revision,
                }
            )
        return cls(
            namespace_id=namespace_id,
            artifact_id=parsed_artifact,
            source=source,
            revision=revision,
            execution_environment=environment,
        )

    @classmethod
    def from_value(cls, value: object) -> "BaselineRef":
        v1_fields = {
            "schema_version",
            "namespace_id",
            "artifact_id",
            "source",
            "revision",
        }
        if not isinstance(value, dict):
            raise ValueError("baseline reference must be a JSON object")
        version = value.get("schema_version")
        fields = (
            v1_fields
            if version == 1
            else v1_fields | {"execution_environment"}
        )
        unknown = sorted(set(value) - fields)
        missing = sorted(fields - set(value))
        if unknown:
            raise ValueError(
                f"baseline reference contains unknown fields: {', '.join(unknown)}"
            )
        if missing:
            raise ValueError(
                f"baseline reference is missing fields: {', '.join(missing)}"
            )
        if type(version) is not int or version not in {1, 2}:
            raise ValueError("baseline reference schema_version must be 1 or 2")
        environment = None
        if version == 2:
            environment = ExecutionEnvironmentDigest.from_value(
                value["execution_environment"]
            )
            if not environment.is_resolved:
                raise ValueError(
                    "V2 baseline requires a resolved execution environment"
                )
        return cls.create(
            namespace=value["namespace_id"],
            artifact_id=value["artifact_id"],
            source=value["source"],
            revision=value["revision"],
            execution_environment=environment,
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": 1,
            "namespace_id": self.namespace_id,
            "artifact_id": str(self.artifact_id),
            "source": self.source,
            "revision": self.revision,
        }
        if self.execution_environment.is_resolved:
            value["schema_version"] = 2
            value["execution_environment"] = self.execution_environment.to_dict()
        return value

    @property
    def is_scientifically_comparable(self) -> bool:
        return self.execution_environment.is_resolved

    def require_environment(
        self, execution_environment: ExecutionEnvironmentDigest
    ) -> None:
        self.execution_environment.require_match(
            execution_environment, context="baseline"
        )


@dataclass(frozen=True)
class ExperimentIdentity:
    """Immutable scientific conditions plus one host-generated execution UID.

    ``condition_digest`` intentionally excludes orchestration coordinates and
    the UID.  Two executions can therefore be recognized as the same
    scientific condition without conflating their durable records.
    """

    experiment_uid: str
    namespace: ResearchNamespace
    mode: str
    candidate_artifact_id: ArtifactId
    parent_artifact_id: ArtifactId
    baseline: BaselineRef
    stage: str
    suite: str
    replicate_kind: str
    replicate_index: int
    proposer_profile: ProfileRef | None
    prompt_digest: str | None
    feedback_digest: str | None
    cohort_id: str | None
    history_cutoff: str | int | None
    campaign_id: str | None
    run_id: str
    iteration: int
    execution_environment: ExecutionEnvironmentDigest | None = None

    def __post_init__(self) -> None:
        _uuid(self.experiment_uid)
        if not isinstance(self.namespace, ResearchNamespace):
            raise ValueError("namespace must be a ResearchNamespace")
        if self.mode not in RUN_MODES:
            allowed = ", ".join(sorted(RUN_MODES))
            raise ValueError(f"mode must be one of: {allowed}")
        if not isinstance(self.candidate_artifact_id, ArtifactId):
            raise ValueError("candidate_artifact_id must be an ArtifactId")
        if not isinstance(self.parent_artifact_id, ArtifactId):
            raise ValueError("parent_artifact_id must be an ArtifactId")
        if not isinstance(self.baseline, BaselineRef):
            raise ValueError("baseline must be a BaselineRef")
        if self.baseline.namespace_id != self.namespace.namespace_id:
            raise ValueError("baseline belongs to a different research namespace")
        if self.execution_environment is None:
            object.__setattr__(
                self,
                "execution_environment",
                self.baseline.execution_environment,
            )
        elif not isinstance(
            self.execution_environment, ExecutionEnvironmentDigest
        ):
            raise ValueError(
                "execution_environment must be an ExecutionEnvironmentDigest"
            )
        baseline_environment = self.baseline.execution_environment
        if baseline_environment.is_resolved or self.execution_environment.is_resolved:
            baseline_environment.require_match(
                self.execution_environment, context="experiment baseline"
            )
        elif baseline_environment != self.execution_environment:
            raise ValueError(
                "legacy experiment and baseline environment scopes differ"
            )
        _name(self.stage, field="stage")
        _name(self.suite, field="suite")
        _name(self.replicate_kind, field="replicate_kind")
        if type(self.replicate_index) is not int or self.replicate_index < 0:
            raise ValueError("replicate_index must be a non-negative integer")
        if self.proposer_profile is not None and (
            not isinstance(self.proposer_profile, ProfileRef)
            or self.proposer_profile.kind != "proposer"
        ):
            raise ValueError("proposer_profile must be a proposer ProfileRef")
        for field in ("prompt_digest", "feedback_digest"):
            value = getattr(self, field)
            if value is not None:
                require_sha256_digest(value, field=field)
        if self.cohort_id is not None:
            _name(self.cohort_id, field="cohort_id")
        object.__setattr__(
            self, "history_cutoff", _history_cutoff(self.history_cutoff)
        )
        if self.mode == "BENCHMARK" and (
            self.cohort_id is None or self.history_cutoff is None
        ):
            raise ValueError(
                "BENCHMARK experiments require cohort_id and history_cutoff"
            )
        if self.campaign_id is not None:
            _name(self.campaign_id, field="campaign_id")
        _name(self.run_id, field="run_id")
        if type(self.iteration) is not int or self.iteration < 0:
            raise ValueError("iteration must be a non-negative integer")

    @classmethod
    def create(
        cls,
        *,
        namespace: ResearchNamespace,
        mode: str,
        candidate_artifact_id: ArtifactId | str,
        parent_artifact_id: ArtifactId | str,
        baseline: BaselineRef,
        execution_environment: ExecutionEnvironmentDigest | None = None,
        stage: str,
        suite: str,
        run_id: str,
        iteration: int,
        replicate_kind: str = "primary",
        replicate_index: int = 0,
        proposer_profile: ProfileRef | None = None,
        prompt_digest: str | None = None,
        feedback_digest: str | None = None,
        cohort_id: str | None = None,
        history_cutoff: str | int | None = None,
        campaign_id: str | None = None,
        experiment_uid: str | None = None,
    ) -> "ExperimentIdentity":
        environment = (
            baseline.execution_environment
            if execution_environment is None
            else execution_environment
        )
        return cls(
            experiment_uid=(
                str(uuid.uuid4())
                if experiment_uid is None
                else experiment_uid
            ),
            namespace=namespace,
            mode=mode,
            candidate_artifact_id=ArtifactId.parse(candidate_artifact_id),
            parent_artifact_id=ArtifactId.parse(parent_artifact_id),
            baseline=baseline,
            execution_environment=environment,
            stage=stage,
            suite=suite,
            replicate_kind=replicate_kind,
            replicate_index=replicate_index,
            proposer_profile=proposer_profile,
            prompt_digest=prompt_digest,
            feedback_digest=feedback_digest,
            cohort_id=cohort_id,
            history_cutoff=history_cutoff,
            campaign_id=campaign_id,
            run_id=run_id,
            iteration=iteration,
        )

    @property
    def namespace_id(self) -> str:
        return self.namespace.namespace_id

    @property
    def evaluation_protocol(self) -> ProfileRef:
        return self.namespace.evaluation_protocol

    @property
    def promotion_policy(self) -> ProfileRef:
        return self.namespace.promotion_policy

    def condition_dict(self) -> dict[str, Any]:
        schema_version = 2 if self.execution_environment.is_resolved else 1
        value = {
            "schema_version": schema_version,
            "namespace_id": self.namespace_id,
            "mode": self.mode,
            "cohort_id": self.cohort_id,
            "history_cutoff": self.history_cutoff,
            "candidate_artifact_id": str(self.candidate_artifact_id),
            "parent_artifact_id": str(self.parent_artifact_id),
            "baseline": self.baseline.to_dict(),
            "stage": self.stage,
            "suite": self.suite,
            "replicate_kind": self.replicate_kind,
            "replicate_index": self.replicate_index,
            "proposer_profile": (
                None
                if self.proposer_profile is None
                else self.proposer_profile.to_dict()
            ),
            "prompt_digest": self.prompt_digest,
            "feedback_digest": self.feedback_digest,
        }
        if schema_version == 2:
            value["execution_environment"] = self.execution_environment.to_dict()
        return value

    @property
    def condition_digest(self) -> str:
        return canonical_sha256(self.condition_dict())

    def to_dict(self) -> dict[str, Any]:
        schema_version = 2 if self.execution_environment.is_resolved else 1
        return {
            "schema_version": schema_version,
            "experiment_uid": self.experiment_uid,
            "condition_digest": self.condition_digest,
            "namespace": self.namespace.to_dict(),
            **{
                key: value
                for key, value in self.condition_dict().items()
                if key not in {"schema_version", "namespace_id"}
            },
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "iteration": self.iteration,
        }

    @classmethod
    def from_value(cls, value: object) -> "ExperimentIdentity":
        v1_fields = {
            "schema_version",
            "experiment_uid",
            "condition_digest",
            "namespace",
            "mode",
            "cohort_id",
            "history_cutoff",
            "candidate_artifact_id",
            "parent_artifact_id",
            "baseline",
            "stage",
            "suite",
            "replicate_kind",
            "replicate_index",
            "proposer_profile",
            "prompt_digest",
            "feedback_digest",
            "campaign_id",
            "run_id",
            "iteration",
        }
        if not isinstance(value, dict):
            raise ValueError("experiment identity must be a JSON object")
        version = value.get("schema_version")
        fields = (
            v1_fields
            if version == 1
            else v1_fields | {"execution_environment"}
        )
        unknown = sorted(set(value) - fields)
        missing = sorted(fields - set(value))
        if unknown:
            raise ValueError(
                f"experiment identity contains unknown fields: {', '.join(unknown)}"
            )
        if missing:
            raise ValueError(
                f"experiment identity is missing fields: {', '.join(missing)}"
            )
        if type(version) is not int or version not in {1, 2}:
            raise ValueError("experiment identity schema_version must be 1 or 2")
        proposer_value = value["proposer_profile"]
        baseline = BaselineRef.from_value(value["baseline"])
        if version == 1:
            if baseline.execution_environment.is_resolved:
                raise ValueError(
                    "V1 experiment cannot claim a resolved baseline environment"
                )
            execution_environment = baseline.execution_environment
        else:
            execution_environment = ExecutionEnvironmentDigest.from_value(
                value["execution_environment"]
            )
            if not execution_environment.is_resolved:
                raise ValueError(
                    "V2 experiment requires a resolved execution environment"
                )
        identity = cls.create(
            experiment_uid=value["experiment_uid"],
            namespace=ResearchNamespace.from_value(value["namespace"]),
            mode=value["mode"],
            candidate_artifact_id=value["candidate_artifact_id"],
            parent_artifact_id=value["parent_artifact_id"],
            baseline=baseline,
            execution_environment=execution_environment,
            stage=value["stage"],
            suite=value["suite"],
            replicate_kind=value["replicate_kind"],
            replicate_index=value["replicate_index"],
            proposer_profile=(
                None
                if proposer_value is None
                else ProfileRef.from_value(proposer_value)
            ),
            prompt_digest=value["prompt_digest"],
            feedback_digest=value["feedback_digest"],
            cohort_id=value["cohort_id"],
            history_cutoff=value["history_cutoff"],
            campaign_id=value["campaign_id"],
            run_id=value["run_id"],
            iteration=value["iteration"],
        )
        require_sha256_digest(value["condition_digest"], field="condition_digest")
        if value["condition_digest"] != identity.condition_digest:
            raise ValueError("condition_digest does not match experiment identity")
        return identity

    @property
    def is_scientifically_comparable(self) -> bool:
        return self.execution_environment.is_resolved


__all__ = [
    "BASELINE_SOURCES",
    "EXECUTION_ENVIRONMENT_STATUSES",
    "RUN_MODES",
    "BaselineRef",
    "ExecutionEnvironmentDigest",
    "ExperimentIdentity",
]
