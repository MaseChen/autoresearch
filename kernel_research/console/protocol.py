"""Strict V1 wire contracts shared by the Console gateway and remote agent."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Any, Mapping

from ..platform.canonical import canonical_json_bytes, canonical_sha256
from ..platform.proposal import CandidateBundle, TRITON_PYTHON_BUNDLE_LIMITS


CONSOLE_PROTOCOL_VERSION = 1
MAX_AGENT_LINE_BYTES = 1024 * 1024
MAX_AGENT_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_CURSOR_VALUE = 2**63 - 1
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,255}$")
OPERATION_KINDS = frozenset(
    {
        "RUN_START",
        "RUN_STOP",
        "RUN_RESUME",
        "MANUAL_EVALUATION_START",
        "CAMPAIGN_CREATE",
        "CAMPAIGN_START",
        "CAMPAIGN_PAUSE",
        "CAMPAIGN_RESUME",
        "CAMPAIGN_CHILD_EXECUTE",
        "CAMPAIGN_LINEAGE_ADVANCE",
        "BENCHMARK_INIT",
        "BENCHMARK_EXECUTE",
    }
)
TASK_KINDS = frozenset(
    {"AUTONOMOUS_RUN", "DISCOVERY_CAMPAIGN", "BENCHMARK_CAMPAIGN", "MANUAL_EVALUATION"}
)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"console JSON repeats key {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"console JSON contains invalid constant {value}")


def strict_json_loads(raw: bytes | str, *, max_bytes: int = MAX_AGENT_LINE_BYTES) -> Any:
    """Decode one bounded UTF-8 JSON value with duplicate/NaN rejection."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if isinstance(raw, str):
        try:
            encoded = raw.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("console JSON is not valid UTF-8") from exc
    elif isinstance(raw, bytes):
        encoded = raw
    else:
        raise TypeError("console JSON must be bytes or text")
    if not encoded or len(encoded) > max_bytes:
        raise ValueError("console JSON is empty or exceeds its size bound")
    if b"\x00" in encoded:
        raise ValueError("console JSON contains NUL")
    try:
        text = encoded.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"console JSON is invalid: {exc}") from exc


def _strict_object(
    value: object,
    *,
    name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    unknown = sorted(set(value) - required - optional)
    missing = sorted(required - set(value))
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {', '.join(unknown)}")
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}")
    return value


def _bounded_text(value: object, field_name: str, *, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > maximum
        or "\x00" in value
    ):
        raise ValueError(f"{field_name} must be bounded non-empty UTF-8 text")
    return value


def _identifier(value: object, field_name: str) -> str:
    selected = _bounded_text(value, field_name, maximum=256)
    if not _IDENTIFIER_RE.fullmatch(selected):
        raise ValueError(f"{field_name} contains unsupported characters")
    return selected


def _digest(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a tagged sha256 digest")
    return value


def _non_negative_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 0 or value > MAX_CURSOR_VALUE:
        raise ValueError(f"{field_name} must be a bounded non-negative integer")
    return value


def _canonical_uuid(value: object, field_name: str) -> str:
    selected = _bounded_text(value, field_name, maximum=36)
    try:
        import uuid

        parsed = uuid.UUID(selected)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be a canonical UUID") from exc
    if str(parsed) != selected:
        raise ValueError(f"{field_name} must be a canonical lowercase UUID")
    return selected


@dataclass(frozen=True)
class DraftV1:
    draft_id: str
    task_kind: str
    title: str
    values: Mapping[str, Any]
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        _canonical_uuid(self.draft_id, "draft_id")
        if self.task_kind not in TASK_KINDS:
            raise ValueError("draft task_kind is not supported")
        _bounded_text(self.title, "draft title", maximum=256)
        if not isinstance(self.values, Mapping):
            raise ValueError("draft values must be an object")
        _bounded_text(self.created_at, "draft created_at", maximum=64)
        _bounded_text(self.updated_at, "draft updated_at", maximum=64)

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": CONSOLE_PROTOCOL_VERSION,
            "draft_id": self.draft_id,
            "task_kind": self.task_kind,
            "title": self.title,
            "values": dict(self.values),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if include_digest:
            result["draft_digest"] = self.digest
        return result


@dataclass(frozen=True)
class ManualEvaluationRequestV1:
    operation_id: str
    candidate: CandidateBundle
    runtime_identity_digest: str

    def __post_init__(self) -> None:
        _canonical_uuid(self.operation_id, "operation_id")
        if not isinstance(self.candidate, CandidateBundle):
            raise ValueError("manual candidate must be a CandidateBundle")
        self.candidate.validate(TRITON_PYTHON_BUNDLE_LIMITS)
        _digest(self.runtime_identity_digest, "runtime_identity_digest")

    @property
    def operation_digest(self) -> str:
        return canonical_sha256(self.to_dict(include_content=True))

    def to_dict(self, *, include_content: bool = True) -> dict[str, Any]:
        return {
            "schema_version": CONSOLE_PROTOCOL_VERSION,
            "operation_id": self.operation_id,
            "candidate": self.candidate.to_dict(include_content=include_content),
            "runtime_identity_digest": self.runtime_identity_digest,
        }

    @classmethod
    def from_value(cls, value: object) -> "ManualEvaluationRequestV1":
        fields = frozenset(
            {"schema_version", "operation_id", "candidate", "runtime_identity_digest"}
        )
        obj = _strict_object(value, name="manual evaluation request", required=fields)
        if obj["schema_version"] != 1 or type(obj["schema_version"]) is not int:
            raise ValueError("manual evaluation schema_version must be 1")
        return cls(
            operation_id=obj["operation_id"],
            candidate=CandidateBundle.from_value(
                obj["candidate"], limits=TRITON_PYTHON_BUNDLE_LIMITS
            ),
            runtime_identity_digest=obj["runtime_identity_digest"],
        )


@dataclass(frozen=True)
class PreparedOperationV1:
    operation_id: str
    kind: str
    operation_digest: str
    runtime_identity_digest: str
    prepared_at: str
    expires_epoch: float
    confirmation_phrase: str
    impact: Mapping[str, Any]

    def __post_init__(self) -> None:
        _canonical_uuid(self.operation_id, "operation_id")
        if self.kind not in OPERATION_KINDS:
            raise ValueError("prepared operation kind is not allowlisted")
        _digest(self.operation_digest, "operation_digest")
        _digest(self.runtime_identity_digest, "runtime_identity_digest")
        _bounded_text(self.prepared_at, "prepared_at", maximum=64)
        if (
            not isinstance(self.expires_epoch, (int, float))
            or isinstance(self.expires_epoch, bool)
            or not math.isfinite(float(self.expires_epoch))
            or float(self.expires_epoch) <= 0
        ):
            raise ValueError("expires_epoch must be finite and positive")
        _bounded_text(self.confirmation_phrase, "confirmation_phrase", maximum=128)
        if not isinstance(self.impact, Mapping):
            raise ValueError("operation impact must be an object")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "operation_id": self.operation_id,
            "kind": self.kind,
            "operation_digest": self.operation_digest,
            "runtime_identity_digest": self.runtime_identity_digest,
            "prepared_at": self.prepared_at,
            "expires_epoch": float(self.expires_epoch),
            "confirmation_phrase": self.confirmation_phrase,
            "impact": dict(self.impact),
        }

    @classmethod
    def from_value(cls, value: object) -> "PreparedOperationV1":
        fields = frozenset(
            {
                "schema_version",
                "operation_id",
                "kind",
                "operation_digest",
                "runtime_identity_digest",
                "prepared_at",
                "expires_epoch",
                "confirmation_phrase",
                "impact",
            }
        )
        obj = _strict_object(value, name="prepared operation", required=fields)
        if obj["schema_version"] != 1 or type(obj["schema_version"]) is not int:
            raise ValueError("prepared operation schema_version must be 1")
        return cls(**{key: obj[key] for key in fields if key != "schema_version"})


@dataclass(frozen=True)
class OperationReceiptV1:
    operation_id: str
    kind: str
    operation_digest: str
    status: str
    observed_at: str
    domain_identity: Mapping[str, Any]
    result: Mapping[str, Any] | None = None
    problem: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _canonical_uuid(self.operation_id, "operation_id")
        if self.kind not in OPERATION_KINDS:
            raise ValueError("receipt operation kind is not allowlisted")
        _digest(self.operation_digest, "operation_digest")
        if self.status not in {
            "PREPARED",
            "EXECUTING",
            "SUCCEEDED",
            "FAILED",
            "UNKNOWN_OUTCOME",
            "EXPIRED",
        }:
            raise ValueError("operation receipt status is invalid")
        _bounded_text(self.observed_at, "receipt observed_at", maximum=64)
        if not isinstance(self.domain_identity, Mapping):
            raise ValueError("receipt domain_identity must be an object")
        if self.result is not None and not isinstance(self.result, Mapping):
            raise ValueError("receipt result must be an object")
        if self.problem is not None and not isinstance(self.problem, Mapping):
            raise ValueError("receipt problem must be an object")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "operation_id": self.operation_id,
            "kind": self.kind,
            "operation_digest": self.operation_digest,
            "status": self.status,
            "observed_at": self.observed_at,
            "domain_identity": dict(self.domain_identity),
            "result": None if self.result is None else dict(self.result),
            "problem": None if self.problem is None else dict(self.problem),
        }

    @classmethod
    def from_value(cls, value: object) -> "OperationReceiptV1":
        fields = frozenset(
            {
                "schema_version",
                "operation_id",
                "kind",
                "operation_digest",
                "status",
                "observed_at",
                "domain_identity",
                "result",
                "problem",
            }
        )
        obj = _strict_object(value, name="operation receipt", required=fields)
        if obj["schema_version"] != 1 or type(obj["schema_version"]) is not int:
            raise ValueError("operation receipt schema_version must be 1")
        return cls(**{key: obj[key] for key in fields if key != "schema_version"})


@dataclass(frozen=True)
class CompositeEventCursorV1:
    controller_event_id: int = 0
    controller_attempt_id: int = 0
    history_experiment_id: int = 0
    history_relation_id: int = 0
    campaign_outbox_id: int = 0
    campaign_child_id: int = 0
    campaign_lease_epoch: int = 0
    soak_generation_id: int = 0
    soak_violation_id: int = 0
    runtime_identity_digest: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "controller_event_id",
            "controller_attempt_id",
            "history_experiment_id",
            "history_relation_id",
            "campaign_outbox_id",
            "campaign_child_id",
            "campaign_lease_epoch",
            "soak_generation_id",
            "soak_violation_id",
        ):
            _non_negative_int(getattr(self, name), name)
        if self.runtime_identity_digest is not None:
            _digest(self.runtime_identity_digest, "runtime_identity_digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONSOLE_PROTOCOL_VERSION,
            "controller_event_id": self.controller_event_id,
            "controller_attempt_id": self.controller_attempt_id,
            "history_experiment_id": self.history_experiment_id,
            "history_relation_id": self.history_relation_id,
            "campaign_outbox_id": self.campaign_outbox_id,
            "campaign_child_id": self.campaign_child_id,
            "campaign_lease_epoch": self.campaign_lease_epoch,
            "soak_generation_id": self.soak_generation_id,
            "soak_violation_id": self.soak_violation_id,
            "runtime_identity_digest": self.runtime_identity_digest,
        }

    @classmethod
    def from_value(cls, value: object) -> "CompositeEventCursorV1":
        fields = frozenset(
            {
                "schema_version",
                "controller_event_id",
                "controller_attempt_id",
                "history_experiment_id",
                "history_relation_id",
                "campaign_outbox_id",
                "campaign_child_id",
                "campaign_lease_epoch",
                "soak_generation_id",
                "soak_violation_id",
                "runtime_identity_digest",
            }
        )
        obj = _strict_object(value, name="event cursor", required=fields)
        if obj["schema_version"] != CONSOLE_PROTOCOL_VERSION or type(
            obj["schema_version"]
        ) is not int:
            raise ValueError("event cursor schema_version must be 1")
        return cls(
            **{name: obj[name] for name in fields if name != "schema_version"}
        )


@dataclass(frozen=True)
class RuntimeIdentityV1:
    git_commit: str
    expected_git_commit: str
    config_digest: str
    deployment_evidence_digest: str
    namespace_id: str
    execution_environment_digest: str
    profiler_activation_profile_digest: str
    scoring_shadow_profile_digest: str
    controller_schema_version: int
    history_schema_version: int
    campaign_schema_version: int
    agent_protocol_digest: str

    def __post_init__(self) -> None:
        for name in ("git_commit", "expected_git_commit"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _COMMIT_RE.fullmatch(value):
                raise ValueError(f"{name} must be a full Git object ID")
        for name in (
            "config_digest",
            "deployment_evidence_digest",
            "namespace_id",
            "execution_environment_digest",
            "profiler_activation_profile_digest",
            "scoring_shadow_profile_digest",
            "agent_protocol_digest",
        ):
            _digest(getattr(self, name), name)
        for name in (
            "controller_schema_version",
            "history_schema_version",
            "campaign_schema_version",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def material(self) -> dict[str, Any]:
        return {
            "schema_version": CONSOLE_PROTOCOL_VERSION,
            "git_commit": self.git_commit,
            "expected_git_commit": self.expected_git_commit,
            "config_digest": self.config_digest,
            "deployment_evidence_digest": self.deployment_evidence_digest,
            "namespace_id": self.namespace_id,
            "execution_environment_digest": self.execution_environment_digest,
            "profiler_activation_profile_digest": (
                self.profiler_activation_profile_digest
            ),
            "scoring_shadow_profile_digest": self.scoring_shadow_profile_digest,
            "controller_schema_version": self.controller_schema_version,
            "history_schema_version": self.history_schema_version,
            "campaign_schema_version": self.campaign_schema_version,
            "agent_protocol_digest": self.agent_protocol_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.material())

    def to_dict(self) -> dict[str, Any]:
        value = self.material()
        value["runtime_identity_digest"] = self.digest
        return value

    @classmethod
    def from_value(cls, value: object) -> "RuntimeIdentityV1":
        material_fields = frozenset(
            {
                "schema_version",
                "git_commit",
                "expected_git_commit",
                "config_digest",
                "deployment_evidence_digest",
                "namespace_id",
                "execution_environment_digest",
                "profiler_activation_profile_digest",
                "scoring_shadow_profile_digest",
                "controller_schema_version",
                "history_schema_version",
                "campaign_schema_version",
                "agent_protocol_digest",
            }
        )
        obj = _strict_object(
            value,
            name="runtime identity",
            required=material_fields | {"runtime_identity_digest"},
        )
        if obj["schema_version"] != CONSOLE_PROTOCOL_VERSION or type(
            obj["schema_version"]
        ) is not int:
            raise ValueError("runtime identity schema_version must be 1")
        identity = cls(
            **{
                name: obj[name]
                for name in material_fields
                if name != "schema_version"
            }
        )
        if identity.agent_protocol_digest != AGENT_PROTOCOL_DIGEST:
            raise ValueError("runtime identity agent protocol digest is not trusted")
        supplied_digest = _digest(
            obj["runtime_identity_digest"], "runtime_identity_digest"
        )
        if supplied_digest != identity.digest:
            raise ValueError("runtime identity digest does not match its material")
        return identity


@dataclass(frozen=True)
class ConsoleSnapshotV1:
    status: str
    identity: RuntimeIdentityV1
    cursor: CompositeEventCursorV1
    observed_at: str
    data: Mapping[str, Any]
    source_digests: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"STABLE", "CHANGING", "UNAVAILABLE"}:
            raise ValueError("snapshot status is invalid")
        if not isinstance(self.identity, RuntimeIdentityV1):
            raise ValueError("snapshot identity is invalid")
        if not isinstance(self.cursor, CompositeEventCursorV1):
            raise ValueError("snapshot cursor is invalid")
        _bounded_text(self.observed_at, "observed_at", maximum=64)
        if not isinstance(self.data, Mapping):
            raise ValueError("snapshot data must be a mapping")
        if not isinstance(self.source_digests, Mapping):
            raise ValueError("source_digests must be a mapping")
        for name, digest in self.source_digests.items():
            _identifier(name, "source digest name")
            _digest(digest, f"source_digests[{name}]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONSOLE_PROTOCOL_VERSION,
            "status": self.status,
            "runtime_identity": self.identity.to_dict(),
            "cursor": self.cursor.to_dict(),
            "observed_at": self.observed_at,
            "source_digests": dict(self.source_digests),
            "data": dict(self.data),
        }


@dataclass(frozen=True)
class ConsoleAgentRequestV1:
    request_id: str
    operation: str
    payload: Mapping[str, Any]

    ALLOWED_OPERATIONS = frozenset(
        {"handshake", "snapshot", "subscribe", "prepare", "execute", "reconcile"}
    )

    def __post_init__(self) -> None:
        _identifier(self.request_id, "request_id")
        if self.operation not in self.ALLOWED_OPERATIONS:
            raise ValueError("console agent operation is not allowlisted")
        if not isinstance(self.payload, Mapping):
            raise ValueError("console agent payload must be an object")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONSOLE_PROTOCOL_VERSION,
            "request_id": self.request_id,
            "operation": self.operation,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_value(cls, value: object) -> "ConsoleAgentRequestV1":
        fields = frozenset({"schema_version", "request_id", "operation", "payload"})
        obj = _strict_object(value, name="console agent request", required=fields)
        if obj["schema_version"] != CONSOLE_PROTOCOL_VERSION or type(
            obj["schema_version"]
        ) is not int:
            raise ValueError("console agent request schema_version must be 1")
        return cls(
            request_id=obj["request_id"],
            operation=obj["operation"],
            payload=obj["payload"],
        )


@dataclass(frozen=True)
class ProblemV1:
    code: str
    title: str
    detail: str
    retryable: bool = False

    def __post_init__(self) -> None:
        _identifier(self.code, "problem code")
        _bounded_text(self.title, "problem title", maximum=256)
        _bounded_text(self.detail, "problem detail", maximum=4096)
        if not isinstance(self.retryable, bool):
            raise ValueError("problem retryable must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONSOLE_PROTOCOL_VERSION,
            "code": self.code,
            "title": self.title,
            "detail": self.detail,
            "retryable": self.retryable,
        }


@dataclass(frozen=True)
class ConsoleAgentResponseV1:
    request_id: str
    status: str
    payload: Mapping[str, Any] | None = None
    problem: ProblemV1 | None = None

    def __post_init__(self) -> None:
        _identifier(self.request_id, "request_id")
        if self.status not in {"SUCCESS", "ERROR"}:
            raise ValueError("console response status is invalid")
        if self.status == "SUCCESS":
            if not isinstance(self.payload, Mapping) or self.problem is not None:
                raise ValueError("successful console response requires only payload")
        elif self.payload is not None or not isinstance(self.problem, ProblemV1):
            raise ValueError("error console response requires only problem")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONSOLE_PROTOCOL_VERSION,
            "request_id": self.request_id,
            "status": self.status,
            "payload": None if self.payload is None else dict(self.payload),
            "problem": None if self.problem is None else self.problem.to_dict(),
        }


AGENT_PROTOCOL_DIGEST = canonical_sha256(
    {
        "schema_version": CONSOLE_PROTOCOL_VERSION,
        "request_operations": sorted(ConsoleAgentRequestV1.ALLOWED_OPERATIONS),
        "max_line_bytes": MAX_AGENT_LINE_BYTES,
        "max_response_bytes": MAX_AGENT_RESPONSE_BYTES,
        "cursor_fields": sorted(
            key
            for key in CompositeEventCursorV1().to_dict()
            if key != "schema_version"
        ),
        "operation_kinds": sorted(OPERATION_KINDS),
        "operation_contracts": (
            "prepare-confirm-reconcile-v1",
            "candidate-bundle-v2",
        ),
        "runtime_identity_fields": (
            "git_commit",
            "expected_git_commit",
            "config_digest",
            "deployment_evidence_digest",
            "namespace_id",
            "execution_environment_digest",
            "profiler_activation_profile_digest",
            "scoring_shadow_profile_digest",
            "controller_schema_version",
            "history_schema_version",
            "campaign_schema_version",
            "agent_protocol_digest",
        ),
    }
)


def encode_wire(value: Mapping[str, Any]) -> bytes:
    encoded = canonical_json_bytes(value) + b"\n"
    if len(encoded) > MAX_AGENT_RESPONSE_BYTES:
        raise ValueError("console response exceeds its size bound")
    return encoded


__all__ = [
    "AGENT_PROTOCOL_DIGEST",
    "CONSOLE_PROTOCOL_VERSION",
    "MAX_AGENT_LINE_BYTES",
    "MAX_AGENT_RESPONSE_BYTES",
    "OPERATION_KINDS",
    "TASK_KINDS",
    "CompositeEventCursorV1",
    "ConsoleAgentRequestV1",
    "ConsoleAgentResponseV1",
    "ConsoleSnapshotV1",
    "DraftV1",
    "ManualEvaluationRequestV1",
    "OperationReceiptV1",
    "PreparedOperationV1",
    "ProblemV1",
    "RuntimeIdentityV1",
    "encode_wire",
    "strict_json_loads",
]
