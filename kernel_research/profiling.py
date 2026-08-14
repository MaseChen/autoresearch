"""Trusted profiling doctor and bounded, advisory-only profiling recipes.

The doctor remains a dependency-light host capability probe.  Collection is a
separate fail-closed path: it accepts only reviewed recipe identifiers, reads
one exact V2 History experiment, requires a fully-qualified Campaign soak
gate, and launches one immutable profiler image with fixed Docker arguments.
Profiling evidence is deliberately incapable of changing promotion, Campaign
lineage, or the deployment baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import subprocess
import tempfile
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from .autorun.controller import FATAL_GPU_MARKERS, ResearchController, gpu_lock
from .autorun.deployment import deployment_runtime_root
from .autorun.errors import ControlledRuntimeError
from .autorun.models import ControllerConfig, GPU1_DEVICES
from .autorun.runtime import CommandResult, CommandRunner
from .campaign.models import (
    BudgetAmount,
    CampaignMode,
    CampaignStatus,
    ResourceLease,
)
from .campaign.paths import (
    campaign_maintenance_fence,
    validate_production_campaign_database,
)
from .campaign.soak import SoakGate
from .campaign.soak_collector import COUNT_FIELDS, SoakObservationCollector
from .campaign.store import CampaignStore
from .constants import CURRENT_C500_EVALUATION_PROTOCOL_ID
from .platform.canonical import canonical_json_text, canonical_sha256, require_sha256_digest
from .platform.identity import BaselineRef, ExperimentIdentity
from .platform.profiles import CURRENT_RESEARCH_NAMESPACE
from .platform.proposal import CandidateBundle, TRITON_PYTHON_BUNDLE_LIMITS
from .profiler_contract import (
    MCTRACER_PATH,
    MCTRACER_SHA256,
    MCTRACER_VERSION,
    MCTOOLS_EXT_LITE_PATH,
    MCTOOLS_EXT_LITE_SHA256,
    MCTOOLS_EXT_PATH,
    MCTOOLS_EXT_SHA256,
    PROFILE_CANDIDATE_CONTAINER_PATH,
    PROFILE_CANDIDATE_LIMIT_BYTES,
    PROFILE_COLLECTION_SCHEMA_VERSION,
    PROFILE_CANARY_GPU_SECONDS,
    PROFILE_CANARY_WALL_SECONDS,
    PROFILE_CPU_LIMIT,
    PROFILE_LEASE_TTL_SECONDS,
    PROFILE_MEMORY_LIMIT,
    PROFILE_OUTCOME_CONTAINER_PATH,
    PROFILE_OUTCOME_LIMIT_BYTES,
    PROFILE_OUTPUT_LIMIT_BYTES,
    PROFILE_PID_LIMIT,
    PROFILE_RAW_TRACE_LIMIT_BYTES,
    PROFILE_RESOURCE_ID,
    PROFILE_RESULT_CONTAINER_PATH,
    PROFILE_RESULT_LIMIT_BYTES,
    PROFILE_TIMEOUT_SECONDS,
    PROFILE_TRACE_CONTAINER_PATH,
    PROFILE_UNAVAILABLE_REASON_CODES,
    PROFILER_ACTIVE,
    PROFILER_ENTRYPOINT,
    PROFILER_IMAGE,
    PROFILER_PROFILE_DIGEST,
    PROFILER_RECIPES,
    PROFILER_WORKER_REVISION,
    require_active_profiler,
)


PROFILE_DOCTOR_API_VERSION = 1
DEFAULT_TIMEOUT_SEC = 5.0
DEFAULT_OUTPUT_LIMIT_BYTES = 64 * 1024

PROFILE_COLLECTION_API_VERSION = PROFILE_COLLECTION_SCHEMA_VERSION
_PROFILE_BUDGET_CEILING_MS = int(PROFILE_TIMEOUT_SECONDS * 1000)

_HISTORY_SCHEMA_VERSION = 3
_PROFILE_REASON_TEXT = {
    "COUNTER_NOT_EXPOSED": "metric is not exposed by this device/tool version",
    "NOT_SUPPORTED": "metric is not supported by the pinned profiler recipe",
    "PERMISSION_DENIED": "metric is unavailable with the reviewed capability set",
    "TOOL_VERSION_UNSUPPORTED": "metric is unavailable in the pinned tool version",
}
if frozenset(_PROFILE_REASON_TEXT) != PROFILE_UNAVAILABLE_REASON_CODES:
    raise RuntimeError("profiler unavailable reason allowlist drifted")
_PROFILE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_STAGE_SUITE_RANK = {
    ("smoke", "smoke"): 1,
    ("quick", "quick"): 2,
    ("full_primary", "full"): 3,
    ("confirmation", "full"): 3,
}


@dataclass(frozen=True, slots=True)
class ProfileMetric:
    """One host-whitelisted metric emitted by a reviewed recipe."""

    metric_id: str
    value_kind: str
    unit: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.metric_id):
            raise ValueError("profile metric_id is invalid")
        if self.value_kind not in {"integer", "number", "digest"}:
            raise ValueError("profile metric value_kind is invalid")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError("profile metric unit must be non-empty")


@dataclass(frozen=True, slots=True)
class ProfileRecipe:
    """A complete immutable runner recipe; no CLI argument can extend it."""

    recipe_id: str
    case_id: str
    minimum_correctness_rank: int
    requires_gpu: bool
    metrics: tuple[ProfileMetric, ...]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", self.recipe_id):
            raise ValueError("profile recipe_id is invalid")
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.case_id):
            raise ValueError("profile case_id is invalid")
        if self.minimum_correctness_rank not in {1, 2}:
            raise ValueError("profile minimum correctness rank is invalid")
        if not self.metrics or len({item.metric_id for item in self.metrics}) != len(
            self.metrics
        ):
            raise ValueError("profile recipe metrics must be non-empty and unique")


BUILTIN_PROFILE_RECIPES: Mapping[str, ProfileRecipe] = MappingProxyType(
    {
        recipe_id: ProfileRecipe(
            recipe_id=recipe_id,
            case_id=str(value["case_id"]),
            minimum_correctness_rank=int(value["minimum_correctness_rank"]),
            requires_gpu=bool(value["requires_gpu"]),
            metrics=tuple(
                ProfileMetric(metric_id, value_kind, unit)
                for metric_id, value_kind, unit in value["metrics"]
            ),
        )
        for recipe_id, value in PROFILER_RECIPES.items()
    }
)
PROFILE_RECIPE_IDS = tuple(BUILTIN_PROFILE_RECIPES)


@dataclass(frozen=True, slots=True)
class _ProfileSubject:
    experiment_uid: str
    namespace_id: str
    condition_digest: str
    artifact_id: str
    execution_environment_digest: str
    campaign_id: str | None
    run_id: str
    stage: str
    suite: str
    replicate_kind: str
    candidate_object_path: Path
    candidate_content_sha256: str


class _ProfileRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None,
        timeout_sec: float,
        max_output_bytes: int,
        container_name: str | None = None,
        docker_binary: Path | None = None,
    ) -> CommandResult: ...


class _ProfilingGate(Protocol):
    def status(self) -> Mapping[str, Any]: ...

    def profiling_allowed(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ToolProbe:
    """One fixed executable probe from the reviewed profiling recipe."""

    tool_id: str
    executable_names: tuple[str, ...]
    version_arguments: tuple[str, ...] = ("--version",)
    required: bool = True
    accepted_exit_codes: tuple[int, ...] = (0,)
    required_output_patterns: tuple[str, ...] = ()
    forbidden_output_markers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tool_id or any(character.isspace() for character in self.tool_id):
            raise ValueError("tool_id must be a non-empty token")
        if not self.executable_names:
            raise ValueError("executable_names must not be empty")
        for executable in self.executable_names:
            if not executable or "/" in executable or "\\" in executable:
                raise ValueError("executable names must be bare trusted names")
        for argument in self.version_arguments:
            if not argument or "\x00" in argument:
                raise ValueError("version arguments must be fixed non-empty strings")
        if (
            not self.accepted_exit_codes
            or len(set(self.accepted_exit_codes)) != len(self.accepted_exit_codes)
            or any(
                type(code) is not int or code < 0 or code > 255
                for code in self.accepted_exit_codes
            )
        ):
            raise ValueError(
                "accepted_exit_codes must be unique byte-sized integers"
            )
        for pattern in self.required_output_patterns:
            if not pattern or "\x00" in pattern:
                raise ValueError(
                    "required output patterns must be non-empty bounded text"
                )
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError("required output pattern is invalid") from exc
        for marker in self.forbidden_output_markers:
            if not marker or "\x00" in marker:
                raise ValueError(
                    "forbidden output markers must be non-empty bounded text"
                )


DEFAULT_TOOL_PROBES = (
    ToolProbe("metax-system-management", ("mx-smi",)),
    # MetaX 3.2.1 mcTracer has no version flag: ``--version`` is interpreted
    # as the target executable to trace and therefore fails with execvpe.
    # Its reviewed ``--help`` path exits zero and includes the tool version.
    ToolProbe(
        "metax-trace-collector",
        ("mcTracer", "mctracer"),
        version_arguments=("--help",),
        accepted_exit_codes=(1,),
        required_output_patterns=(
            r"\b3\.2\.1\.10-df74b02\b",
            r"(?i)\bhelp\b",
            r"(?i)\busage\b",
        ),
        forbidden_output_markers=(
            "fatal",
            "segmentation fault",
            "core dumped",
            "execvpe",
            "failed to execute",
            "failed to run command",
            "no such file or directory",
            "error while loading shared libraries",
            "cannot open shared object file",
            "symbol lookup error",
        ),
    ),
    # mcProfiler is commonly a UI/client while the Linux target exposes its
    # collector.  It is therefore useful evidence but not a Linux hard gate.
    ToolProbe(
        "metax-profiler-client",
        ("mcProfiler", "mcprofiler"),
        required=False,
    ),
)

DEFAULT_LIBRARY_PATHS = (
    Path("/opt/maca/lib/libmcToolsExt.so"),
    Path("/opt/maca/lib64/libmcToolsExt.so"),
    Path("/usr/local/maca/lib/libmcToolsExt.so"),
)

DEFAULT_DEVICE_PATHS = (
    Path("/dev/mxcd"),
    Path("/dev/dri/card2"),
    Path("/dev/dri/renderD129"),
)


def _bounded_version_command(
    argv: Sequence[str],
    *,
    timeout_sec: float,
    output_limit_bytes: int,
    accepted_exit_codes: Sequence[int] = (0,),
    required_output_patterns: Sequence[str] = (),
    forbidden_output_markers: Sequence[str] = (),
) -> Mapping[str, Any]:
    """Execute one already-reviewed argv without a shell or inherited stdin."""

    try:
        completed = subprocess.run(
            tuple(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_sec,
            check=False,
            shell=False,
            env={
                "PATH": os.environ.get("PATH", ""),
                "LANG": "C",
                "LC_ALL": "C",
            },
        )
    except subprocess.TimeoutExpired:
        return {"status": "UNAVAILABLE", "reason": "version probe timed out"}
    except OSError as exc:
        return {
            "status": "UNAVAILABLE",
            "reason": f"version probe could not start: {type(exc).__name__}",
        }

    combined = completed.stdout + b"\n" + completed.stderr
    if len(combined) > output_limit_bytes:
        return {
            "status": "UNAVAILABLE",
            "reason": "version probe exceeded output limit",
            "exit_code": completed.returncode,
        }
    text = combined.decode("utf-8", errors="replace").strip()
    lowered = text.casefold()
    forbidden = next(
        (
            marker
            for marker in forbidden_output_markers
            if marker.casefold() in lowered
        ),
        None,
    )
    if forbidden is not None:
        return {
            "status": "UNAVAILABLE",
            "exit_code": completed.returncode,
            "version_output": text[:4096],
            "reason": "version probe output contains a forbidden failure marker",
        }
    if completed.returncode not in set(accepted_exit_codes):
        return {
            "status": "UNAVAILABLE",
            "exit_code": completed.returncode,
            "version_output": text[:4096],
            "reason": "version probe returned an unaccepted status",
        }
    if any(re.search(pattern, text) is None for pattern in required_output_patterns):
        return {
            "status": "UNAVAILABLE",
            "exit_code": completed.returncode,
            "version_output": text[:4096],
            "reason": "version probe output signature mismatch",
        }
    return {
        "status": "AVAILABLE",
        "exit_code": completed.returncode,
        "version_output": text[:4096],
    }


def _path_access(path: Path, *, executable: bool = False) -> dict[str, Any]:
    exists = path.exists()
    readable = exists and os.access(path, os.R_OK)
    result: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "readable": readable,
    }
    if executable:
        result["writable"] = exists and os.access(path, os.W_OK)
    return result


def run_profiling_doctor(
    *,
    tool_probes: Sequence[ToolProbe] = DEFAULT_TOOL_PROBES,
    library_paths: Sequence[Path] = DEFAULT_LIBRARY_PATHS,
    device_paths: Sequence[Path] = DEFAULT_DEVICE_PATHS,
    executable_resolver: Callable[[str], str | None] = shutil.which,
    command_runner: Callable[..., Mapping[str, Any]] = _bounded_version_command,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT_BYTES,
) -> dict[str, Any]:
    """Return a bounded capability report with explicit unavailable reasons."""

    if timeout_sec <= 0:
        raise ValueError("timeout_sec must be positive")
    if output_limit_bytes <= 0:
        raise ValueError("output_limit_bytes must be positive")

    tools: list[dict[str, Any]] = []
    for probe in tool_probes:
        resolved: str | None = None
        selected_name: str | None = None
        for name in probe.executable_names:
            candidate = executable_resolver(name)
            if candidate is not None:
                resolved = str(Path(candidate).resolve(strict=False))
                selected_name = name
                break
        if resolved is None:
            tools.append(
                {
                    "tool_id": probe.tool_id,
                    "required": probe.required,
                    "status": "UNAVAILABLE",
                    "reason": "executable not found",
                    "searched_names": list(probe.executable_names),
                }
            )
            continue
        outcome = dict(
            command_runner(
                (resolved, *probe.version_arguments),
                timeout_sec=timeout_sec,
                output_limit_bytes=output_limit_bytes,
                accepted_exit_codes=probe.accepted_exit_codes,
                required_output_patterns=probe.required_output_patterns,
                forbidden_output_markers=probe.forbidden_output_markers,
            )
        )
        tools.append(
            {
                "tool_id": probe.tool_id,
                "required": probe.required,
                "executable": resolved,
                "selected_name": selected_name,
                **outcome,
            }
        )

    libraries = [_path_access(Path(path)) for path in library_paths]
    devices = [_path_access(Path(path), executable=True) for path in device_paths]
    missing_required_tools = [
        tool["tool_id"]
        for tool in tools
        if tool["required"] and tool["status"] != "AVAILABLE"
    ]
    library_ready = any(item["readable"] for item in libraries)
    devices_ready = all(item["readable"] and item["writable"] for item in devices)
    reasons: list[str] = []
    if missing_required_tools:
        reasons.append(
            "required tools unavailable: " + ", ".join(missing_required_tools)
        )
    if not library_ready:
        reasons.append("libmcToolsExt.so was not found at a trusted path")
    if not devices_ready:
        reasons.append("one or more required C500 device nodes are inaccessible")
    status = "READY" if not reasons else "UNAVAILABLE"
    return {
        "api_version": PROFILE_DOCTOR_API_VERSION,
        "command": "profile doctor",
        "status": status,
        "tools": tools,
        "support_library": {
            "status": "AVAILABLE" if library_ready else "UNAVAILABLE",
            "candidates": libraries,
            **(
                {}
                if library_ready
                else {"reason": "libmcToolsExt.so not found or unreadable"}
            ),
        },
        "devices": {
            "status": "AVAILABLE" if devices_ready else "UNAVAILABLE",
            "nodes": devices,
            **(
                {}
                if devices_ready
                else {"reason": "required device permissions are incomplete"}
            ),
        },
        "reasons": reasons,
        "advisory_only": True,
        "promotion_effect": "none",
    }


def _strict_json_object(data: bytes, *, field: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"{field} contains duplicate JSON keys")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{field} contains non-finite JSON number {value}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    # This also rejects values outside the canonical JSON data model.
    canonical_json_text(value)
    return value


def _read_regular_file(path: Path, *, limit: int, field: str) -> bytes:
    if limit <= 0:
        raise ValueError("file limit must be positive")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{field} is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{field} must be a regular non-symlink file")
        if metadata.st_size > limit:
            raise ValueError(f"{field} exceeds its fixed size limit")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, limit + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > limit:
                raise ValueError(f"{field} exceeds its fixed size limit")
        if os.fstat(descriptor).st_size != metadata.st_size:
            raise ValueError(f"{field} changed while it was being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_only_history_subject(
    *,
    state_dir: Path,
    experiment_uid: str,
    namespace_id: str,
    execution_environment_digest: str,
    recipe: ProfileRecipe,
) -> _ProfileSubject:
    """Load and validate one exact, correct V2 experiment without DB writes."""

    require_sha256_digest(namespace_id, field="profile namespace_id")
    require_sha256_digest(
        execution_environment_digest,
        field="profile execution_environment_digest",
    )
    history_path = state_dir / "history.sqlite3"
    if history_path.is_symlink() or not history_path.is_file():
        raise ValueError("state_dir does not contain a regular History database")
    connection = sqlite3.connect(
        history_path.resolve().as_uri() + "?mode=ro",
        uri=True,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        integrity = connection.execute("PRAGMA quick_check").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise ValueError("History integrity is unavailable")
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise ValueError("History foreign-key integrity is unavailable")
        if int(connection.execute("PRAGMA user_version").fetchone()[0]) != (
            _HISTORY_SCHEMA_VERSION
        ):
            raise ValueError("bounded profiling requires History schema V3")
        row = connection.execute(
            """
            SELECT experiment_uid, namespace_id, condition_digest, artifact_id,
                   candidate_hash,
                   backend, suite, status, replicate_kind, replicate_index,
                   identity_json, result_json, error_summary
            FROM experiments WHERE experiment_uid = ?
            """,
            (experiment_uid,),
        ).fetchone()
        if row is None:
            raise ValueError("profiling experiment UID does not exist")
        cases = connection.execute(
            """
            SELECT case_name, passed FROM case_measurements
            WHERE experiment_id = (
                SELECT id FROM experiments WHERE experiment_uid = ?
            ) ORDER BY id
            """,
            (experiment_uid,),
        ).fetchall()
        bundle_artifact = connection.execute(
            """
            SELECT artifact_id, artifact_kind, content_sha256, object_path,
                   byte_size, manifest_json
            FROM candidate_artifacts WHERE artifact_id = ?
            """,
            (str(row["artifact_id"]),),
        ).fetchone()
    finally:
        connection.close()

    try:
        identity = ExperimentIdentity.from_value(
            _strict_json_object(
                str(row["identity_json"]).encode("utf-8"),
                field="History experiment identity",
            )
        )
        result = _strict_json_object(
            str(row["result_json"]).encode("utf-8"),
            field="History experiment result",
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("History experiment has invalid V2 evidence") from exc
    if not identity.is_scientifically_comparable:
        raise ValueError("profiling requires a resolved V2 execution environment")
    expected_columns = {
        "experiment_uid": identity.experiment_uid,
        "namespace_id": identity.namespace_id,
        "condition_digest": identity.condition_digest,
        "artifact_id": str(identity.candidate_artifact_id),
        "suite": identity.suite,
        "replicate_kind": identity.replicate_kind,
        "replicate_index": identity.replicate_index,
    }
    mismatched = [
        field for field, expected in expected_columns.items() if row[field] != expected
    ]
    if mismatched:
        raise ValueError(
            "History identity columns disagree: " + ", ".join(sorted(mismatched))
        )
    if identity.namespace_id != namespace_id:
        raise ValueError("profiling experiment belongs to another namespace")
    if identity.execution_environment.digest != execution_environment_digest:
        raise ValueError("profiling execution environment assertion mismatch")
    if row["backend"] != "c500":
        raise ValueError("bounded MetaX profiling requires c500 evidence")
    if (
        row["status"] != "SUCCESS"
        or row["error_summary"] is not None
        or result.get("status") != "SUCCESS"
    ):
        raise ValueError("profiling requires a successful evaluation")
    if not cases or any(case["passed"] is not True and case["passed"] != 1 for case in cases):
        raise ValueError("profiling requires complete passing correctness evidence")
    rank = _STAGE_SUITE_RANK.get((identity.stage, identity.suite))
    if rank is None:
        raise ValueError("profiling experiment stage/suite is not a reviewed stage")
    if rank < recipe.minimum_correctness_rank:
        minimum = "quick" if recipe.minimum_correctness_rank == 2 else "smoke"
        raise ValueError(f"{recipe.recipe_id} requires at least correct {minimum} evidence")
    fixed_case = [case for case in cases if case["case_name"] == recipe.case_id]
    if len(fixed_case) != 1 or fixed_case[0]["passed"] != 1:
        raise ValueError(
            "profiling requires passing correctness evidence for its fixed case"
        )
    if bundle_artifact is None:
        raise ValueError("profiling candidate has no authoritative CAS record")
    if bundle_artifact["artifact_id"] != str(identity.candidate_artifact_id):
        raise ValueError("profiling candidate artifact identity mismatch")
    if type(bundle_artifact["byte_size"]) is not int or not (
        0 <= bundle_artifact["byte_size"] <= PROFILE_CANDIDATE_LIMIT_BYTES
    ):
        raise ValueError("profiling candidate size is unavailable or excessive")

    relative_text = str(bundle_artifact["object_path"])
    relative = PurePosixPath(relative_text)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != relative_text
        or len(relative.parts) != 4
        or relative.parts[:2] != ("objects", "sha256")
        or not re.fullmatch(r"[0-9a-f]{2}", relative.parts[2])
        or not re.fullmatch(r"[0-9a-f]{62}", relative.parts[3])
    ):
        raise ValueError("profiling candidate CAS path is unsafe")
    object_path = state_dir.joinpath(*relative.parts)
    cursor = state_dir
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError("profiling candidate CAS path traverses a symlink")
    try:
        resolved = object_path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("profiling candidate CAS object is unavailable") from exc
    if not resolved.is_relative_to(state_dir) or resolved != object_path:
        raise ValueError("profiling candidate CAS object escapes state_dir")
    mounted_candidate = _read_regular_file(
        object_path,
        limit=PROFILE_CANDIDATE_LIMIT_BYTES,
        field="profiling candidate CAS object",
    )
    content_digest = hashlib.sha256(mounted_candidate).hexdigest()
    if (
        content_digest != bundle_artifact["content_sha256"]
        or len(mounted_candidate) != bundle_artifact["byte_size"]
        or object_path.parts[-2:] != (content_digest[:2], content_digest[2:])
    ):
        raise ValueError("profiling candidate CAS object is corrupted")
    if bundle_artifact["artifact_kind"] == "source_bundle_v1":
        try:
            bundle = CandidateBundle.from_value(
                _strict_json_object(
                    mounted_candidate, field="profiling bundle"
                ),
                limits=TRITON_PYTHON_BUNDLE_LIMITS,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("profiling bundle manifest is invalid") from exc
        entrypoint = next(
            item for item in bundle.files if item.path == bundle.entrypoint
        )
        if (
            str(bundle.artifact_id) != str(identity.candidate_artifact_id)
            or hashlib.sha256(entrypoint.content_bytes).hexdigest()
            != row["candidate_hash"]
        ):
            raise ValueError("profiling bundle and entrypoint source disagree")
    elif (
        bundle_artifact["artifact_kind"]
        not in {"source_text_v1", "legacy_source_v1"}
        or content_digest != row["candidate_hash"]
    ):
        raise ValueError("profiling artifact kind is unsupported")
    return _ProfileSubject(
        experiment_uid=identity.experiment_uid,
        namespace_id=identity.namespace_id,
        condition_digest=identity.condition_digest,
        artifact_id=str(identity.candidate_artifact_id),
        execution_environment_digest=identity.execution_environment.digest,
        campaign_id=identity.campaign_id,
        run_id=identity.run_id,
        stage=identity.stage,
        suite=identity.suite,
        replicate_kind=identity.replicate_kind,
        candidate_object_path=object_path,
        candidate_content_sha256=str(row["candidate_hash"]),
    )


def _runtime_root(config: ControllerConfig) -> Path:
    return deployment_runtime_root(
        state_dir=config.state_dir,
        controller_dir=config.controller_dir,
        checkpoint_dir=config.checkpoint_dir,
    )


def _gate_authorization(
    gate: _ProfilingGate, *, expected_invariant_digest: str | None = None
) -> str:
    first = dict(gate.status())
    if (
        first.get("current_observation_status") != "AVAILABLE"
        or first.get("profiling_allowed") is not True
        or first.get("active_invariant_matches") is not True
        or not gate.profiling_allowed()
    ):
        raise ValueError("current Campaign soak evidence does not allow profiling")
    second = dict(gate.status())
    digest = first.get("current_invariant_snapshot_digest")
    if not isinstance(digest, str):
        raise ValueError("Campaign soak has no current invariant digest")
    require_sha256_digest(digest, field="soak invariant digest")
    if (
        second.get("current_observation_status") != "AVAILABLE"
        or second.get("profiling_allowed") is not True
        or second.get("active_invariant_matches") is not True
        or second.get("current_invariant_snapshot_digest") != digest
    ):
        raise ValueError("Campaign soak invariant changed during authorization")
    if expected_invariant_digest is not None and digest != expected_invariant_digest:
        raise ValueError("Campaign soak invariant changed during profiling")
    return digest


def _gate_authorization_with_reserved_profile_intent(
    gate: _ProfilingGate,
    *,
    collector: SoakObservationCollector,
    store: CampaignStore,
    gate_id: str,
    expected_invariant_digest: str,
) -> None:
    """Revalidate Soak while the caller's durable intent is RESERVED.

    The canonical collector deliberately counts every non-child RESERVED
    action as a potential budget leak.  Profiling must reserve before running,
    so exactly its own intent temporarily makes ``profiling_allowed`` false.
    This narrow recheck accepts only that one expected count; every other Soak
    violation and every invariant change remains fatal to the action.
    """

    first = dict(gate.status())
    second = dict(gate.status())
    for status in (first, second):
        if (
            status.get("current_observation_status") != "AVAILABLE"
            or status.get("active_invariant_matches") is not True
            or status.get("current_invariant_snapshot_digest")
            != expected_invariant_digest
        ):
            raise ValueError(
                "Campaign soak invariant changed during reserved profiling"
            )
    if (
        first.get("profiling_allowed") is True
        and second.get("profiling_allowed") is True
        and gate.profiling_allowed()
    ):
        # Injected gates and future collectors may natively understand this
        # reviewed action kind.  Their stronger affirmative decision wins.
        return

    now = time.time()
    observation = collector.collect(
        interval_start_epoch=now,
        interval_end_epoch=now,
    )
    expected_counts = {field: 0 for field in COUNT_FIELDS}
    expected_counts["budget_leak_count"] = 1
    if (
        observation.status != "AVAILABLE"
        or observation.invariant_snapshot_digest != expected_invariant_digest
        or dict(observation.counts) != expected_counts
        or not store.soak_profiling_allowed(
            gate_id, expected_invariant_digest
        )
    ):
        raise ValueError(
            "current Campaign soak evidence has violations beyond the reserved "
            "profiling intent"
        )


def _docker_mount(source: Path, destination: str, *, readonly: bool) -> str:
    rendered = str(source)
    if "," in rendered or "\x00" in rendered or "\n" in rendered:
        raise ValueError("profiling mount source cannot be represented safely")
    suffix = ",readonly" if readonly else ""
    return f"type=bind,src={rendered},dst={destination}{suffix}"


def _profile_argv(
    config: ControllerConfig,
    *,
    recipe: ProfileRecipe,
    subject: _ProfileSubject,
    output_directory: Path,
    container_name: str,
    invariant_digest: str,
    budget_action_key: str,
    lease: ResourceLease | None,
) -> tuple[str, ...]:
    if subject.campaign_id is None:
        raise ValueError("Campaign profiling requires a Campaign-owned subject")
    argv = [
        str(config.docker_binary),
        "run",
        "--rm",
        "--pull=never",
        "--name",
        container_name,
        "--label",
        "kernel-autoresearch.role=bounded-profiler",
        "--init",
        "--read-only",
        "--network=none",
        "--ipc=none",
        "--user",
        f"{config.container_uid}:{config.container_gid}",
        "--group-add",
        str(config.video_gid),
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        str(PROFILE_PID_LIMIT),
        "--memory",
        PROFILE_MEMORY_LIMIT,
        "--cpus",
        str(PROFILE_CPU_LIMIT),
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=64m,mode=700",
        "--mount",
        _docker_mount(
            subject.candidate_object_path,
            PROFILE_CANDIDATE_CONTAINER_PATH,
            readonly=True,
        ),
        "--mount",
        _docker_mount(output_directory, "/output", readonly=False),
    ]
    if recipe.requires_gpu:
        for device in config.gpu_devices:
            argv.extend(("--device", f"{device}:{device}:rwm"))
    argv.extend(
        (
            "--entrypoint",
            PROFILER_ENTRYPOINT,
            PROFILER_IMAGE,
            "run",
            "--recipe",
            recipe.recipe_id,
            "--case-id",
            recipe.case_id,
            "--candidate",
            PROFILE_CANDIDATE_CONTAINER_PATH,
            "--result",
            PROFILE_RESULT_CONTAINER_PATH,
            "--outcome",
            PROFILE_OUTCOME_CONTAINER_PATH,
            "--raw-trace",
            PROFILE_TRACE_CONTAINER_PATH,
            "--max-result-bytes",
            str(PROFILE_RESULT_LIMIT_BYTES),
            "--max-outcome-bytes",
            str(PROFILE_OUTCOME_LIMIT_BYTES),
            "--max-raw-trace-bytes",
            str(PROFILE_RAW_TRACE_LIMIT_BYTES),
            "--experiment-uid",
            subject.experiment_uid,
            "--namespace-id",
            subject.namespace_id,
            "--condition-digest",
            subject.condition_digest,
            "--artifact-id",
            subject.artifact_id,
            "--candidate-sha256",
            subject.candidate_content_sha256,
            "--environment-digest",
            subject.execution_environment_digest,
            "--soak-invariant-digest",
            invariant_digest,
            "--campaign-id",
            subject.campaign_id,
            "--run-id",
            subject.run_id,
            "--budget-action-key",
            budget_action_key,
        )
    )
    if recipe.requires_gpu:
        if lease is None:
            raise ValueError("hardware profiling requires an active trusted lease")
        argv.extend(
            (
                "--resource-id",
                lease.resource_id,
                "--fencing-epoch",
                str(lease.fencing_epoch),
            )
        )
    elif lease is not None:
        raise ValueError("compile metadata profiling must not own a GPU lease")
    return tuple(argv)


def _metric_summary(
    recipe: ProfileRecipe, result: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    metrics = result.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("profiler result metrics must be a JSON object")
    definitions = {item.metric_id: item for item in recipe.metrics}
    unknown = sorted(set(metrics) - set(definitions))
    if unknown:
        raise ValueError("profiler result contains non-whitelisted metrics")
    missing = sorted(set(definitions) - set(metrics))
    if missing:
        raise ValueError("profiler result omits required metric statuses")
    summary: dict[str, dict[str, Any]] = {}
    for metric_id, definition in definitions.items():
        value = metrics.get(metric_id)
        if not isinstance(value, dict) or set(value) not in (
            {"status", "value"},
            {"status", "reason_code"},
        ):
            raise ValueError("profiler metric does not match the fixed schema")
        status_value = value.get("status")
        if status_value == "UNAVAILABLE":
            reason_code = value.get("reason_code")
            if reason_code not in _PROFILE_REASON_TEXT:
                raise ValueError("profiler metric has an unknown unavailable reason")
            summary[metric_id] = {
                "status": "UNAVAILABLE",
                "reason": _PROFILE_REASON_TEXT[str(reason_code)],
                "unit": definition.unit,
            }
            continue
        if status_value != "AVAILABLE" or "value" not in value:
            raise ValueError("profiler metric availability is invalid")
        raw = value["value"]
        if definition.value_kind == "integer":
            if type(raw) is not int or raw < 0:
                raise ValueError("profiler integer metric is invalid")
        elif definition.value_kind == "number":
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or float(raw) < 0
            ):
                raise ValueError("profiler numeric metric is invalid")
        else:
            try:
                require_sha256_digest(raw, field="profiler digest metric")
            except (TypeError, ValueError) as exc:
                raise ValueError("profiler digest metric is invalid") from exc
        summary[metric_id] = {
            "status": "AVAILABLE",
            "value": raw,
            "unit": definition.unit,
        }
    return summary


def _toolchain_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    toolchain = result.get("toolchain")
    if not isinstance(toolchain, dict):
        raise ValueError("profiler result has no exact toolchain descriptor")
    if set(toolchain) != {
        "schema_version",
        "worker_revision",
        "tools",
        "help_contract",
        "digest",
    }:
        raise ValueError("profiler toolchain descriptor has an invalid schema")
    if (
        toolchain["schema_version"] != 1
        or toolchain["worker_revision"] != PROFILER_WORKER_REVISION
    ):
        raise ValueError("profiler toolchain descriptor identity drifted")
    tools = toolchain["tools"]
    if not isinstance(tools, dict) or set(tools) != {
        "mctracer",
        "libmcToolsExt_lite.so",
        "libmcToolsExt.so",
    }:
        raise ValueError("profiler toolchain files are incomplete")
    expected_tools = {
        "mctracer": {
            "path": MCTRACER_PATH,
            "sha256": MCTRACER_SHA256,
            "version": MCTRACER_VERSION,
        },
        "libmcToolsExt_lite.so": {
            "path": MCTOOLS_EXT_LITE_PATH,
            "sha256": MCTOOLS_EXT_LITE_SHA256,
        },
        "libmcToolsExt.so": {
            "path": MCTOOLS_EXT_PATH,
            "sha256": MCTOOLS_EXT_SHA256,
        },
    }
    if tools != expected_tools:
        raise ValueError("profiler toolchain files differ from the frozen profile")
    help_contract = toolchain["help_contract"]
    if (
        not isinstance(help_contract, dict)
        or set(help_contract)
        != {
            "argv",
            "stdin",
            "exit_code",
            "version",
            "required_markers",
        }
        or help_contract.get("argv") != [MCTRACER_PATH, "--help"]
        or help_contract.get("stdin") != "DEVNULL"
        or help_contract.get("exit_code") != 1
        or help_contract.get("version") != MCTRACER_VERSION
        or help_contract.get("required_markers") != ["Help Info", "Usage:"]
    ):
        raise ValueError("profiler mcTracer help contract is invalid")
    material = dict(toolchain)
    digest = material.pop("digest")
    if digest != canonical_sha256(material):
        raise ValueError("profiler toolchain digest is invalid")
    return dict(toolchain)


def _trace_descriptor(
    recipe: ProfileRecipe,
    result: Mapping[str, Any],
    *,
    raw_trace: bytes,
) -> dict[str, Any]:
    descriptor = result.get("trace_descriptor")
    if not isinstance(descriptor, dict):
        raise ValueError("profiler result has no trace descriptor")
    required = {"format", "sha256", "byte_size", "file_count"}
    if recipe.requires_gpu:
        required.update(
            {
                "source_bytes",
                "mctracer_exit_code",
                "target_stdout_sha256",
                "target_stderr_sha256",
                "toolchain_digest",
            }
        )
    if set(descriptor) != required:
        raise ValueError("profiler trace descriptor has an invalid schema")
    expected_format = (
        "deterministic-tar-v1"
        if recipe.requires_gpu
        else "canonical-json-manifest-v1"
    )
    if (
        descriptor["format"] != expected_format
        or descriptor["sha256"]
        != "sha256:" + hashlib.sha256(raw_trace).hexdigest()
        or descriptor["byte_size"] != len(raw_trace)
        or type(descriptor["file_count"]) is not int
        or descriptor["file_count"] <= 0
    ):
        raise ValueError("profiler trace descriptor does not match raw bytes")
    if recipe.requires_gpu:
        if (
            type(descriptor["source_bytes"]) is not int
            or descriptor["source_bytes"] <= 0
            or descriptor["source_bytes"] > len(raw_trace)
            or descriptor["mctracer_exit_code"] not in {0, 1}
        ):
            raise ValueError("profiler mctx trace descriptor is invalid")
        for field in ("target_stdout_sha256", "target_stderr_sha256"):
            require_sha256_digest(descriptor[field], field=field)
        require_sha256_digest(
            descriptor["toolchain_digest"], field="toolchain_digest"
        )
    return dict(descriptor)


def _read_worker_outcome(
    output_directory: Path,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = output_directory / "outcome.json"
    if not path.exists() and not path.is_symlink():
        return None
    outcome = _strict_json_object(
        _read_regular_file(
            path,
            limit=PROFILE_OUTCOME_LIMIT_BYTES,
            field="profiler outcome",
        ),
        field="profiler outcome",
    )
    if set(outcome) != set(expected) | {
        "status",
        "gpu_state",
        "completion_trusted",
        "reason_code",
    } or any(outcome.get(field) != value for field, value in expected.items()):
        raise ValueError("profiler outcome did not echo the trusted request")
    if (
        outcome.get("status")
        not in {"RUNNING", "SUCCESS", "FAILED", "HARD_FAILURE"}
        or outcome.get("gpu_state") not in {"NOT_STARTED", "STARTED", "COMPLETED"}
        or type(outcome.get("completion_trusted")) is not bool
        or not isinstance(outcome.get("reason_code"), str)
        or _PROFILE_TOKEN.fullmatch(outcome["reason_code"]) is None
    ):
        raise ValueError("profiler outcome state is invalid")
    if outcome["gpu_state"] == "COMPLETED" and not outcome["completion_trusted"]:
        raise ValueError("profiler completed outcome lacks trusted completion")
    return outcome


def _outcome_is_known(
    recipe: ProfileRecipe,
    outcome: Mapping[str, Any] | None,
) -> bool:
    if not recipe.requires_gpu:
        return True
    return bool(
        outcome is not None
        and (
            outcome.get("gpu_state") == "NOT_STARTED"
            or (
                outcome.get("gpu_state") == "COMPLETED"
                and outcome.get("completion_trusted") is True
            )
        )
    )


def _store_cas_object(root: Path, content: bytes, *, field: str) -> str:
    digest = hashlib.sha256(content).hexdigest()
    path = root / digest[:2] / digest[2:]
    for candidate in (root, path.parent):
        if candidate.exists() and candidate.is_symlink():
            raise ValueError(f"{field} CAS path traverses a symlink")
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    if path.exists() or path.is_symlink():
        existing = _read_regular_file(path, limit=max(len(content), 1), field=field)
        if existing != content:
            raise ValueError(f"{field} CAS collision")
        return f"sha256:{digest}"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{digest}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, path)
        temporary_name = None
        path.chmod(0o600)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return f"sha256:{digest}"


def _profile_budget(recipe: ProfileRecipe) -> BudgetAmount:
    return BudgetAmount(
        wall_ms=_PROFILE_BUDGET_CEILING_MS,
        gpu_ms=_PROFILE_BUDGET_CEILING_MS if recipe.requires_gpu else 0,
    )


def _actual_profile_budget(
    recipe: ProfileRecipe, *, started_monotonic: float
) -> BudgetAmount:
    elapsed_ms = max(0, math.ceil((time.monotonic() - started_monotonic) * 1000))
    bounded_ms = min(elapsed_ms, _PROFILE_BUDGET_CEILING_MS)
    return BudgetAmount(
        wall_ms=bounded_ms,
        gpu_ms=bounded_ms if recipe.requires_gpu else 0,
    )


def _profile_action_key(
    *,
    campaign_id: str,
    gate_id: str,
    recipe: ProfileRecipe,
    subject: _ProfileSubject,
) -> str:
    digest = canonical_sha256(
        {
            "kind": "bounded_profile_action_v1",
            "campaign_id": campaign_id,
            "gate_id": gate_id,
            "recipe_id": recipe.recipe_id,
            "case_id": recipe.case_id,
            "experiment_uid": subject.experiment_uid,
            "namespace_id": subject.namespace_id,
            "condition_digest": subject.condition_digest,
            "artifact_id": subject.artifact_id,
            "execution_environment_digest": (
                subject.execution_environment_digest
            ),
        }
    ).removeprefix("sha256:")
    return "bounded-profile-" + digest


def _profile_action_kind(recipe: ProfileRecipe) -> str:
    return (
        "PROFILE_HARDWARE_COUNTERS"
        if recipe.requires_gpu
        else "PROFILE_COMPILE_METADATA"
    )


def _require_campaign_scope(
    store: CampaignStore,
    *,
    campaign_id: str,
    namespace_id: str,
    require_running: bool,
) -> dict[str, Any]:
    try:
        campaign = store.get_campaign(campaign_id)
    except KeyError as exc:
        raise ValueError("profiling Campaign does not exist") from exc
    if campaign["namespace_id"] != namespace_id:
        raise ValueError("profiling Campaign belongs to another namespace")
    if require_running and campaign["status"] != CampaignStatus.RUNNING.value:
        raise ValueError("profiling requires the Campaign to be RUNNING")
    return campaign


def _require_campaign_subject(
    store: CampaignStore,
    *,
    campaign_id: str,
    subject: _ProfileSubject,
) -> dict[str, Any]:
    """Bind profiling evidence to the subject's durable Campaign child run."""

    if subject.campaign_id is None:
        raise ValueError(
            "Campaign profiling cannot use an experiment from an ordinary Run"
        )
    if subject.campaign_id != campaign_id:
        raise ValueError("profiling experiment belongs to another Campaign")
    child = store.get_child_run_by_controller_run_id(subject.run_id)
    if child is None:
        raise ValueError("profiling experiment run is not a Campaign child run")
    if (
        child.get("campaign_id") != subject.campaign_id
        or child.get("controller_run_id") != subject.run_id
    ):
        raise ValueError("profiling experiment Campaign/run identity mismatch")
    return child


def _require_reserved_profile_action(
    store: CampaignStore,
    *,
    campaign_id: str,
    budget_action_key: str,
    recipe: ProfileRecipe,
) -> dict[str, Any]:
    action = store.get_budget_action(
        campaign_id, idempotency_key=budget_action_key
    )
    if (
        action["status"] != "RESERVED"
        or action["action_kind"] != _profile_action_kind(recipe)
        or action["reserved"] != _profile_budget(recipe).to_dict()
        or action["actual"] is not None
    ):
        raise ValueError("profiling budget intent changed before execution")
    return action


def _require_profile_lease(
    store: CampaignStore,
    *,
    lease: ResourceLease,
) -> None:
    current = store.get_active_resource_lease(PROFILE_RESOURCE_ID)
    if (
        current is None
        or current.resource_id != PROFILE_RESOURCE_ID
        or current.campaign_id != lease.campaign_id
        or current.fencing_epoch != lease.fencing_epoch
        or current.status != "ACTIVE"
        or current.expires_epoch != lease.expires_epoch
        or current.expires_epoch <= time.time()
    ):
        raise ValueError("profiling GPU lease or fencing epoch changed")


def _pause_profile_campaign(
    store: CampaignStore,
    *,
    campaign_id: str,
    hard: bool,
    reason_code: str,
) -> None:
    target = (
        CampaignStatus.PAUSED_HARD_FAILURE
        if hard
        else CampaignStatus.PAUSED_UNKNOWN_OUTCOME
    )
    current = store.get_campaign(campaign_id)
    if current["status"] == target.value:
        return
    if not hard and current["status"] == CampaignStatus.PAUSED_HARD_FAILURE.value:
        return
    store.pause_campaign(
        campaign_id,
        status=target.value,
        reason=f"bounded profiling {reason_code}",
    )


def _quarantine_profile_outcome(
    store: CampaignStore,
    *,
    campaign_id: str,
    lease: ResourceLease | None,
    hard: bool,
    reason_code: str,
) -> None:
    quarantine_error: BaseException | None = None
    if lease is not None:
        try:
            store.release_resource(
                lease,
                quarantine=True,
                reason=f"bounded profiling {reason_code}",
            )
        except BaseException as exc:  # preserve the Campaign pause attempt
            quarantine_error = exc
    try:
        _pause_profile_campaign(
            store,
            campaign_id=campaign_id,
            hard=hard,
            reason_code=reason_code,
        )
    except BaseException:
        if quarantine_error is not None:
            raise quarantine_error
        raise
    if quarantine_error is not None:
        raise quarantine_error


def _reconcile_reserved_profile_action(
    config: ControllerConfig,
    store: CampaignStore,
    *,
    campaign_id: str,
    recipe: ProfileRecipe,
) -> None:
    """Turn a crash-left RESERVED action into a durable UNKNOWN stop.

    The host lock prevents a second CLI process from quarantining a profiler
    that is still executing.  Once that lock is free, an ACTIVE lease owned by
    this Campaign can only be an unknown result from the interrupted action.
    """

    if not recipe.requires_gpu:
        _pause_profile_campaign(
            store,
            campaign_id=campaign_id,
            hard=False,
            reason_code="reserved-action-unknown",
        )
        return
    try:
        with gpu_lock(config.controller_dir / "gpu1.lock"):
            lease = store.get_active_resource_lease(PROFILE_RESOURCE_ID)
            owned = (
                lease
                if lease is not None and lease.campaign_id == campaign_id
                else None
            )
            _quarantine_profile_outcome(
                store,
                campaign_id=campaign_id,
                lease=owned,
                hard=False,
                reason_code="reserved-action-unknown",
            )
    except RuntimeError as exc:
        raise ValueError(
            "reserved profiling action may still be running; replay is forbidden"
        ) from exc


def _command_has_fatal_gpu_marker(command: CommandResult) -> bool:
    combined = f"{command.stdout}\n{command.stderr}".lower()
    return any(marker in combined for marker in FATAL_GPU_MARKERS)


def run_bounded_profile(
    config: ControllerConfig,
    *,
    campaign_database: str | os.PathLike[str],
    campaign_id: str,
    gate_id: str,
    experiment_uid: str,
    namespace_id: str,
    execution_environment_digest: str,
    recipe_id: str,
    runner: _ProfileRunner | None = None,
    _gate_factory: Callable[
        [CampaignStore, str, SoakObservationCollector], _ProfilingGate
    ]
    | None = None,
) -> dict[str, Any]:
    """Collect one bounded trace and return only a compressed safe summary.

    ``_gate_factory`` is an injected-runner test seam.  Production callers use
    the real :class:`SoakGate` and trusted observation collector.
    """

    if not isinstance(config, ControllerConfig):
        raise TypeError("config must be ControllerConfig")
    if tuple(config.gpu_devices) != GPU1_DEVICES:
        raise ValueError("bounded profiling requires the exact trusted C500 devices")
    if (
        not isinstance(campaign_id, str)
        or _PROFILE_TOKEN.fullmatch(campaign_id) is None
    ):
        raise ValueError("campaign_id must be a stable bounded identifier")
    if not isinstance(gate_id, str) or _PROFILE_TOKEN.fullmatch(gate_id) is None:
        raise ValueError("gate_id must be a stable bounded identifier")
    if (
        not isinstance(experiment_uid, str)
        or _PROFILE_TOKEN.fullmatch(experiment_uid) is None
    ):
        raise ValueError("experiment_uid must be a stable bounded identifier")
    try:
        recipe = BUILTIN_PROFILE_RECIPES[recipe_id]
    except (KeyError, TypeError) as exc:
        raise ValueError("profile recipe is not a reviewed built-in recipe") from exc
    if runner is None:
        # Submit A is intentionally inert.  Tests may inject a runner to prove
        # all pre-launch and evidence paths without weakening production.
        require_active_profiler()

    runtime_root = _runtime_root(config)
    database = validate_production_campaign_database(
        campaign_database,
        runtime_root=runtime_root,
    )
    if database.is_symlink() or not database.is_file():
        raise ValueError("bounded profiling requires the live Campaign database")
    subject = _read_only_history_subject(
        state_dir=config.state_dir,
        experiment_uid=experiment_uid,
        namespace_id=namespace_id,
        execution_environment_digest=execution_environment_digest,
        recipe=recipe,
    )
    collector = SoakObservationCollector(config, campaign_database=database)
    profile_runner = runner or CommandRunner()
    with CampaignStore(database) as campaign_store:
        _require_campaign_scope(
            campaign_store,
            campaign_id=campaign_id,
            namespace_id=subject.namespace_id,
            require_running=False,
        )
        _require_campaign_subject(
            campaign_store,
            campaign_id=campaign_id,
            subject=subject,
        )
        budget_action_key = _profile_action_key(
            campaign_id=campaign_id,
            gate_id=gate_id,
            recipe=recipe,
            subject=subject,
        )
        try:
            existing_action = campaign_store.get_budget_action(
                campaign_id, idempotency_key=budget_action_key
            )
        except KeyError:
            existing_action = None
        if existing_action is not None:
            if existing_action["status"] == "RESERVED":
                _reconcile_reserved_profile_action(
                    config,
                    campaign_store,
                    campaign_id=campaign_id,
                    recipe=recipe,
                )
                raise ValueError(
                    "profiling action has an UNKNOWN outcome; replay is forbidden"
                )
            raise ValueError(
                "profiling action already has a terminal budget intent; replay is forbidden"
            )
        _require_campaign_scope(
            campaign_store,
            campaign_id=campaign_id,
            namespace_id=subject.namespace_id,
            require_running=True,
        )
        gate = (
            SoakGate(campaign_store, gate_id=gate_id, collector=collector)
            if _gate_factory is None
            else _gate_factory(campaign_store, gate_id, collector)
        )
        invariant_digest = _gate_authorization(gate)
        campaign_store.reserve_budget(
            campaign_id,
            idempotency_key=budget_action_key,
            action_kind=_profile_action_kind(recipe),
            amount=_profile_budget(recipe),
        )
        name_material = canonical_sha256(
            {
                "budget_action_key": budget_action_key,
                "invariant_digest": invariant_digest,
            }
        ).removeprefix("sha256:")
        container_name = "kar-profile-" + name_material[:40]
        started_monotonic = time.monotonic()
        lease: ResourceLease | None = None
        outcome_kind = "known"
        failure_reason_code = "pre-execution-failure"
        hard_failure = False

        def execute_action() -> dict[str, Any]:
            nonlocal lease, outcome_kind, failure_reason_code, hard_failure
            if recipe.requires_gpu:
                lease = campaign_store.acquire_resource(
                    campaign_id,
                    resource_id=PROFILE_RESOURCE_ID,
                    ttl_seconds=PROFILE_LEASE_TTL_SECONDS,
                )
            _require_campaign_scope(
                campaign_store,
                campaign_id=campaign_id,
                namespace_id=subject.namespace_id,
                require_running=True,
            )
            _require_campaign_subject(
                campaign_store,
                campaign_id=campaign_id,
                subject=subject,
            )
            _require_reserved_profile_action(
                campaign_store,
                campaign_id=campaign_id,
                budget_action_key=budget_action_key,
                recipe=recipe,
            )
            _gate_authorization_with_reserved_profile_intent(
                gate,
                collector=collector,
                store=campaign_store,
                gate_id=gate_id,
                expected_invariant_digest=invariant_digest,
            )
            _require_campaign_scope(
                campaign_store,
                campaign_id=campaign_id,
                namespace_id=subject.namespace_id,
                require_running=True,
            )
            _require_campaign_subject(
                campaign_store,
                campaign_id=campaign_id,
                subject=subject,
            )
            _require_reserved_profile_action(
                campaign_store,
                campaign_id=campaign_id,
                budget_action_key=budget_action_key,
                recipe=recipe,
            )
            if lease is not None:
                _require_profile_lease(campaign_store, lease=lease)

            with tempfile.TemporaryDirectory(
                prefix="kar-profile-output-"
            ) as temporary:
                output_directory = Path(temporary).resolve()
                output_directory.chmod(0o700)
                argv = _profile_argv(
                    config,
                    recipe=recipe,
                    subject=subject,
                    output_directory=output_directory,
                    container_name=container_name,
                    invariant_digest=invariant_digest,
                    budget_action_key=budget_action_key,
                    lease=lease,
                )
                outcome_kind = "unknown"
                failure_reason_code = "runner-outcome-unknown"
                worker_outcome: dict[str, Any] | None = None
                try:
                    command = profile_runner.run(
                        argv,
                        input_text=None,
                        timeout_sec=PROFILE_TIMEOUT_SECONDS,
                        max_output_bytes=PROFILE_OUTPUT_LIMIT_BYTES,
                        container_name=container_name,
                        docker_binary=config.docker_binary,
                    )
                except OSError as exc:
                    outcome_kind = "known"
                    failure_reason_code = "runner-start-failed"
                    raise ValueError("bounded profiler could not start") from exc
                if not recipe.requires_gpu:
                    outcome_kind = "known"
                expected_result: dict[str, Any] = {
                    "schema_version": PROFILE_COLLECTION_API_VERSION,
                    "worker_revision": PROFILER_WORKER_REVISION,
                    "profiler_profile_digest": PROFILER_PROFILE_DIGEST,
                    "profiler_image": PROFILER_IMAGE,
                    "recipe_id": recipe.recipe_id,
                    "case_id": recipe.case_id,
                    "experiment_uid": subject.experiment_uid,
                    "namespace_id": subject.namespace_id,
                    "condition_digest": subject.condition_digest,
                    "artifact_id": subject.artifact_id,
                    "candidate_sha256": subject.candidate_content_sha256,
                    "environment_digest": subject.execution_environment_digest,
                    "soak_invariant_digest": invariant_digest,
                    "campaign_id": subject.campaign_id,
                    "run_id": subject.run_id,
                    "budget_action_key": budget_action_key,
                }
                if lease is not None:
                    expected_result.update(
                        {
                            "resource_id": lease.resource_id,
                            "fencing_epoch": lease.fencing_epoch,
                        }
                    )
                worker_outcome = _read_worker_outcome(
                    output_directory,
                    expected=expected_result,
                )
                if recipe.requires_gpu and (
                    _command_has_fatal_gpu_marker(command)
                    or (
                        worker_outcome is not None
                        and worker_outcome.get("status") == "HARD_FAILURE"
                    )
                ):
                    hard_failure = True
                    failure_reason_code = "fatal-gpu-marker"
                    raise ValueError("bounded profiler reported a fatal GPU marker")
                if tuple(command.argv) != argv:
                    failure_reason_code = "runner-argv-mismatch"
                    outcome_kind = (
                        "known"
                        if _outcome_is_known(recipe, worker_outcome)
                        else "unknown"
                    )
                    raise ValueError("profile runner changed the fixed argv")
                if command.timed_out:
                    failure_reason_code = "runner-timeout"
                    outcome_kind = (
                        "known"
                        if _outcome_is_known(recipe, worker_outcome)
                        else "unknown"
                    )
                    raise ValueError("bounded profiler timed out")
                if command.output_limited:
                    failure_reason_code = "runner-output-limit"
                    outcome_kind = (
                        "known"
                        if _outcome_is_known(recipe, worker_outcome)
                        else "unknown"
                    )
                    raise ValueError("bounded profiler exceeded its output cap")
                if command.returncode != 0:
                    outcome_kind = (
                        "known"
                        if _outcome_is_known(recipe, worker_outcome)
                        else "unknown"
                    )
                    failure_reason_code = (
                        "runner-known-nonzero"
                        if outcome_kind == "known"
                        else "runner-nonzero-unknown"
                    )
                    raise ValueError("bounded profiler returned a non-zero status")
                if (
                    worker_outcome is None
                    or worker_outcome.get("status") != "SUCCESS"
                    or worker_outcome.get("completion_trusted") is not True
                ):
                    outcome_kind = (
                        "known"
                        if _outcome_is_known(recipe, worker_outcome)
                        else "unknown"
                    )
                    failure_reason_code = "worker-success-outcome-missing"
                    raise ValueError(
                        "bounded profiler returned without trusted success outcome"
                    )
                outcome_kind = "known"

                failure_reason_code = "evidence-validation-or-persistence-failed"
                result = _strict_json_object(
                    _read_regular_file(
                        output_directory / "result.json",
                        limit=PROFILE_RESULT_LIMIT_BYTES,
                        field="profiler result",
                    ),
                    field="profiler result",
                )
                if set(result) != set(expected_result) | {
                    "metrics",
                    "toolchain",
                    "trace_descriptor",
                } or any(
                    result.get(field) != value
                    for field, value in expected_result.items()
                ):
                    raise ValueError(
                        "profiler result did not echo the trusted request"
                    )
                metrics = _metric_summary(recipe, result)
                raw_trace = _read_regular_file(
                    output_directory / "raw.trace",
                    limit=PROFILE_RAW_TRACE_LIMIT_BYTES,
                    field="profiler raw trace",
                )
                if not raw_trace:
                    raise ValueError("profiler raw trace must not be empty")
                toolchain = _toolchain_summary(result)
                trace_descriptor = _trace_descriptor(
                    recipe, result, raw_trace=raw_trace
                )
                if recipe.requires_gpu and (
                    trace_descriptor.get("toolchain_digest")
                    != toolchain.get("digest")
                ):
                    raise ValueError(
                        "profiler trace and toolchain descriptors are inconsistent"
                    )

                confirmed_subject = _read_only_history_subject(
                    state_dir=config.state_dir,
                    experiment_uid=experiment_uid,
                    namespace_id=namespace_id,
                    execution_environment_digest=execution_environment_digest,
                    recipe=recipe,
                )
                if confirmed_subject != subject:
                    raise ValueError(
                        "History profiling subject changed during collection"
                    )
                _require_campaign_scope(
                    campaign_store,
                    campaign_id=campaign_id,
                    namespace_id=subject.namespace_id,
                    require_running=True,
                )
                _require_campaign_subject(
                    campaign_store,
                    campaign_id=campaign_id,
                    subject=subject,
                )
                _require_reserved_profile_action(
                    campaign_store,
                    campaign_id=campaign_id,
                    budget_action_key=budget_action_key,
                    recipe=recipe,
                )
                _gate_authorization_with_reserved_profile_intent(
                    gate,
                    collector=collector,
                    store=campaign_store,
                    gate_id=gate_id,
                    expected_invariant_digest=invariant_digest,
                )
                _require_campaign_scope(
                    campaign_store,
                    campaign_id=campaign_id,
                    namespace_id=subject.namespace_id,
                    require_running=True,
                )
                _require_campaign_subject(
                    campaign_store,
                    campaign_id=campaign_id,
                    subject=subject,
                )
                _require_reserved_profile_action(
                    campaign_store,
                    campaign_id=campaign_id,
                    budget_action_key=budget_action_key,
                    recipe=recipe,
                )
                if lease is not None:
                    _require_profile_lease(campaign_store, lease=lease)

                raw_trace_object_id = _store_cas_object(
                    config.controller_dir / "objects" / "sha256",
                    raw_trace,
                    field="private profiling trace",
                )
                resource_evidence = (
                    None
                    if lease is None
                    else {
                        "resource_id": lease.resource_id,
                        "fencing_epoch": lease.fencing_epoch,
                    }
                )
                evidence = {
                    "schema_version": PROFILE_COLLECTION_API_VERSION,
                    "kind": "bounded_profiling_evidence_v2",
                    "recipe_id": recipe.recipe_id,
                    "case_id": recipe.case_id,
                    "profiler_image": PROFILER_IMAGE,
                    "profiler_profile_digest": PROFILER_PROFILE_DIGEST,
                    "worker_revision": PROFILER_WORKER_REVISION,
                    "campaign_id": subject.campaign_id,
                    "run_id": subject.run_id,
                    "budget_action_key": budget_action_key,
                    "resource_lease": resource_evidence,
                    "experiment_uid": subject.experiment_uid,
                    "namespace_id": subject.namespace_id,
                    "condition_digest": subject.condition_digest,
                    "artifact_id": subject.artifact_id,
                    "candidate_content_sha256": subject.candidate_content_sha256,
                    "execution_environment_digest": (
                        subject.execution_environment_digest
                    ),
                    "stage": subject.stage,
                    "suite": subject.suite,
                    "replicate_kind": subject.replicate_kind,
                    "soak_gate_id": gate_id,
                    "soak_invariant_digest": invariant_digest,
                    "metrics": metrics,
                    "toolchain": toolchain,
                    "trace_descriptor": trace_descriptor,
                    "raw_trace": {
                        "object_id": raw_trace_object_id,
                        "byte_size": len(raw_trace),
                        "media_type": "application/octet-stream",
                    },
                    "advisory_only": True,
                    "promotion_effect": "none",
                    "baseline_effect": "none",
                }
                evidence_bytes = canonical_json_text(evidence).encode("utf-8")
                evidence_object_id = _store_cas_object(
                    config.state_dir / "objects" / "sha256",
                    evidence_bytes,
                    field="scientific profiling evidence",
                )

            failure_reason_code = "budget-or-lease-settlement-failed"
            campaign_store.settle_budget(
                campaign_id,
                idempotency_key=budget_action_key,
                actual=_actual_profile_budget(
                    recipe, started_monotonic=started_monotonic
                ),
            )
            if lease is not None:
                campaign_store.release_resource(lease)

            return {
                "schema_version": PROFILE_COLLECTION_API_VERSION,
                "command": "profile collect",
                "status": "SUCCESS",
                "recipe_id": recipe.recipe_id,
                "case_id": recipe.case_id,
                "profiler_image": PROFILER_IMAGE,
                "profiler_profile_digest": PROFILER_PROFILE_DIGEST,
                "worker_revision": PROFILER_WORKER_REVISION,
                "campaign_id": subject.campaign_id,
                "run_id": subject.run_id,
                "budget_action_key": budget_action_key,
                "resource_lease": resource_evidence,
                "subject": {
                    "experiment_uid": subject.experiment_uid,
                    "namespace_id": subject.namespace_id,
                    "condition_digest": subject.condition_digest,
                    "artifact_id": subject.artifact_id,
                    "campaign_id": subject.campaign_id,
                    "run_id": subject.run_id,
                    "execution_environment_digest": (
                        subject.execution_environment_digest
                    ),
                    "stage": subject.stage,
                    "suite": subject.suite,
                    "replicate_kind": subject.replicate_kind,
                },
                "authorization": {
                    "soak_gate_id": gate_id,
                    "current_invariant_snapshot_digest": invariant_digest,
                    "profiling_allowed": True,
                },
                "metrics": metrics,
                "toolchain": toolchain,
                "trace_descriptor": trace_descriptor,
                "raw_trace": {
                    "object_id": raw_trace_object_id,
                    "byte_size": len(raw_trace),
                    "media_type": "application/octet-stream",
                    "visibility": "controller-private-cas",
                },
                "evidence_object_id": evidence_object_id,
                "advisory_only": True,
                "promotion_effect": "none",
                "baseline_effect": "none",
            }

        try:
            if recipe.requires_gpu:
                with gpu_lock(config.controller_dir / "gpu1.lock"):
                    return execute_action()
            return execute_action()
        except BaseException as exc:
            if isinstance(exc, ControlledRuntimeError):
                outcome_kind = "known"
                failure_reason_code = "controller-gpu-lock-held"
            if outcome_kind == "known":
                try:
                    campaign_store.settle_budget(
                        campaign_id,
                        idempotency_key=budget_action_key,
                        actual=_actual_profile_budget(
                            recipe, started_monotonic=started_monotonic
                        ),
                    )
                    if lease is not None:
                        campaign_store.release_resource(lease)
                except BaseException as cleanup_error:
                    try:
                        _quarantine_profile_outcome(
                            campaign_store,
                            campaign_id=campaign_id,
                            lease=lease,
                            hard=False,
                            reason_code="known-failure-cleanup-failed",
                        )
                    except BaseException:
                        pass
                    raise ValueError(
                        "bounded profiler cleanup could not be persisted"
                    ) from cleanup_error
            else:
                try:
                    _quarantine_profile_outcome(
                        campaign_store,
                        campaign_id=campaign_id,
                        lease=lease,
                        hard=hard_failure,
                        reason_code=failure_reason_code,
                    )
                except BaseException as cleanup_error:
                    raise ValueError(
                        "bounded profiler UNKNOWN outcome could not be persisted"
                    ) from cleanup_error
            if isinstance(exc, ControlledRuntimeError):
                raise ValueError(
                    "GPU1 controller lock is already held"
                ) from exc
            raise


class _CanaryWorkerFailure(Exception):
    def __init__(self, cause: BaseException, *, known: bool, hard: bool = False):
        super().__init__(str(cause))
        self.cause = cause
        self.known = known
        self.hard = hard


def _current_deployment_profile_subject(
    config: ControllerConfig,
    *,
    campaign_id: str,
) -> tuple[_ProfileSubject, BaselineRef, dict[str, Any]]:
    """Resolve the one exact CURRENT bootstrap confirmation for deployment."""

    controller = ResearchController(config)
    pin = controller._deployment_pin()
    if pin is None:
        raise ValueError("profile image canary requires a deployment baseline pin")
    deployed = controller._best()
    environment = controller._resolved_execution_environment(
        CURRENT_RESEARCH_NAMESPACE
    )
    if not pin.execution_environment.is_resolved:
        raise ValueError("profile image canary requires resolved deployment evidence")
    try:
        pin.execution_environment.require_match(
            environment, context="profile image canary deployment"
        )
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    history_path = config.state_dir / "history.sqlite3"
    connection = sqlite3.connect(
        history_path.resolve().as_uri() + "?mode=ro",
        uri=True,
        isolation_level=None,
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(
            """
            SELECT experiment_uid, artifact_id, candidate_hash, status,
                   promotable, identity_json, result_json
            FROM experiments
            WHERE namespace_id = ? AND candidate_hash = ?
              AND status = 'SUCCESS'
            ORDER BY id
            """,
            (
                CURRENT_RESEARCH_NAMESPACE.namespace_id,
                pin.candidate_hash,
            ),
        ).fetchall()
    finally:
        connection.close()
    confirmations: list[tuple[str, ExperimentIdentity]] = []
    parsed: list[tuple[str, ExperimentIdentity, dict[str, Any], bool]] = []
    for row in rows:
        try:
            identity = ExperimentIdentity.from_value(
                _strict_json_object(
                    str(row["identity_json"]).encode("utf-8"),
                    field="CURRENT bootstrap identity",
                )
            )
            result = _strict_json_object(
                str(row["result_json"]).encode("utf-8"),
                field="CURRENT bootstrap result",
            )
        except (TypeError, ValueError):
            continue
        promotion = result.get("promotion")
        parsed.append(
            (
                str(row["experiment_uid"]),
                identity,
                result,
                bool(row["promotable"]),
            )
        )
        if (
            bool(row["promotable"])
            and identity.namespace == CURRENT_RESEARCH_NAMESPACE
            and identity.stage == "confirmation"
            and identity.suite == "full"
            and identity.replicate_kind == "confirmation"
            and identity.execution_environment == environment
            and str(identity.candidate_artifact_id) == row["artifact_id"]
            and row["artifact_id"] == deployed.artifact_id
            and result.get("status") == "SUCCESS"
            and result.get("evaluation_protocol_id")
            == CURRENT_C500_EVALUATION_PROTOCOL_ID
            and result.get("request_identity") == identity.to_dict()
            and isinstance(promotion, Mapping)
            and promotion.get("phase") == "confirmation"
            and promotion.get("confirmed") is True
        ):
            confirmations.append((str(row["experiment_uid"]), identity))
    if len(confirmations) != 1:
        raise ValueError(
            "deployment does not have one exact CURRENT bootstrap confirmation"
        )
    confirmation_uid, identity = confirmations[0]
    quick_matches = [
        (uid, candidate_identity)
        for uid, candidate_identity, candidate_result, promotable in parsed
        if (
            not promotable
            and candidate_identity.run_id == identity.run_id
            and candidate_identity.iteration == identity.iteration
            and candidate_identity.namespace == CURRENT_RESEARCH_NAMESPACE
            and candidate_identity.stage == "quick"
            and candidate_identity.suite == "quick"
            and candidate_identity.replicate_kind == "validation"
            and candidate_identity.candidate_artifact_id
            == identity.candidate_artifact_id
            and candidate_identity.baseline == identity.baseline
            and candidate_identity.execution_environment == environment
            and candidate_result.get("status") == "SUCCESS"
            and candidate_result.get("evaluation_protocol_id")
            == CURRENT_C500_EVALUATION_PROTOCOL_ID
            and candidate_result.get("request_identity")
            == candidate_identity.to_dict()
        )
    ]
    if len(quick_matches) != 1:
        raise ValueError(
            "CURRENT bootstrap confirmation has no exact quick case evidence"
        )
    experiment_uid, quick_identity = quick_matches[0]
    subject = _read_only_history_subject(
        state_dir=config.state_dir,
        experiment_uid=experiment_uid,
        namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
        execution_environment_digest=environment.digest,
        recipe=BUILTIN_PROFILE_RECIPES["metax-hardware-counters-v1"],
    )
    subject = _ProfileSubject(
        experiment_uid=subject.experiment_uid,
        namespace_id=subject.namespace_id,
        condition_digest=subject.condition_digest,
        artifact_id=subject.artifact_id,
        execution_environment_digest=subject.execution_environment_digest,
        campaign_id=campaign_id,
        run_id="profile-image-canary-" + experiment_uid.replace("-", "")[:32],
        stage=subject.stage,
        suite=subject.suite,
        replicate_kind=subject.replicate_kind,
        candidate_object_path=subject.candidate_object_path,
        candidate_content_sha256=subject.candidate_content_sha256,
    )
    baseline_ref = BaselineRef.create(
        namespace=CURRENT_RESEARCH_NAMESPACE,
        artifact_id=subject.artifact_id,
        source="campaign",
        revision="profile-image-canary-" + confirmation_uid,
        execution_environment=environment,
    )
    binding = {
        "deployment_evidence_digest": pin.evidence_digest,
        "deployment_git_commit": pin.git_commit,
        "deployment_candidate_hash": pin.candidate_hash,
        "current_confirmation_experiment_uid": confirmation_uid,
        "current_confirmation_identity": identity.to_dict(),
        "current_quick_experiment_uid": experiment_uid,
        "current_quick_identity": quick_identity.to_dict(),
        "execution_environment": environment.to_dict(),
    }
    return subject, baseline_ref, binding


def _canary_snapshot(
    *,
    subject: _ProfileSubject,
    baseline_ref: BaselineRef,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "kind": "PROFILE_IMAGE_CANARY",
        "namespace_id": subject.namespace_id,
        "artifact_id": subject.artifact_id,
        "candidate_sha256": subject.candidate_content_sha256,
        "baseline_ref": baseline_ref.to_dict(),
        "profiler_image": PROFILER_IMAGE,
        "profiler_profile_digest": PROFILER_PROFILE_DIGEST,
        "worker_revision": PROFILER_WORKER_REVISION,
        "recipes": list(PROFILE_RECIPE_IDS),
        "binding": dict(binding),
    }
    return json.loads(canonical_json_text(value))


def _canary_budget() -> BudgetAmount:
    return BudgetAmount(
        wall_ms=int(PROFILE_CANARY_WALL_SECONDS * 1000),
        gpu_ms=int(PROFILE_CANARY_GPU_SECONDS * 1000),
    )


def _require_canary_campaign(
    store: CampaignStore,
    *,
    campaign_id: str,
    snapshot: Mapping[str, Any],
    baseline_ref: BaselineRef,
) -> dict[str, Any]:
    campaign = store.get_campaign(campaign_id)
    expected_budget = _canary_budget().to_dict()
    if (
        campaign["namespace_id"] != CURRENT_RESEARCH_NAMESPACE.namespace_id
        or campaign["mode"] != CampaignMode.DISCOVERY.value
        or campaign["snapshot"] != dict(snapshot)
        or campaign["allow_staged_lineage"] is not False
        or any(
            int(campaign[field]) != expected_budget[budget_field]
            for field, budget_field in (
                ("max_candidates", "candidates"),
                ("max_wall_ms", "wall_ms"),
                ("max_gpu_ms", "gpu_ms"),
                ("max_tokens", "tokens"),
                ("max_cost_microusd", "cost_microusd"),
            )
        )
    ):
        raise ValueError("profile image canary Campaign identity or budget drifted")
    if store.list_child_runs(campaign_id):
        raise ValueError("profile image canary Campaign must not contain child runs")
    revisions = store.list_baseline_revisions(campaign_id)
    if (
        len(revisions) != 1
        or revisions[0]["revision_kind"] != "DEPLOYMENT_SEED"
        or revisions[0]["artifact_id"] != str(baseline_ref.artifact_id)
        or revisions[0]["baseline_ref"] != baseline_ref.to_dict()
        or campaign["active_baseline_revision_id"] != revisions[0]["id"]
    ):
        raise ValueError("profile image canary baseline revision drifted")
    return campaign


def _expected_worker_echo(
    *,
    recipe: ProfileRecipe,
    subject: _ProfileSubject,
    invariant_digest: str,
    budget_action_key: str,
    lease: ResourceLease | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": PROFILE_COLLECTION_API_VERSION,
        "worker_revision": PROFILER_WORKER_REVISION,
        "profiler_profile_digest": PROFILER_PROFILE_DIGEST,
        "profiler_image": PROFILER_IMAGE,
        "recipe_id": recipe.recipe_id,
        "case_id": recipe.case_id,
        "experiment_uid": subject.experiment_uid,
        "namespace_id": subject.namespace_id,
        "condition_digest": subject.condition_digest,
        "artifact_id": subject.artifact_id,
        "candidate_sha256": subject.candidate_content_sha256,
        "environment_digest": subject.execution_environment_digest,
        "soak_invariant_digest": invariant_digest,
        "campaign_id": subject.campaign_id,
        "run_id": subject.run_id,
        "budget_action_key": budget_action_key,
    }
    if lease is not None:
        result.update(
            {"resource_id": lease.resource_id, "fencing_epoch": lease.fencing_epoch}
        )
    return result


def _execute_canary_recipe(
    config: ControllerConfig,
    *,
    runner: _ProfileRunner,
    recipe: ProfileRecipe,
    subject: _ProfileSubject,
    invariant_digest: str,
    budget_action_key: str,
    lease: ResourceLease | None,
    container_suffix: str,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="kar-profile-canary-") as temporary:
        output_directory = Path(temporary).resolve()
        output_directory.chmod(0o700)
        container_name = "kar-profile-canary-" + container_suffix
        argv = _profile_argv(
            config,
            recipe=recipe,
            subject=subject,
            output_directory=output_directory,
            container_name=container_name,
            invariant_digest=invariant_digest,
            budget_action_key=budget_action_key,
            lease=lease if recipe.requires_gpu else None,
        )
        expected = _expected_worker_echo(
            recipe=recipe,
            subject=subject,
            invariant_digest=invariant_digest,
            budget_action_key=budget_action_key,
            lease=lease if recipe.requires_gpu else None,
        )
        try:
            command = runner.run(
                argv,
                input_text=None,
                timeout_sec=PROFILE_TIMEOUT_SECONDS,
                max_output_bytes=PROFILE_OUTPUT_LIMIT_BYTES,
                container_name=container_name,
                docker_binary=config.docker_binary,
            )
        except OSError as exc:
            raise _CanaryWorkerFailure(
                ValueError("bounded profiler could not start"), known=True
            ) from exc
        except BaseException as exc:
            outcome = None
            try:
                outcome = _read_worker_outcome(
                    output_directory, expected=expected
                )
            except ValueError:
                pass
            raise _CanaryWorkerFailure(
                exc,
                known=_outcome_is_known(recipe, outcome),
                hard=bool(outcome and outcome.get("status") == "HARD_FAILURE"),
            ) from exc
        outcome = _read_worker_outcome(output_directory, expected=expected)
        hard = _command_has_fatal_gpu_marker(command) or bool(
            outcome and outcome.get("status") == "HARD_FAILURE"
        )
        if hard:
            raise _CanaryWorkerFailure(
                ValueError("bounded profiler reported a fatal GPU marker"),
                known=False,
                hard=True,
            )
        known = _outcome_is_known(recipe, outcome)
        if tuple(command.argv) != argv:
            raise _CanaryWorkerFailure(
                ValueError("profile runner changed the fixed argv"), known=known
            )
        if command.timed_out:
            raise _CanaryWorkerFailure(
                ValueError("bounded profiler timed out"), known=known
            )
        if command.output_limited:
            raise _CanaryWorkerFailure(
                ValueError("bounded profiler exceeded its output cap"), known=known
            )
        if command.returncode != 0:
            raise _CanaryWorkerFailure(
                ValueError("bounded profiler returned a non-zero status"), known=known
            )
        if (
            outcome is None
            or outcome.get("status") != "SUCCESS"
            or outcome.get("completion_trusted") is not True
        ):
            raise _CanaryWorkerFailure(
                ValueError("bounded profiler has no trusted success outcome"),
                known=known,
            )
        result = _strict_json_object(
            _read_regular_file(
                output_directory / "result.json",
                limit=PROFILE_RESULT_LIMIT_BYTES,
                field="profiler result",
            ),
            field="profiler result",
        )
        if set(result) != set(expected) | {
            "metrics",
            "toolchain",
            "trace_descriptor",
        } or any(result.get(field) != value for field, value in expected.items()):
            raise _CanaryWorkerFailure(
                ValueError("profiler result did not echo the trusted request"),
                known=True,
            )
        raw_trace = _read_regular_file(
            output_directory / "raw.trace",
            limit=PROFILE_RAW_TRACE_LIMIT_BYTES,
            field="profiler raw trace",
        )
        if not raw_trace:
            raise _CanaryWorkerFailure(
                ValueError("profiler raw trace must not be empty"), known=True
            )
        try:
            toolchain = _toolchain_summary(result)
            trace_descriptor = _trace_descriptor(
                recipe, result, raw_trace=raw_trace
            )
            metrics = _metric_summary(recipe, result)
        except ValueError as exc:
            raise _CanaryWorkerFailure(exc, known=True) from exc
        raw_object = _store_cas_object(
            config.controller_dir / "objects" / "sha256",
            raw_trace,
            field="private profiler image canary trace",
        )
        return {
            "recipe_id": recipe.recipe_id,
            "case_id": recipe.case_id,
            "metrics": metrics,
            "toolchain": toolchain,
            "trace_descriptor": trace_descriptor,
            "raw_trace": {
                "object_id": raw_object,
                "byte_size": len(raw_trace),
                "visibility": "controller-private-cas",
            },
        }


def run_profile_image_doctor(
    config: ControllerConfig,
    *,
    campaign_database: str | os.PathLike[str],
    campaign_id: str,
    runner: _ProfileRunner | None = None,
) -> dict[str, Any]:
    """Qualify the active profiler image once before any soak can begin."""

    if not isinstance(config, ControllerConfig):
        raise TypeError("config must be ControllerConfig")
    require_active_profiler()
    if tuple(config.gpu_devices) != GPU1_DEVICES:
        raise ValueError("profiler image canary requires the exact C500 devices")
    _strict_token = _PROFILE_TOKEN.fullmatch(campaign_id) if isinstance(campaign_id, str) else None
    if _strict_token is None:
        raise ValueError("campaign_id must be a stable bounded identifier")
    runtime_root = _runtime_root(config)
    database = validate_production_campaign_database(
        campaign_database, runtime_root=runtime_root
    )
    action_key = "profile-image-canary-v1"
    profile_runner = runner or CommandRunner()
    started = time.monotonic()
    lease: ResourceLease | None = None
    with campaign_maintenance_fence(runtime_root):
        subject, baseline_ref, binding = _current_deployment_profile_subject(
            config, campaign_id=campaign_id
        )
        snapshot = _canary_snapshot(
            subject=subject, baseline_ref=baseline_ref, binding=binding
        )
        snapshot_digest = canonical_sha256(snapshot)
        with gpu_lock(config.controller_dir / "gpu1.lock"):
            with CampaignStore(database) as store:
                other = store.connection.execute(
                    """
                    SELECT id FROM campaigns
                    WHERE id != ? AND status NOT IN ('COMPLETED', 'CANCELLED')
                    ORDER BY id LIMIT 1
                    """,
                    (campaign_id,),
                ).fetchone()
                if other is not None:
                    raise ValueError(
                        "profile image canary is forbidden while another Campaign is active"
                    )
                try:
                    campaign = store.get_campaign(campaign_id)
                except KeyError:
                    campaign = store.create_campaign(
                        campaign_id=campaign_id,
                        namespace_id=CURRENT_RESEARCH_NAMESPACE.namespace_id,
                        mode=CampaignMode.DISCOVERY.value,
                        snapshot=snapshot,
                        budget_limit=_canary_budget(),
                        initial_baseline_ref=baseline_ref,
                        initial_policy_snapshot={
                            "kind": "PROFILE_IMAGE_CANARY",
                            "profiler_profile_digest": PROFILER_PROFILE_DIGEST,
                        },
                        allow_staged_lineage=False,
                    )
                _require_canary_campaign(
                    store,
                    campaign_id=campaign_id,
                    snapshot=snapshot,
                    baseline_ref=baseline_ref,
                )
                try:
                    existing = store.get_budget_action(
                        campaign_id, idempotency_key=action_key
                    )
                except KeyError:
                    existing = None
                if existing is not None:
                    if (
                        existing["status"] == "SETTLED"
                        and campaign["status"] == CampaignStatus.COMPLETED.value
                    ):
                        return {
                            "schema_version": PROFILE_COLLECTION_API_VERSION,
                            "command": "profile image-doctor",
                            "status": "ALREADY_COMPLETED",
                            "campaign_id": campaign_id,
                            "budget_action_key": action_key,
                            "profiler_image": PROFILER_IMAGE,
                            "profiler_profile_digest": PROFILER_PROFILE_DIGEST,
                            "advisory_only": True,
                        }
                    if existing["status"] != "RESERVED":
                        raise ValueError(
                            "profile image canary has a terminal incomplete action"
                        )
                    owned = store.get_active_resource_lease(PROFILE_RESOURCE_ID)
                    _quarantine_profile_outcome(
                        store,
                        campaign_id=campaign_id,
                        lease=(
                            owned
                            if owned is not None
                            and owned.campaign_id == campaign_id
                            else None
                        ),
                        hard=False,
                        reason_code="profile-image-canary-crash-left-reservation",
                    )
                    raise ValueError(
                        "profile image canary has an UNKNOWN outcome; replay is forbidden"
                    )
                if campaign["status"] == CampaignStatus.CREATED.value:
                    store.start_campaign(campaign_id)
                elif campaign["status"] != CampaignStatus.RUNNING.value:
                    raise ValueError("profile image canary Campaign is not runnable")
                store.reserve_budget(
                    campaign_id,
                    idempotency_key=action_key,
                    action_kind="PROFILE_IMAGE_CANARY",
                    amount=_canary_budget(),
                )
                lease = store.acquire_resource(
                    campaign_id,
                    resource_id=PROFILE_RESOURCE_ID,
                    ttl_seconds=PROFILE_LEASE_TTL_SECONDS,
                )
                try:
                    reports: list[dict[str, Any]] = []
                    for recipe_id in PROFILE_RECIPE_IDS:
                        confirmed, confirmed_ref, confirmed_binding = (
                            _current_deployment_profile_subject(
                                config, campaign_id=campaign_id
                            )
                        )
                        if (
                            confirmed != subject
                            or confirmed_ref != baseline_ref
                            or confirmed_binding != binding
                        ):
                            raise _CanaryWorkerFailure(
                                ValueError(
                                    "deployment profiling subject changed before Docker"
                                ),
                                known=True,
                            )
                        _require_canary_campaign(
                            store,
                            campaign_id=campaign_id,
                            snapshot=snapshot,
                            baseline_ref=baseline_ref,
                        )
                        _require_profile_lease(store, lease=lease)
                        report = _execute_canary_recipe(
                            config,
                            runner=profile_runner,
                            recipe=BUILTIN_PROFILE_RECIPES[recipe_id],
                            subject=subject,
                            invariant_digest=snapshot_digest,
                            budget_action_key=action_key,
                            lease=lease,
                            container_suffix=hashlib.sha256(
                                f"{campaign_id}:{recipe_id}".encode("utf-8")
                            ).hexdigest()[:32],
                        )
                        reports.append(report)
                    confirmed, confirmed_ref, confirmed_binding = (
                        _current_deployment_profile_subject(
                            config, campaign_id=campaign_id
                        )
                    )
                    if (
                        confirmed != subject
                        or confirmed_ref != baseline_ref
                        or confirmed_binding != binding
                    ):
                        raise _CanaryWorkerFailure(
                            ValueError(
                                "deployment profiling subject changed after Docker"
                            ),
                            known=True,
                        )
                    _require_profile_lease(store, lease=lease)
                    elapsed = min(
                        int(
                            math.ceil((time.monotonic() - started) * 1000)
                        ),
                        int(PROFILE_CANARY_WALL_SECONDS * 1000),
                    )
                    summary = {
                        "schema_version": PROFILE_COLLECTION_API_VERSION,
                        "command": "profile image-doctor",
                        "status": "READY",
                        "campaign_id": campaign_id,
                        "budget_action_key": action_key,
                        "profiler_image": PROFILER_IMAGE,
                        "profiler_profile_digest": PROFILER_PROFILE_DIGEST,
                        "worker_revision": PROFILER_WORKER_REVISION,
                        "subject": {
                            "experiment_uid": subject.experiment_uid,
                            "namespace_id": subject.namespace_id,
                            "artifact_id": subject.artifact_id,
                            "candidate_sha256": subject.candidate_content_sha256,
                            "execution_environment_digest": (
                                subject.execution_environment_digest
                            ),
                        },
                        "recipes": reports,
                        "advisory_only": True,
                        "promotion_effect": "none",
                        "baseline_effect": "none",
                    }
                    evidence_bytes = canonical_json_text(summary).encode("utf-8")
                    summary["evidence_object_id"] = _store_cas_object(
                        config.state_dir / "objects" / "sha256",
                        evidence_bytes,
                        field="scientific profiler image canary summary",
                    )
                    store.settle_budget(
                        campaign_id,
                        idempotency_key=action_key,
                        actual=BudgetAmount(
                            wall_ms=elapsed,
                            gpu_ms=min(
                                elapsed,
                                int(PROFILE_CANARY_GPU_SECONDS * 1000),
                            ),
                        ),
                    )
                    store.release_resource(lease)
                    store.finish_campaign(
                        campaign_id,
                        reason="profile image canary passed",
                    )
                    return summary
                except _CanaryWorkerFailure as failure:
                    if failure.known and not failure.hard:
                        elapsed = min(
                            int(
                                math.ceil((time.monotonic() - started) * 1000)
                            ),
                            int(PROFILE_CANARY_WALL_SECONDS * 1000),
                        )
                        store.settle_budget(
                            campaign_id,
                            idempotency_key=action_key,
                            actual=BudgetAmount(
                                wall_ms=elapsed,
                                gpu_ms=min(
                                    elapsed,
                                    int(PROFILE_CANARY_GPU_SECONDS * 1000),
                                ),
                            ),
                        )
                        store.release_resource(lease)
                        store.pause_campaign(
                            campaign_id,
                            status=CampaignStatus.PAUSED_OPERATOR.value,
                            reason="profiler image canary known failure",
                        )
                    else:
                        _quarantine_profile_outcome(
                            store,
                            campaign_id=campaign_id,
                            lease=lease,
                            hard=failure.hard,
                            reason_code=(
                                "profile-image-canary-hard-failure"
                                if failure.hard
                                else "profile-image-canary-unknown-outcome"
                            ),
                        )
                    raise failure.cause
                except BaseException:
                    try:
                        _quarantine_profile_outcome(
                            store,
                            campaign_id=campaign_id,
                            lease=lease,
                            hard=False,
                            reason_code="profile-image-canary-host-unknown",
                        )
                    except BaseException:
                        pass
                    raise


__all__ = [
    "BUILTIN_PROFILE_RECIPES",
    "DEFAULT_DEVICE_PATHS",
    "DEFAULT_LIBRARY_PATHS",
    "DEFAULT_TOOL_PROBES",
    "PROFILE_COLLECTION_API_VERSION",
    "PROFILE_DOCTOR_API_VERSION",
    "PROFILE_RECIPE_IDS",
    "PROFILER_ACTIVE",
    "PROFILER_IMAGE",
    "PROFILER_PROFILE_DIGEST",
    "PROFILER_WORKER_REVISION",
    "ProfileMetric",
    "ProfileRecipe",
    "ToolProbe",
    "run_bounded_profile",
    "run_profile_image_doctor",
    "run_profiling_doctor",
]
